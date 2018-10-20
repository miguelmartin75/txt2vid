import random
import argparse

import sys

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms

import txt2vid.model as model
#import txt2vid.videogan as model
from txt2vid.model import SentenceEncoder

from txt2vid.data import Vocab
from util.log import status, warn, error

FRAME_SIZE=64

def load_data(video_dir=None, anno=None, vocab=None, batch_size=64, val=False, num_workers=4, num_channels=3, random_frames=0):
    # TODO
    from data import get_loader

    if num_channels == 3:
        transform = transforms.Compose([transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.Resize((FRAME_SIZE, FRAME_SIZE)),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5])])

    return get_loader(video_dir, anno, vocab, transform, batch_size, shuffle=not val, num_workers=num_workers, random_frames=random_frames)

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 100000)
    status('Seed: %d' % args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    from util.dir import ensure_exists
    ensure_exists(args.out)
    ensure_exists(args.out_samples)

    import txt2vid.model
    txt2vid.model.USE_NORMAL_INIT=args.use_normal_init

    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)

    from util.pickle import load
    vocab = load(args.vocab)
    dataset = load_data(args.data, args.anno, vocab, batch_size=args.batch_size, val=False, num_workers=args.workers, num_channels=args.num_channels, random_frames=args.random_frames)

    # TODO: params
    txt_encoder = SentenceEncoder(embed_size=args.word_embed,
                                  hidden_size=args.hidden_state, 
                                  encoding_size=args.sent_encode, 
                                  num_layers=args.txt_layers, 
                                  vocab_size=len(vocab)).to(device)

    discrim = model.Discrim(txt_encode_size=txt_encoder.encoding_size,
                            num_channels=args.num_channels).to(device)
    gen = model.Generator(latent_size=(txt_encoder.encoding_size + args.latent_size), 
                          num_channels=args.num_channels).to(device)

    print(discrim)
    print(gen)
    print(txt_encoder)

    optimizerD = optim.Adam(discrim.parameters(), lr=args.lr, betas=(args.beta1, args.beta2), weight_decay=0.0005)
    optimizerG = optim.Adam([ { "params": gen.parameters() }, { "params": txt_encoder.parameters() } ], lr=args.lr, betas=(args.beta1, args.beta2))

    REAL_LABEL = 1
    FAKE_LABEL = 0
    SAMPLE_LATENT_SIZE = args.latent_size

    REAL_LABELS = torch.full((args.batch_size,), REAL_LABEL, device=device, dtype=torch.float, requires_grad=False)
    FAKE_LABELS = torch.full((args.batch_size,), FAKE_LABEL, device=device, dtype=torch.float, requires_grad=False)
    REAL_LABELS = REAL_LABELS.view(REAL_LABELS.size(0), 1)
    FAKE_LABELS = FAKE_LABELS.view(FAKE_LABELS.size(0), 1)

    criteria = nn.BCELoss()
    recon = nn.L1Loss()
    if args.recon_l2:
        recon = nn.MSELoss()

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))

    def gen_step(fake=None, cap_fv=None, real_labels=None, last=True, real_videos=None):
        gen.zero_grad()

        real_pred = discrim(vids=fake, sent=cap_fv)
        loss = criteria(real_pred, real_labels)
        if args.recon_lambda > 0:
            loss += args.recon_lambda * recon(fake, real_videos)

        loss.backward(retain_graph=not last)


        optimizerG.step()

        return loss

    def discrim_step(videos=None, cap_fv=None, real_labels=None, fake_labels=None, last=True, fake=None):
        # real example
        real_pred = discrim(vids=videos, sent=cap_fv.detach())
        loss_discrim_real = criteria(real_pred, real_labels)
        loss_discrim_real.backward(retain_graph=not last)

        # fake example
        fake_pred = discrim(vids=fake.detach(), sent=cap_fv.detach())
        loss_discrim_fake = criteria(fake_pred, fake_labels)
        loss_discrim_fake.backward(retain_graph=not last)
        
        loss_discrim = loss_discrim_real + loss_discrim_fake

        optimizerD.step()

        return loss_discrim

    
    DISCRIM_STEPS = args.discrim_steps
    GEN_STEPS = args.gen_steps

    # TODO: when to backprop for txt_encoder
    for epoch in range(args.epoch):
        print('epoch=', epoch + 1)

        gen_rolling = 0
        discrim_rolling = 0
        rolling = 1
        for i, (videos, captions, lengths) in enumerate(dataset):
            sys.stdout.flush()
            # TODO: hyper-params for GAN training
            videos = videos.to(device).permute(0, 2, 1, 3, 4)
            captions = captions.to(device)

            print(videos.size())
            print(captions.size())

            batch_size = videos.size(0)
            real_labels = REAL_LABELS[0:batch_size]
            fake_labels = FAKE_LABELS[0:batch_size]

            #print('cap=', vocab.to_words(captions[0]))

            txt_encoder.zero_grad()
            discrim.zero_grad()

            cap_fv = txt_encoder(captions, lengths)
            latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
            fake_inp = torch.cat((cap_fv.detach(), latent), dim=1)
            fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1, 1)

            fake = gen(fake_inp)

            # discrim step
            for j in range(DISCRIM_STEPS):
                loss_discrim = discrim_step(videos=videos,
                                            cap_fv=cap_fv, 
                                            real_labels=real_labels, 
                                            fake_labels=fake_labels,
                                            fake=fake,
                                            last=(j == DISCRIM_STEPS - 1))

            # generator
            for j in range(GEN_STEPS):
                if j != 0:
                    fake = gen(fake_inp)

                loss_gen = gen_step(fake=fake, 
                                    cap_fv=cap_fv,
                                    real_labels=real_labels,
                                    real_videos=videos,
                                    last=(j == GEN_STEPS - 1))

            gen_rolling += loss_gen.item()
            discrim_rolling += loss_discrim.item()

            iteration = epoch*len(dataset) + i

            if iteration != 0 and iteration % 100 == 0:
                
                to_save = {
                    'gen': gen,
                    'discrim': discrim,
                    'txt': txt_encoder,
                    'optG': optimizerG,
                    'optD': optimizerD
                }

                torch.save(to_save, '%s/iter_%d_lossG_%.4f_lossD_%.4f' % (args.out, iteration, gen_rolling / rolling, discrim_rolling / rolling))

            if iteration % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f)' % 
                        (epoch, args.epoch, i, len(dataset), 
                        loss_discrim.item(), discrim_rolling / rolling, 
                        loss_gen.item(), gen_rolling / (rolling)))

            rolling += 1

            if iteration % 50 == 0:
                gen_rolling = 0
                discrim_rolling = 0
                rolling = 1

                # TODO: output sentences
                to_save_real = videos
                to_save_fake = fake

                num_frames = to_save_real.size(2)

                #print(captions[0])
                to_save_real = to_save_real.permute(0, 2, 1, 3, 4)
                to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
                to_save_fake = to_save_fake.permute(0, 2, 1, 3, 4)
                to_save_fake = to_save_fake.view(-1, to_save_fake.size(2), to_save_fake.size(3), to_save_fake.size(4))

                print('saving to %s' % args.out)
                #print(to_save_real.size())
                vutils.save_image(to_save_real, '%s/real_samples.png' % args.out_samples, normalize=True, nrow=num_frames) #to_save_real.size(0))
                vutils.save_image(to_save_fake, '%s/fake_samples_epoch_%03d_iter_%06d.png' % (args.out_samples, epoch, iteration), normalize=True, nrow=num_frames)#to_save_fake.size(0))




if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--data', type=str, help='video directory', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')
    
    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')
    parser.add_argument('--discrim_steps', type=int, default=1, help='Number of discriminator steps to use per iteration')

    parser.add_argument('--word_embed', type=int, default=128, help='Dimensionality of each word (sentence model)')
    parser.add_argument('--hidden_state', type=int, default=256, help='Dimensionality of hidden state (sentence model)')
    parser.add_argument('--txt_layers', type=int, default=3, help='Number of layers in the sentence model')
    parser.add_argument('--sent_encode', type=int, default=256, help='Encoding for the sentence')

    parser.add_argument('--latent_size', type=int, default=100, help='Additional number of dimensions for random variable')

    parser.add_argument('--recon_lambda', type=float, default=0.1, help='Multiplier for reconstruction loss')
    parser.add_argument('--recon_l2', action='store_true', help='Use L2 loss for recon')

    parser.add_argument('--use_normal_init', action='store_true', help='Use normal init')

    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    args = parser.parse_args()
    main(args)
