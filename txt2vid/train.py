import random
import argparse

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

def load_data(video_dir=None, anno=None, vocab=None, batch_size=64, val=False, num_workers=4):
    # TODO
    from data import get_loader

    transform = transforms.Compose([transforms.Resize(FRAME_SIZE),
                                    transforms.Grayscale(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5],[0.5])])
                                    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                    #transforms.Normalize((0.485, 0.456, 0.406), 
                                    #                     (0.229, 0.224, 0.225))])
    return get_loader(video_dir, anno, vocab, transform, batch_size, shuffle=not val, num_workers=num_workers)

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 100000)
    status('Seed: %d' % args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    try:
        import os
        os.makedirs(args.out)
    except OSError:
        pass

    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)

    from util.pickle import load
    vocab = load(args.vocab)
    dataset = load_data(args.data, args.anno, vocab, batch_size=args.batch_size, val=False, num_workers=args.workers)

    # TODO: params
    txt_encoder = SentenceEncoder(embed_size=50,
                                  hidden_size=50, 
                                  encoding_size=100, 
                                  num_layers=2, 
                                  vocab_size=len(vocab)).to(device)

    discrim = model.Discrim(txt_encode_size=txt_encoder.encoding_size).to(device)
    gen = model.Generator(latent_size=(txt_encoder.encoding_size + 100)).to(device)
    #discrim = model.Discriminator().to(device)
    #gen = model.Generator().to(device)

    print(discrim)
    print(gen)
    print(txt_encoder)

    optimizerD = optim.Adam(discrim.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
    optimizerG = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    REAL_LABEL = 1
    FAKE_LABEL = 0
    SAMPLE_LATENT_SIZE = gen.latent_size - txt_encoder.encoding_size

    #fixed_noise = torch.randn(args.batch_size, SAMPLE_LATENT_SIZE, 1, 1, device=device)
    assert SAMPLE_LATENT_SIZE > 0

    REAL_LABELS = torch.full((args.batch_size,), REAL_LABEL, device=device, dtype=torch.float, requires_grad=False)
    FAKE_LABELS = torch.full((args.batch_size,), FAKE_LABEL, device=device, dtype=torch.float, requires_grad=False)
    REAL_LABELS = REAL_LABELS.view(REAL_LABELS.size(0), 1)
    FAKE_LABELS = FAKE_LABELS.view(FAKE_LABELS.size(0), 1)

    criteria = nn.BCELoss()

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))
    # TODO: param
    #embed = nn.Embedding(len(vocab), 128).to(device)

    def gen_step(fake=None, cap_fv=None, real_labels=None, last=True):
        gen.zero_grad()

        real_pred = discrim(vids=fake, sent=cap_fv)#.detach()
        loss = criteria(real_pred, real_labels)
        #loss = -torch.mean(output)
        loss.backward(retain_graph=not last)

        optimizerG.step()

        return loss#, output.mean().item()

    def discrim_step(videos=None, cap_fv=None, real_labels=None, fake_labels=None, last=True):
        # real example
        real_pred = discrim(vids=videos, sent=cap_fv.detach())
        loss_discrim_real = criteria(real_pred, real_labels)
        #loss_discrim_real.backward(retain_graph=not last)

        # fake example
        latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
        fake_inp = torch.cat((cap_fv.detach(), latent), dim=1)
        fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1, 1)

        fake = gen(fake_inp)
        fake_pred = discrim(vids=fake.detach(), sent=cap_fv.detach())
        loss_discrim_fake = criteria(fake_pred, fake_labels)
        #loss_discrim_fake.backward(retain_graph=not last)
        
        #loss = -(torch.mean(fake_pred) - torch.mean(real_pred))
        #loss.backward(retain_graph=not last)
        loss_discrim = loss_discrim_real + loss_discrim_fake
        loss_discrim.backward(retain_graph=not last)

        optimizerD.step()

        return loss_discrim, fake

    
    DISCRIM_STEPS = 1
    GEN_STEPS = 2

    # TODO: when to backprop for txt_encoder
    for epoch in range(args.epoch):
        print('epoch=', epoch + 1)

        gen_rolling = 0
        discrim_rolling = 0
        rolling = 1
        for i, (videos, captions, lengths) in enumerate(dataset):
            # TODO: hyper-params for GAN training
            videos = videos.to(device).permute(0, 2, 1, 3, 4)
            captions = captions.to(device)

            batch_size = videos.size(0)
            real_labels = REAL_LABELS[0:batch_size]
            fake_labels = FAKE_LABELS[0:batch_size]

            #print('cap=', vocab.to_words(captions[0]))

            txt_encoder.zero_grad()
            discrim.zero_grad()

            cap_fv = txt_encoder(captions, lengths)

            # discrim step
            for j in range(DISCRIM_STEPS):
                loss_discrim, fake = discrim_step(videos=videos,
                                                  cap_fv=cap_fv, 
                                                  real_labels=real_labels, 
                                                  fake_labels=fake_labels,
                                                  last=(j == DISCRIM_STEPS - 1))

            # generator
            for j in range(GEN_STEPS):
                loss_gen = gen_step(fake=fake, 
                                    cap_fv=cap_fv,
                                    real_labels=real_labels,
                                    last=(j == GEN_STEPS - 1))

            gen_rolling += loss_gen.item()
            discrim_rolling += loss_discrim.item()

            iteration = epoch*len(dataset) + i

            if iteration != 0 and iteration % 200 == 0:
                
                to_save = {
                    'gen': gen,
                    'discrim': discrim,
                    'txt': txt_encoder,
                    'optG': optimizerG,
                    'optD': optimizerD
                }

                torch.save(to_save, '%s/iter_%d_lossG_%.4f_lossD_%.4f' % (args.out, iteration, gen_rolling / rolling, discrim_rolling / rolling))

            if iteration % 5 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f)' % 
                        (epoch, args.epoch, i, len(dataset), 
                        loss_discrim.item(), discrim_rolling / rolling, 
                        loss_gen.item(), gen_rolling / (rolling)))

            rolling += 1

            if iteration % 20 == 0:
                gen_rolling = 0
                discrim_rolling = 0
                rolling = 1

                # TODO: output sentences
                to_save_real = videos[0]
                to_save_fake = fake[0]
                #print(captions[0])
                to_save_real = to_save_real.permute(1, 0, 2, 3)
                to_save_fake = to_save_fake.permute(1, 0, 2, 3)
                print('saving to %s' % args.out)
                #print(to_save_real.size())
                vutils.save_image(to_save_real, '%s/real_samples.png' % args.out, normalize=True, nrow=to_save_real.size(0))
                vutils.save_image(to_save_fake, '%s/fake_samples_epoch_%03d.png' % (args.out, epoch), normalize=True, nrow=to_save_fake.size(0))




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

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')

    parser.add_argument('--out', type=str, default='out', help='dir output path')

    args = parser.parse_args()
    main(args)
