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
from txt2vid.data import Vocab
from util.log import status, warn, error

FRAME_SIZE=64

def load_data(video_dir=None, anno=None, vocab=None, batch_size=64, val=False, num_workers=4):
    # TODO
    from data import get_loader

    transform = transforms.Compose([transforms.Resize(FRAME_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
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
    txt_encoder = model.SentenceEncoder(vocab_size=len(vocab)).to(device)
    discrim = model.Discrim(txt_encode_size=txt_encoder.encoding_size).to(device)
    gen = model.Generator(latent_size=256).to(device)

    print(discrim)
    print(gen)
    print(txt_encoder)

    optimizerD = optim.Adam(discrim.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    REAL_LABEL = 1
    FAKE_LABEL = 0
    SAMPLE_LATENT_SIZE = gen.latent_size - txt_encoder.encoding_size
    #fixed_noise = torch.randn(args.batch_size, SAMPLE_LATENT_SIZE, 1, 1, device=device)
    assert SAMPLE_LATENT_SIZE > 0

    REAL_LABELS = torch.full((args.batch_size,), REAL_LABEL, device=device, dtype=torch.float)
    FAKE_LABELS = torch.full((args.batch_size,), FAKE_LABEL, device=device, dtype=torch.float)
    REAL_LABELS = REAL_LABELS.view(REAL_LABELS.size(0), 1)
    FAKE_LABELS = FAKE_LABELS.view(FAKE_LABELS.size(0), 1)

    criteria = nn.BCELoss()

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))
    # TODO: param
    embed = nn.Embedding(len(vocab), 128).to(device)

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

            txt_encoder.zero_grad()
            discrim.zero_grad()

            cap_fv = txt_encoder(captions, lengths)

            # real example
            output = discrim(vids=videos, sent=cap_fv.detach())
            loss_discrim_real = criteria(output, real_labels)
            loss_discrim_real.backward()

            # fake example
            latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
            fake_inp = torch.cat((cap_fv.detach(), latent), dim=1)
            fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1, 1)
            fake = gen(fake_inp)
            output = discrim(vids=fake.detach(), sent=cap_fv.detach())
            loss_discrim_fake = criteria(output, fake_labels)
            loss_discrim_fake.backward()

            loss_discrim = loss_discrim_real + loss_discrim_fake

            optimizerD.step()

            # generator
            gen.zero_grad()

            output = discrim(vids=fake, sent=cap_fv)
            loss_gen = criteria(output, real_labels)
            loss_gen.backward()

            optimizerG.step()

            gen_rolling += loss_gen.item()
            discrim_rolling += loss_discrim.item()

            iteration = epoch*len(dataset) + i

            if iteration != 0 and iteration % 1000 == 0:
                
                to_save = {
                    'gen': gen,
                    'discrim': discrim,
                    'optG': optimizerG,
                    'optD': optimizerD
                }

                torch.save(to_save, '{}/iter_{}_lossG_{}_lossD_{}'.format(args.out, iteration, gen_rolling / rolling, discrim_rolling / rolling))

            if iteration % 20 == 0 and rolling != 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f)' % 
                        (epoch, args.epoch, i, len(dataset), 
                        loss_discrim.item(), discrim_rolling / rolling, 
                        loss_gen.item(), gen_rolling / rolling))

            rolling += 1

            if iteration % 100 == 0:
                gen_rolling = 1
                discrim_rolling = 1
                rolling = 1

                # TODO: output sentences

                #vutils.save_image(images, '%s/real_samples.png' % args.out, normalize=True)
                #fake = gen(fixed_noise)
                #vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.out, epoch), normalize=True)




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

    parser.add_argument('--out', type=str, default='out', help='dir output path')



    args = parser.parse_args()
    main(args)
