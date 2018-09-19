import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms

import txt2img.model as model
from coco.build_vocab import Vocabulary
from util.log import status, warn, error

RESIZE_SIZE=100
CROP_SIZE=64

def load_data(root=None, anno=None, vocab=None, batch_size=64, val=False, num_workers=4):
    from coco.data_loader import get_loader

    transform = transforms.Compose([transforms.Resize(RESIZE_SIZE), 
                                    transforms.RandomCrop(CROP_SIZE),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
                                    #transforms.Normalize((0.485, 0.456, 0.406), 
                                    #                     (0.229, 0.224, 0.225))])
    return get_loader(root, anno, vocab, transform, batch_size, shuffle=not val, num_workers=num_workers)

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
    discrim = model.Discrim().to(device)
    gen = model.Generator().to(device)
    txt_encoder = model.SentenceEncoder(vocab_size=len(vocab)).to(device)

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
    # TODO: param
    embed = nn.Embedding(len(vocab), 128).to(device)

    # TODO: when to backprop for txt_encoder
    for epoch in range(args.epoch):
        print('epoch=', epoch)

        gen_rolling = 0
        discrim_rolling = 0
        for i, (images, captions, lengths) in enumerate(dataset):
            # TODO: hyper-params for GAN training

            images = images.to(device)
            captions = captions.to(device)

            batch_size = images.size(0)

            real_labels = REAL_LABELS[0:batch_size]
            fake_labels = FAKE_LABELS[0:batch_size]

            txt_encoder.zero_grad()
            discrim.zero_grad()

            cap_fv = txt_encoder(captions, lengths)

            # real example
            output = discrim(img=images, sent=cap_fv.detach())
            loss_discrim_real = criteria(output, real_labels)
            loss_discrim_real.backward()

            # fake example
            latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
            fake_inp = torch.cat((cap_fv.detach(), latent), dim=1)
            fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1)
            fake = gen(fake_inp)
            output = discrim(img=fake.detach(), sent=cap_fv.detach())
            loss_discrim_fake = criteria(output, fake_labels)
            loss_discrim_fake.backward()

            loss_discrim = loss_discrim_real + loss_discrim_fake

            optimizerD.step()

            # generator
            gen.zero_grad()

            output = discrim(img=fake, sent=cap_fv)
            loss_gen = criteria(output, real_labels)
            loss_gen.backward()

            optimizerG.step()

            gen_rolling += loss_gen.item()
            discrim_rolling += loss_discrim.item()

            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f)' % 
                        (epoch, args.epoch, i + 1, len(dataset), 
                        loss_discrim.item(), discrim_rolling / 100, 
                        loss_gen.item(), gen_rolling / 100))

                gen_rolling = 0
                discrim_rolling = 0

                # TODO: output sentences
                vutils.save_image(images, '%s/real_samples.png' % args.out, normalize=True)
                #fake = gen(fixed_noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.out, epoch), normalize=True)


if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--data', type=str, help='img location', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--out', type=str, default='out', help='dir output path')



    args = parser.parse_args()
    main(args)
