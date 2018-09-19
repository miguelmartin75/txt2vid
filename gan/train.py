import random
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from util.log import status, warn, error

import gan.models as gan

def load_data(args):
    if len(args.image_size) == 1:
        args.image_size = args.image_size[0]

    data = dsets.CIFAR10(root=args.data, download=True, transform=
                        transforms.Compose([transforms.Resize(args.image_size),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    return data


def main(args):
    if args.seed is None:
        args.seed = randint(1, 100000)
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

    dataset = load_data(args)
    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)

    device = torch.device("cuda:0" if args.cuda else "cpu")
    ngpu = int(args.ngpu)

    # TODO: params
    discrim = gan.Discrim().to(device)
    gen = gan.Generator().to(device)

    discrim.apply(gan.weights_init)
    gen.apply(gan.weights_init)

    print(discrim)
    print(gen)

    optimizerD = optim.Adam(discrim.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(gen.parameters(), lr=args.lr, betas=(args.beta1, 0.999))

    REAL_LABEL = 1
    FAKE_LABEL = 0
    LATENT_SIZE = gen.latent_size

    fixed_noise = torch.randn(args.batch_size, LATENT_SIZE, 1, 1, device=device)

    criteria = nn.BCELoss()
    #criteria = nn.NLLLoss()

    for epoch in range(args.epoch):
        for i, data in enumerate(dataloader):
            # real example
            discrim.zero_grad()

            real = data[0].to(device)
            batch_size = real.size(0)
            labels = torch.full((batch_size,), REAL_LABEL, device=device, dtype=torch.float)
            
            output = discrim(real)
            loss_discrim_real = criteria(output, labels)
            loss_discrim_real.backward()

            # fake example
            noise = torch.randn(batch_size, LATENT_SIZE, 1, 1, device=device)
            fake = gen(noise)
            labels.fill_(FAKE_LABEL)

            output = discrim(fake.detach())
            loss_discrim_fake = criteria(output, labels)
            loss_discrim_fake.backward()

            loss_discrim = loss_discrim_real + loss_discrim_fake

            optimizerD.step()

            # generator
            gen.zero_grad()
            labels.fill_(REAL_LABEL)

            output = discrim(fake)
            loss_gen = criteria(output, labels)
            loss_gen.backward()

            optimizerG.step()


            if i % 100 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % 
                        (epoch, args.epoch, i, len(dataloader), 
                        loss_discrim.item(), loss_gen.item()))

                vutils.save_image(real, '%s/real_samples.png' % args.out, normalize=True)
                fake = gen(fixed_noise)
                vutils.save_image(fake.detach(), '%s/fake_samples_epoch_%03d.png' % (args.out, epoch), normalize=True)



if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--data', type=str, help='dir data is located', required=True)
    parser.add_argument('--image_size', type=int, nargs='+', default=[64], help='image size')
    parser.add_argument('--batch_size', type=int, default=20, help='input batch size')

    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')

    parser.add_argument('--out', type=str, default='out', help='dir output path')



    args = parser.parse_args()
    main(args)
