import random
import argparse
import numpy as np

import sys
import gc

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms

from txt2vid.train.setup import setup
from txt2vid.gan.trainer import train, add_params_to_parser
from txt2vid.gan.losses import MixedGanLoss
from txt2vid.util.log import status, warn, error
from txt2vid.util.pick import load
from txt2vid.util.misc import gen_perm
from txt2vid.util.reflection import create_object
from txt2vid.util.torch.init import init
from txt2vid.gan.cond_gan import CondGan
from txt2vid.data import Vocab
import txt2vid.data as data

def main(args):
    seed, device = setup(args)

    status("Loading vocab from %s" % args.vocab)
    vocab = load(args.vocab)

    txt_encoder = None
    if not args.dont_use_sent:
        if args.sent_weights:
            status("Loading pre-trained sentence model from %s" % args.sent_weights)
            txt_encoder = torch.load(args.sent_weights)
            if 'txt' in txt_encoder:
                txt_encoder = txt_encoder['txt'].to(device)
        else:
            status("Using random init sentence encoder")
            txt_encoder = create_object(args.sent, vocab_size=len(vocab)).to(device)
            if args.sent_init_method is None:
                args.sent_init_method = args.init_method
            status("Initialising txt_encoder with %s" % args.sent_init_method)
            init(txt_encoder, init_method=args.sent_init_method)

        assert(txt_encoder is not None)

    cond_dim = 0
    if txt_encoder is not None:
        cond_dim = txt_encoder.encoder.encoding_size
        status("Sentence encode size = %d" % cond_dim)
    else:
        status("Not using sentence encoder")

    gen = create_object(args.G, cond_dim=cond_dim).to(device)
    discrims = [ create_object(d, cond_dim=cond_dim).to(device) for d in args.D ]

    init(gen, init_method=args.init_method)
    status("Initialising gen with %s" % args.init_method)
    for i, discrim in enumerate(discrims):
        discrim_name = "discrim-%d" % i
        if args.D_names is not None and len(args.D_names) > i:
            discrim_name = args.D_names[i]
        status("Initialising discrim %s with %s" % (discrim_name, args.init_method))
        init(discrim, init_method=args.init_method)

    sample_mapping = None
    if args.M:
        status("Creating sample mapping: %s" % args.M)
        sample_mapping = create_object(args.M).to(device)
        status("Initialising sample_mapping %s with %s" % (args.M, args.init_method))
        init(sample_mapping, init_method=args.init_method)

    gan = CondGan(gen=gen, discrims=discrims, cond_encoder=txt_encoder, sample_mapping=sample_mapping, discrim_names=args.D_names, discrim_lambdas=args.D_lambdas)

    D_params = [ { "params": p } for p in gan.discrims_params ]
    G_params = [ { "params": gen.parameters() } ]

    if args.end2end and txt_encoder is not None:
        # TODO: should I update txt_encoder in generator or discriminator exclusively?
        D_params.append({"params": txt_encoder.parameters()})
        G_params.append({"params": txt_encoder.parameters()})

    if args.sgd:
        status("Using SGD")
        optD = optim.SGD(D_params, lr=args.D_lr, momentum=args.D_beta1)
        optG = optim.SGD(G_params, lr=args.G_lr, momentum=args.G_beta1)
    else:
        status("Using Adam")
        optD = optim.Adam(D_params, lr=args.D_lr, betas=(args.D_beta1, args.D_beta2))
        optG = optim.Adam(G_params, lr=args.G_lr, betas=(args.G_beta1, args.G_beta2))

    if args.weights is not None:
        to_load = torch.load(args.weights)

        gan.load_from_dict(to_load)
        if 'optD' in to_load:
            optD.load_state_dict(to_load['optD'])
        if 'optG' in to_load:
            optG.load_state_dict(to_load['optG'])

    status('Loading data from %s' % args.data)

    transform = data.default_transform(frame_size=[args.frame_sizes[-1]], num_channels=args.num_channels)
    dset = create_object(args.data, vocab=vocab, anno=args.anno, transform=transform)

    #if args.multi_scale:
    #    from txt2vid.models.tganv2.dset import MultiScaleDataset
    #    dset = MultiScaleDataset(dset=dset)

    dataset = data.get_loader(dset=dset, batch_size=args.batch_size, val=False, num_workers=args.workers, has_captions=args.anno is not None)

    print("D optim=", optD)
    print("G optim=", optG)
    print('gen=', gen)
    print('txt=', txt_encoder)
    for filepath, name, discrim in zip(args.D, gan.discrim_names, gan.discrims):
        print("%s (%s)" % (filepath, name))
        print(discrim)

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))

    if args.G_loss is None:
        args.G_loss = args.D_loss

    losses = MixedGanLoss(g_loss=create_object(args.G_loss), d_loss=create_object(args.D_loss))

    train(gan=gan, num_epoch=args.epochs, dataset=dataset, device=device, optD=optD, optG=optG, params=args, losses=losses, vocab=vocab, channel_first=not args.sequence_first, end2end=args.end2end)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    add_params_to_parser(parser)

    # General setup stuff
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    
    # input
    parser.add_argument('--frame_sizes', type=int, nargs='+', default=[64], help='frame sizes')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    # Training params
    parser.add_argument('--epochs', type=int, default=5, help='number of epochs to perform')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--init_method', type=str, default='xavier', help='method for initialisation')

    parser.add_argument('--G_loss', type=str, default=None, help='class for loss for G (default is None => same as D)')
    parser.add_argument('--G_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--G_beta1', type=float, default=0.5, help='beta1 for adam for G')
    parser.add_argument('--G_beta2', type=float, default=0.9, help='beta1 for adam for G')

    parser.add_argument('--D_loss', type=str, default='txt2vid.gan.losses.VanillaGanLoss', help='class for loss for D (class-name or JSON file)')
    parser.add_argument('--D_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--D_beta1', type=float, default=0.5, help='beta1 for adam for D')
    parser.add_argument('--D_beta2', type=float, default=0.9, help='beta1 for adam for D')

    parser.add_argument('--weights', type=str, help='pretrained weights for G and D', default=None)
    parser.add_argument('--sent_weights', type=str, default=None, help='pretrained model for the sentence encoder')

    parser.add_argument('--data', type=str, help='dataset function (use json to provide arguments)', required=True)
    parser.add_argument('--anno', type=str, help='annotation', default=None)
    parser.add_argument('--vocab', type=str, help='vocab', default=None)

    # Model specific
    parser.add_argument('--M', type=str, default=None, help='mapping for x')
    parser.add_argument('--G', type=str, default=None, help='G model', required=True)
    parser.add_argument('--D', type=str, default=None, nargs='+', help='D model(s)', required=True)
    parser.add_argument('--D_names', type=str, default=None, nargs='+', help='D model names')
    parser.add_argument('--D_lambdas', type=float, default=None, nargs='+', help='associated lambdas for each discriminator (used for weighted sum in loss); if None will assume each are associated with an equal weight, i.e. equivalent to a .mean()')
    parser.add_argument('--sent', type=str, default=None, help='Sentence model')
    parser.add_argument('--sent_init_method', type=str, default=None, help='Sentence model init, by default will do the same as regular init_method')

    parser.add_argument('--dont_use_sent', action='store_true', default=False, help='uses the sentence model (i.e. non-conditional case)')
    parser.add_argument('--end2end', action='store_true', default=False, help='trains the model end2end, i.e. the sentence (cond) model is not frozen')
    parser.add_argument('--sgd', action='store_true', default=False, help='use SGD with momentum instead of adam (uses beta1)')
    parser.add_argument('--sequence_first', action='store_true', default=False, help='puts sequence first before channels in input to models')
    parser.add_argument('--debug', action='store_true', default=False, help='debug logging')

    args = parser.parse_args()
    main(args)
