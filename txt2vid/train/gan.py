import random
import argparse
import numpy as np

import sys
import gc

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
from torchvision import transforms

import txt2vid.model as model
from txt2vid.data import Vocab
from txt2vid.model import SentenceEncoder

from util.log import status, warn, error
from util.pickle import load
from util.misc import gen_perm

def set_seed(seed):
    if seed is None:
        seed = random.randint(1, 100000)

    random.seed(seed)
    torch.manual_seed(seed)
    return seed

def setup_torch(use_cuda):
    cudnn.benchmark = True
    if torch.cuda.is_available() and not use_cuda:
        warn('cuda is available')

    return torch.device("cuda:0" if args.cuda else "cpu")

def main(args):
    seed = set_seed(args.seed)
    device = setup_torch(use_cuda=args.cuda)

    status('Seed: %d' % seed)
    status('Device set to: %s' % device)

    gen = create_object(args.G).to(device)
    discrims = [ create_object(d).to(device) for d in args.D ]

    # TODO
    #gen.init(args.init_method)
    #for discrim in discrims:
    #    discrim.init(args.init_method)

    status('Loading vocab from %s' % args.vocab)
    vocab = load(args.vocab)

    if args.sent_encode_path:
        status("Loading pre-trained sentence model from %s" % args.sent_encode_path)
        txt_encoder = torch.load(args.sent_encode_path)
        if 'txt' in txt_encoder:
            txt_encoder = txt_encoder['txt'].to(device)
        status("Sentence encode size = %d" % txt_encoder.encoding_size)
    else:
        status("Using random init sentence encoder")
        txt_encoder = create_object(args.sent, vocab_size=len(vocab))
        txt_encoder = SentenceEncoder(embed_size=args.word_embed,
                                      hidden_size=args.hidden_state, 
                                      encoding_size=args.sent_encode, 
                                      num_layers=args.txt_layers, 
                                      vocab_size=len(vocab)).to(device)

    gan = CondGan(gen=gen, discrims=discrims, cond_encoder=txt_encoder)
    
    from util.dir import ensure_exists
    ensure_exists(args.out)
    ensure_exists(args.out_samples)

    status('Loading data from %s' % args.data)
    dataset = load_data(video_dir=args.data, anno=args.anno, vocab=vocab, batch_size=args.batch_size, val=False, num_workers=args.workers, num_channels=args.num_channels, random_frames=args.random_frames)

    optD = optim.Adam([ { "params": p } for p in gan.discrims_params ], lr=args.lr, betas=(args.beta1, args.beta2))

    optG = optim.Adam([ 
        { "params": gen.parameters() }, 
    ], lr=args.lr, betas=(args.beta1, args.beta2))

    if args.model is not None:
        to_load = torch.load(args.model)

        gan.load_from_dict(to_load)
        if 'optD' in to_load:
            optD.load_state_dict(to_load['optD'])
        if 'optG' in to_load:
            optG.load_state_dict(to_load['optG'])

    print(discrim)
    print(gen)
    print(txt_encoder)

    print("Vocab Size %d" % len(vocab))
    print("Dataset len= %d (%d batches)" % (len(dataset)*args.batch_size, len(dataset)))

    params = TrainParams().read_from(args)
    print(device)
    # TODO: load loss dynamically
    losses = WassersteinGanLoss()
    gan_train(gan=gan, num_epoch=args.epoch, dataset=dataset, device=device, optD=optD, optG=optG, params=params, losses=losses)


if __name__ == '__main__':
    # TODO: args
    parser = argparse.ArgumentParser()
    # General setup stuff
    parser.add_argument('--seed', type=int, default=None, help='Seed to use')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')
    #parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')

    parser.add_argument('--out', type=str, default='out', help='dir output path')
    parser.add_argument('--out_samples', type=str, default='out_samples', help='dir output path')

    parser.add_argument('--frame_size', type=int, default=64, help='frame size')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    # Training params
    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--init_method', type=str, default='xavier', help='method for initialisation')

    parser.add_argument('--gen_steps', type=int, default=1, help='Number of generator steps to use per iteration')
    parser.add_argument('--G_lr', type=float, default=0.0001, help='learning rate for G')
    parser.add_argument('--G_beta1', type=float, default=0.5, help='beta1 for adam for G')
    parser.add_argument('--G_beta2', type=float, default=0.9, help='beta1 for adam for G')

    parser.add_argument('--discrim_steps', type=int, default=2, help='Number of discriminator steps to use per iteration')
    parser.add_argument('--D_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--D_beta1', type=float, default=0.5, help='beta1 for adam for D')
    parser.add_argument('--D_beta2', type=float, default=0.9, help='beta1 for adam for D')

    parser.add_argument('--weights', type=str, help='pretrained weights for G and D', default=None)
    parser.add_argument('--sent_weights', type=str, default=None, help='pretrained model for the sentence encoder')

    parser.add_argument('--data', type=str, help='video directory', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)

    # Model specific
    parser.add_argument('--G', type=str, default=None, help='G model', required=True)
    parser.add_argument('--D', type=str, default=None, nargs='+', help='D model(s)', required=True)
    parser.add_argument('--sent', type=str, default=None, help='Sentence model')

    args = parser.parse_args()
    main(args)
