# TODO: refactor
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
#import txt2vid.videogan as model
from txt2vid.model import SentenceEncoder

from util.log import status, warn, error
from util.pickle import load

def main(args):
    seed = random.randint(1, 10000)
    if args.seed is not None:
        seed = args.seed

    from util.dir import ensure_exists
    ensure_exists(args.out)

    status('Seed = %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    device = torch.device("cuda:0" if args.cuda else "cpu")
    models = torch.load(args.model)

    vocab = load(args.vocab)

    if args.sent_encode_path:
        txt_encoder = torch.load(args.sent_encode_path)
        if 'txt' in txt_encoder:
            txt_encoder = txt_encoder['txt'].to(device)
    else:
        txt_encoder = SentenceEncoder(embed_size=args.word_embed,
                                      hidden_size=args.hidden_state, 
                                      encoding_size=args.sent_encode, 
                                      num_layers=args.txt_layers, 
                                      vocab_size=len(vocab)).to(device)

    gen = model.Generator(latent_size=(txt_encoder.encoding_size + args.latent_size), num_channels=args.num_channels).to(device)

    SAMPLE_LATENT_SIZE = args.latent_size

    if args.model:
        to_load = torch.load(args.model)
        gen.load_state_dict(to_load['gen'])
        txt_encoder.load_state_dict(to_load['txt'])

    gen.eval()
    txt_encoder.eval()

    from txt2vid.train import load_data
    dataset = load_data(video_dir=args.data, anno=args.anno, vocab=vocab, batch_size=args.batch_size, val=False, num_workers=args.workers, num_channels=args.num_channels, random_frames=args.random_frames)

    for i, (videos, captions, lengths) in enumerate(dataset):
        videos = videos.to(device).permute(0, 2, 1, 3, 4)
        captions = captions.to(device)

        text = [ vocab.to_words(sent) for sent in captions ]
        for line in text:
            print(line)

        num_frames = videos.size(2)
        print('num_frames=', num_frames)
        to_save_real = videos
        to_save_real = to_save_real.permute(0, 2, 1, 3, 4)
        to_save_real = to_save_real.view(-1, to_save_real.size(2), to_save_real.size(3), to_save_real.size(4))
        vutils.save_image(to_save_real, '%s/real_samples.png' % args.out, normalize=True, nrow=num_frames)

        for j in range(10):
            batch_size = videos.size(0)
            _, _, cap_fv = txt_encoder(captions, lengths)
            cap_fv = cap_fv.detach()

            latent = torch.randn(batch_size, SAMPLE_LATENT_SIZE, device=device)
            fake_inp = torch.cat((cap_fv, latent), dim=1)
            fake_inp = fake_inp.view(fake_inp.size(0), fake_inp.size(1), 1, 1, 1)

            fake = gen(fake_inp)

            to_save_fake = fake

            to_save_fake = to_save_fake.permute(0, 2, 1, 3, 4).contiguous()
            to_save_fake = to_save_fake.view(-1, to_save_fake.size(2), to_save_fake.size(3), to_save_fake.size(4))

            print('saving to %s, %d' % (args.out, j))
            vutils.save_image(to_save_fake, '%s/fake_samples_%d.png' % (args.out, j), normalize=True, nrow=num_frames)

        sys.exit(0)


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--out', type=str, help='output dir', default='sampled_outputs')

    parser.add_argument('--seed', type=int, help='seed', default=None)

    parser.add_argument('--model', type=str, help='pretrained model', default=None, required=True)

    parser.add_argument('--data', type=str, help='video directory', required=True)
    parser.add_argument('--anno', type=str, help='annotation location', required=True)
    parser.add_argument('--vocab', type=str, help='vocab location', required=True)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')

    parser.add_argument('--sent_encode_path', type=str, default=None, help='Initial model for the sentence encoder')

    parser.add_argument('--word_embed', type=int, default=128, help='Dimensionality of each word (sentence model)')
    parser.add_argument('--hidden_state', type=int, default=256, help='Dimensionality of hidden state (sentence model)')
    parser.add_argument('--txt_layers', type=int, default=4, help='Number of layers in the sentence model')
    parser.add_argument('--sent_encode', type=int, default=256, help='Encoding for the sentence')

    parser.add_argument('--latent_size', type=int, default=100, help='Additional number of dimensions for random variable')

    parser.add_argument('--num_channels', type=int, default=1, help='number of channels in input')
    parser.add_argument('--random_frames', type=int, default=0, help='use random frames')

    parser.add_argument('--workers', type=int, default=0, help='num workers')

    args = parser.parse_args()
    main(args)
