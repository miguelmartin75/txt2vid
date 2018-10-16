import argparse
import random

import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from txt2vid.data import Vocab

import util.pickle
from util.log import status, warn, error

def main(args):
    seed = random.randint(1, 10000)
    if args.seed is not None:
        seed = args.seed

    status('Seed = %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    device = torch.device("cuda:0" if args.cuda else "cpu")
    models = torch.load(args.model)
    gen = models['gen'].to(device)
    txt_encoder = models['txt'].to(device)
    vocab = util.pickle.load(args.vocab)

    txt = util.pickle.load(args.input) 
    txt = [ s for k in txt for s in txt[k] ]
    txt = txt[random.randint(0, len(txt) - 1)]
    print(txt)
    txt = [ vocab(token) for token in vocab.tokenize(args.input) ]
    text = torch.Tensor(txt).to(device, dtype=torch.int64).view(1, -1)
    lengths = [len(txt)]

    txt = txt_encoder(text, lengths)
    noise = torch.randn(1, gen.latent_size - txt_encoder.encoding_size).to(device)

    gen_input = torch.cat((txt, noise), dim=1)
    gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1, 1)
    frames = gen(gen_input)
    frames = frames.permute(0, 2, 1, 3, 4)
    frames = frames.view(frames.size(1), frames.size(2), frames.size(3), frames.size(3))
    
    vutils.save_image(frames, args.out, nrow=frames.size(0), normalize=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--input', type=str, help='input file with strings', required=True)
    parser.add_argument('--vocab', type=str, help='vocab file', required=True)
    parser.add_argument('--model', type=str, help='model name', required=True)
    parser.add_argument('--out', type=str, help='output file', default='output.jpg')

    parser.add_argument('--seed', type=int, help='seed', default=None)

    args = parser.parse_args()
    main(args)
