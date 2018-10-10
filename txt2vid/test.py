import argparse
import random

import torch
import torchvision.utils as vutils

import util.pickle

def main(args):
    models = torch.load(args.out)
    gen = models['gen']
    txt_encoder = models['txt']
    vocab = load(args.vocab)

    txt = [ vocab(token) for token in vocab.tokenize ]

    seed = random.randint(1, 10000)
    if args.seed:
        seed = args.seed

    status('Seed = %d' % seed)
    random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    txt = torch.Tensor([txt]).to(device)
    txt = txt_encoder(txt)
    noise = torch.randn(2, 1).to(device)

    gen_input = torch.cat((txt, noise), dim=1)
    gen_input = gen_input.view(gen_input.size(0), gen_input.size(1), 1, 1, 1)
    frames = gen(gen_input)[0]
    
    vutils.save_image(frames, args.out, nrow=1, normalize=True)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--input', type=str, help='input string', required=True)
    parser.add_argument('--vocab', type=str, help='vocab file', required=True)
    parser.add_argument('--model', type=str, help='model name', required=True)
    parser.add_argument('--out', type=str, help='output file', default='output.jpg')

    parser.add_argument('--seed', type=int, help='seed', default=None)

    args = parser.parse_args()
    main(args)

