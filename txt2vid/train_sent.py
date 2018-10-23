import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from txt2vid.model import SentenceEncoder, SentenceDecoder
from txt2vid.data import Vocab
from util.pickle import load
from util.log import status, warn, error

class SentenceDataset(torch.utils.data.Dataset):

    def __init__(self, vocab=None, sent_path=None):
        assert vocab is not None
        assert sent_path is not None

        self.vocab = vocab
        self.sent_path = sent_path

        temp = load(sent_path) 
        self.sents = [ s for x in temp for s in temp[x] ]

    def __len__(self):
        return len(self.sents)
    
    def __getitem__(self, idx):
        sent = [ self.vocab(token) for token in self.vocab.tokenize(self.sents[idx]) ]
        return torch.Tensor(sent)


def collate_fn(data):
    data.sort(key=lambda x: len(x), reverse=True)
    lengths = [ len(sent) for sent in data ]
    targets = torch.zeros(len(data), max(lengths)).long()
    for i, sent in enumerate(data):
        end = lengths[i]
        targets[i, :end] = sent[:end]
    return targets, lengths

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

    vocab = load(args.vocab)
    encoder = SentenceEncoder(vocab_size=len(vocab))
    decoder = SentenceDecoder()

    optimizer = optim.Adam([ { "params": encoder.parameters() }, { "params": decoder.parameters() } ], lr=args.lr, betas=(args.beta1, args.beta2))


    data = SentenceDataset(vocab=vocab, sent_path=args.sents)
    dataset = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)

    from txt2vid.metrics import RollingAvgLoss
    rolling_loss = RollingAvgLoss(window_size=100)

    iteration = 0
    for epoch in range(args.epoch):
        for i, (sent, lengths) in enumerate(dataset):
            encoder.zero_grad()
            decoder.zero_grad()

            encoded = encoder(sent, lengths=lengths)
            print(encoded.size())

            first_input = torch.Tensor(vocab(vocab.START))
            decoded = decoder(first_input, initial_state=encoded)

            loss = criteria(decoded, sent)
            loss.backward()

            rolling_loss.update(loss)

            optimizer.step()

            iteration += 1
            if iteration % 100 == 0:
                print('[%d/%d][%d/%d] Loss: %.4f' % (epoch, 
                                                     args.epoch, 
                                                     i, len(dataset),
                                                     rolling_loss.get()))

            if iteration % 1000 == 0:
                torch.save(to_save, '%s/iter_%d_loss_%.4f' % (args.out, iteration, rolling_loss.get()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sents', type=str, default=None, help='Input sentences', required=True)
    parser.add_argument('--vocab', type=str, default=None, help='vocab', required=True)
    parser.add_argument('--out', type=str, default=None, help='output path', required=True)

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')

    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    args = parser.parse_args()
    main(args)

