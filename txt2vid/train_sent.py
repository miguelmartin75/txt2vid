import sys
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from txt2vid.model import SentenceEncoder
from txt2vid.data import Vocab
from util.pickle import load
from util.log import status, warn, error

from tensorboardX import SummaryWriter


class SentenceDataset(torch.utils.data.Dataset):

    def __init__(self, vocab=None, sent_path=None, sents=None):
        assert vocab is not None

        self.vocab = vocab
        self.sent_path = sent_path

        if sent_path is not None:
            temp = load(sent_path) 
            self.sents = [ s for x in temp for s in temp[x] ]
        else:
            assert sents is not None
            self.sents = sents

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

def eval(split, seq2seq, device, vocab, debug=False):
    criteria = nn.CrossEntropyLoss(reduction='sum')
    loss = 0
    num_examples = 0
    with torch.no_grad():
        for i, (sent, lengths) in enumerate(split):
            batch_size = sent.size(0)

            sent = sent.to(device)

            _, hs, _ = seq2seq(sent, lengths=lengths)

            decoded, d_symbols = seq2seq.sample(true_inputs=sent, initial_hidden=hs, max_seq_len=lengths[0], teacher_force=False)

            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(sent, lengths, batch_first=True)
            targets , _ = pad_packed_sequence(packed, batch_first=True, total_length=lengths[0])

            predicted_words = vocab.to_words(d_symbols[-1])
            if debug:
                print('real idx=', sent[-1])
                print('predicted idx=', d_symbols[-1])
                print('real words=', vocab.to_words(sent[-1]))
                print('predicted words=', predicted_words)

            temp = criteria(decoded.permute(0, 2, 1), targets)
            if debug:
                print('loss=', temp)
            loss += temp
            num_examples += batch_size

    return loss / num_examples

def main(args):
    if args.seed is None:
        args.seed = random.randint(1, 100000)
    status('Seed: %d' % args.seed)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not args.cuda:
        warn('cuda is available, you probably want to use --cuda')

    device = torch.device("cuda:0" if args.cuda else "cpu")

    from util.dir import ensure_exists
    ensure_exists(args.out)

    vocab = load(args.vocab)
    seq2seq = SentenceEncoder(vocab_size=len(vocab), num_layers=4).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

    if args.model:
        status("Loading model")
        temp = torch.load(args.model)
        if 'txt' in temp:
            seq2seq = temp['txt']
        if 'optim' in temp:
            optimizer = temp['optim']

    train = []
    val = []
    test = []

    data = SentenceDataset(vocab=vocab, sent_path=args.sents)

    random.shuffle(data.sents)
    for i in range(len(data.sents)):
        r = random.uniform(0, 1)
        if r <= 0.8:
            train.append(data.sents[i])
        elif r <= 0.9:
            val.append(data.sents[i])
        else:
            test.append(data.sents[i])

    assert len(val) != 0
    assert len(test) != 0
    assert len(train) != 0

    data.sents = train
    train = data

    test = SentenceDataset(vocab=vocab, sents=test)
    val = SentenceDataset(vocab=vocab, sents=val)

    train_dataset = torch.utils.data.DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    test_dataset = torch.utils.data.DataLoader(test, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)
    val_dataset = torch.utils.data.DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    if args.test:
        status("Testing...")
        loss_val = eval(test_dataset, seq2seq, device, vocab, debug=True)
        print("Test loss = %.4f" % loss_val)
        sys.exit(0)

    from txt2vid.metrics import RollingAvgLoss
    rolling_loss = RollingAvgLoss(window_size=10)

    criteria = nn.CrossEntropyLoss()

    writer = SummaryWriter()

    iteration = 0
    val_loss = -1
    for epoch in range(args.epoch):
        for i, (sent, lengths) in enumerate(train_dataset):
            batch_size = sent.size(0)

            sent = sent.to(device)

            seq2seq.zero_grad()
            _, hidden_states, _ = seq2seq(sent, lengths=lengths)

            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            packed = pack_padded_sequence(sent, lengths, batch_first=True)
            targets, _ = pad_packed_sequence(packed, batch_first=True, total_length=lengths[0])

            teacher_force = random.uniform(0, 1) <= args.teacher_force
            decoded, d_symbols = seq2seq.sample(true_inputs=sent, initial_hidden=hidden_states, max_seq_len=lengths[0], teacher_force=teacher_force)

            loss = criteria(decoded.permute(0, 2, 1), targets)

            rolling_loss.update(float(loss))

            loss.backward()
            optimizer.step()

            writer.add_scalar('data/train_loss', loss, iteration)

            iteration += 1

            if iteration % 500 == 0:
                # TODO: fix eval
                val_loss = eval(val_dataset, seq2seq, device, vocab)#, criteria)
                writer.add_scalar('data/val_loss', val_loss, iteration)

                where_to_save = '%s/iter_%d_loss_%.4f_val_%.4f' % (args.out, iteration, rolling_loss.get(), val_loss)
                print("saving to: %s" % where_to_save)
                to_save = { 'optim': optimizer, 'txt': seq2seq }
                torch.save(to_save, where_to_save)

            if iteration % 50 == 0:
                #_, predicted = decoded.max(1)
                #print(decoded)
                predicted_words = vocab.to_words(d_symbols[0])
                print("predicted[0] = ", d_symbols[0])
                print("sent[0] = ", sent[0])
                print('real words=', vocab.to_words(sent[0]))
                print('predicted words=', predicted_words)
                #sys.exit(0)
                print('[%d/%d][%d/%d] Loss: %.4f (val = %.4f)' % (epoch, 
                                                     args.epoch, 
                                                     i, len(train_dataset), 
                                                     rolling_loss.get(), val_loss))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sents', type=str, default=None, help='Input sentences', required=True)
    parser.add_argument('--vocab', type=str, default=None, help='vocab', required=True)
    parser.add_argument('--out', type=str, default=None, help='output path', required=True)
    
    parser.add_argument('--model', type=str, default=None, help='model path')
    parser.add_argument('--test', type=int, default=0, help='to test or not to test')

    parser.add_argument('--epoch', type=int, default=5, help='number of epochs to perform')

    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.5')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta1 for adam. default=0.5')

    parser.add_argument('--seed', type=int, default=None, help='seed')
    parser.add_argument('--cuda', action='store_true', help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--workers', type=int, default=2, help='number of workers to help with loading/pre-processing data')

    parser.add_argument('--teacher_force', type=float, default=0.5, help='teacher force ratio')

    parser.add_argument('--max_seq_len', type=int, default=10, help='max sequence length')

    args = parser.parse_args()
    main(args)

