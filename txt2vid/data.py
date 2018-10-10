import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import cv2

import numpy as np

class SynthDataset(data.Dataset):

    def __init__(self, video_dir=None, vocab=None, captions=None, transform=None):
        from util.pickle import load
        self.video_dir = video_dir
        self.vocab = vocab
        self.transform = transform

        captions = load(captions)

        self.video_ids = []
        self.captions = []
        for vid in captions:
            for cap in captions[vid]:
                self.video_ids.append(vid)
                self.captions.append(cap)

    def __getitem__(self, idx):
        vid = self.video_ids[idx]
        caption = self.captions[idx]

        video_path = '%s/%s.avi' % (self.video_dir, vid)
        video = cv2.VideoCapture(video_path)

        frames = []
        while(video.isOpened()):
            ok, frame = video.read()

            if not ok:
                break

            from PIL import Image
            frame = Image.fromarray(frame)

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        video.release()

        caption = [self.vocab(token) for token in self.vocab.tokenize(caption)]
        if caption[-1] != self.vocab(self.vocab.END):
            caption.append(self.vocab(self.vocab.END))

        frames = torch.stack(frames)
        caption = torch.Tensor(caption)
        return frames, caption
    
    def __len__(self):
        return len(self.captions)
    
class Vocab(object):

    # don't change these
    START = '<start>'
    END = '<end>'
    UNKNOWN = '<unk>'
    PAD = '<pad>' # always zero

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        self.add_word(self.PAD)
        self.add_word(self.START)
        self.add_word(self.END)
        self.add_word(self.UNKNOWN)

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        return self.idx2word[idx]

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx[self.UNKNOWN]
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)

    def tokenize(self, sentence):
        yield self.START
        for word in sentence.split():
            if word[-1] == '.':
                yield word[0:-1]
                yield self.END
            else:
                yield word

def build_vocab(sentences):
    vocab = Vocab()
    for sent in sentences:
        for word in vocab.tokenize(sent):
            vocab.add_word(word)

    return vocab

def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (video, caption).

    We should build custom collate_fn rather than using default collate_fn, 
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    vids, captions = zip(*data)

    # Merge images (from tuple of 4D tensor to 5D tensor).
    vids = torch.stack(vids, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]        
    return vids, targets, lengths

def get_loader(video_dir, captions, vocab, transform, batch_size, shuffle, num_workers):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = SynthDataset(video_dir=video_dir, captions=captions, vocab=vocab, transform=transform) 

    data_loader = torch.utils.data.DataLoader(dataset=dset, 
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

def main(args):
    from util.pickle import load
    ex_to_sent = load(args.sents)
    sentences = [ s for x in ex_to_sent for s in ex_to_sent[x] ]

    vocab = build_vocab(sentences)

    print("saving...")
    with open(args.out, 'wb') as out_f:
        import pickle
        pickle.dump(vocab, out_f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sents', type=str, default=None, help='location of sentence pickle file')
    parser.add_argument('--out', type=str, default=None, help='output path', required=True)

    args = parser.parse_args()

    main(args)
