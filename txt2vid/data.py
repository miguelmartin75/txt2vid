from pathlib import Path

import torch
import torchvision.transforms as transforms
import torch.utils.data as data

import cv2
from PIL import Image

import numpy as np

def to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

def read_video_file(video_path, vid, cache=None):
    video = cv2.VideoCapture(str(video_path))

    idx = 0
    while(video.isOpened()):
        ok, frame = video.read()

        if not ok:
            break

        frame = to_pil(frame)

        if cache:
            path_to_save = '%s/%s/%d.jpg' % (cache, vid, idx)
            parent_path = Path(path_to_save).parent
            if not parent_path.exists():
                parent_path.mkdir()
            frame.save(path_to_save)

        idx += 1

        yield frame

    video.release()

class SynthDataset(data.Dataset):

    def get_video_path(self, vid_id):
        return '%s/%s.avi' % (self.video_dir, vid_id)

    def get_cache_video_path(self, vid_id):
        return '%s/%s' % (self.video_dir, vid_id)


    def __init__(self, video_dir=None, vocab=None, captions=None, transform=None, random_frames=0):
        from util.pickle import load
        self.video_dir = video_dir
        self.transform = transform
        self.random_frames = random_frames
        self.vocab = vocab

        captions = load(captions)

        self.video_ids = []
        self.captions = []
        self.missing = 0

        for vid in captions:
            path = Path(self.get_video_path(vid))
            if not path.exists():
                self.missing += 1
                continue

            for cap in captions[vid]:
                self.video_ids.append(vid)
                self.captions.append(cap)

        print("Missing: %d videos" % self.missing)

    def __getitem__(self, idx):
        vid = str(self.video_ids[idx])
        caption = self.captions[idx]

        cache = Path(self.get_cache_video_path(vid))
        assert(cache.exists())
        
        frames = []
        #if cache.exists():
        for frame_path in cache.iterdir():
            if frame_path.suffix != '.jpg' and frame_path.suffix != '.png':
                continue

            frames.append(int(frame_path.stem))

        frames.sort()

        #else:
        #    cache.mkdir()

        #    for frame in read_video_file(self.get_video_path(vid), vid=vid, cache=cache):
        #        if self.transform:
        #            frame = self.transform(frame)

        #        frames.append(frame)


        if self.random_frames == 0:
            new_frames = []
            # TODO: remove 32 constant
            factor=int(len(frames)/32)
            i = 0
            while i < 32:
                new_frames.append(frames[factor*i])
                i += 1
            frames = new_frames
        else:
            if len(frames) < self.random_frames:
                print('Video %s with %d frames' % (video_path, len(frames)))

            assert(len(frames) >= self.random_frames)

            perm = np.random.permutation(range(len(frames)))[0:self.random_frames]
            assert(len(perm) == self.random_frames)

            perm.sort()
            new_frames = []
            for i in perm:
                new_frames.append(frames[i])
            frames = new_frames
            assert(len(frames) == self.random_frames)

        def map_frame(path):
            path = '%s/%s.jpg' % (cache, path)
            img = Image.open(path)
            if self.transform:
                img = self.transform(img)
            return img

        frames = [ map_frame(path) for path in frames ]

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
        word = word.lower()
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def get_word(self, idx):
        return self.idx2word[idx]

    def __call__(self, word):
        word = word.lower()
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

    def to_words(self, tokens):
        result = ''
        for i, tok in enumerate(tokens):
            word = self.get_word(int(tok)) 
            if word != self.END and i != 0:
                result += ' '
            result += word

        return result

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

def get_loader(video_dir, captions, vocab, transform, batch_size, shuffle, num_workers, random_frames):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    dset = SynthDataset(video_dir=video_dir, vocab=vocab, captions=captions, transform=transform, random_frames=random_frames) 

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
