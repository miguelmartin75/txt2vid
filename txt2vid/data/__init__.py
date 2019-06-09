import random
from pathlib import Path

import numpy as np

import cv2
from PIL import Image

import torch
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch.utils.data as data

from txt2vid.util.pick import load

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

#import nvidia.dali.plugin_manager as plugin_manager
#plugin_manager.load_library('./dali_custom_ops/build/libmyreader.so')

#class Caffe2Pipeline(Pipeline):
#    def __init__(self, db_folder=None, resize_size=None, crop_size=None, batch_size=None, num_threads=None, device_id=None, ngpu=None):
#        super(Caffe2Pipeline, self).__init__(batch_size,
#                                         num_threads,
#                                         device_id)
#        if ngpu is None:
#            ngpu = torch.cuda.device_count()
#
#        self.input = ops.MyReaderOp(path = db_folder, num_shards=ngpu)
#        #self.decode= ops.nvJPEGDecoder(device = "mixed", output_type = types.RGB)
#        #self.cmnp = ops.CropMirrorNormalize(device = "gpu",
#        #                                    output_dtype = types.FLOAT,
#        #                                    crop = (224, 224),
#        #                                    image_type = types.RGB,
#        #                                    mean = [0., 0., 0.],
#        #                                    std = [1., 1., 1.])
#        #self.uniform = ops.Uniform(range = (0.0, 1.0))
#        self.iter = 0
#
#    def define_graph(self):
#        self.videos, self.captions = self.input(name="Reader")
#        return self.videos, self.captions
#
#    def iter_setup(self):
#        pass
#
#
#class VideoReaderPipeline(Pipeline):
#    def __init__(self, batch_size, sequence_length, num_threads, device_id, files, crop_size):
#        super(VideoReaderPipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
#        self.reader = ops.VideoReader(device="gpu", filenames=files, sequence_length=sequence_length, normalized=False,
#                                     random_shuffle=True, image_type=types.RGB, dtype=types.UINT8, initial_fill=16)
#        self.crop = ops.CropCastPermute(device="gpu", crop=crop_size, output_layout=types.NHWC, output_dtype=types.FLOAT)
#        self.uniform = ops.Uniform(range=(0.0, 1.0))
#        self.transpose = ops.Transpose(device="gpu", perm=[3, 0, 1, 2])
#
#    def define_graph(self):
#        input = self.reader(name="Reader")
#        cropped = self.crop(input, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
#        output = self.transpose(cropped)
#        return output
#
#class DALILoader():
#    def __init__(self, batch_size, file_root, sequence_length, crop_size):
#        container_files = os.listdir(file_root)
#        container_files = [file_root + '/' + f for f in container_files]
#        self.pipeline = VideoReaderPipeline(batch_size=batch_size,
#                                            sequence_length=sequence_length,
#                                            num_threads=2,
#                                            device_id=0,
#                                            files=container_files,
#                                            crop_size=crop_size)
#        self.pipeline.build()
#        self.epoch_size = self.pipeline.epoch_size("Reader")
#        self.dali_iterator = pytorch.DALIGenericIterator(self.pipeline,
#                                                         ["data"],
#                                                         self.epoch_size,
#                                                         auto_reset=True)
#    def __len__(self):
#        return int(self.epoch_size)
#    def __iter__(self):
#        return self.dali_iterator.__iter__()

def to_pil(cv2_img):
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(cv2_img)

# TODO: fix this shit, it's gross
def read_video_file(video_path, convert_to_pil=True):
    video = cv2.VideoCapture(str(video_path))

    while video.isOpened():
        ok, frame = video.read()

        if not ok:
            break

        if convert_to_pil:
            frame = to_pil(frame)

        yield frame

    video.release()

def pick_frames(from_frames, random=False, num_frames=16):
    frames = None
    if not random:
        new_frames = []
        factor=int(len(from_frames)/num_frames)
        i = 0
        while i < num_frames:
            new_frames.append(from_frames[factor*i])
            i += 1
        frames = new_frames
    else:
        assert(len(frames) >= self.random_frames)

        perm = np.random.permutation(range(len(frames)))[0:num_frames]
        assert(len(perm) == num_frames)

        perm.sort()
        new_frames = []
        for i in perm:
            new_frames.append(from_frames[i])
        #frames = new_frames
        assert(len(frames) == self.random_frames)
    return new_frames

class Dataset(data.Dataset):

    def get_video_path(self, vid_id):
        return "%s/%s" % (self.video_dir, vid_id)
        #return '%s/%s.avi' % (self.video_dir, vid_id)

    def get_cache_video_path(self, vid_id):
        return '%s/%s' % (self.video_dir, vid_id)

    def __init__(self, video_dir=None, vocab=None, captions=None, transform=None, random_frames=0, num_frames=16, use_lmdb=False):
        self.video_dir = video_dir
        self.transform = transform
        self.random_frames = random_frames
        self.vocab = vocab
        self.num_frames = num_frames

        captions = load(captions)

        self.video_ids = []
        self.captions = []
        self.missing = 0

        self.use_lmdb = use_lmdb
        if use_lmdb:
            import lmdb
            self.env = lmdb.open(video_dir + '.lmdb', readonly=True)

        for vid in captions:
            try:
                path = Path(self.get_video_path(vid))
                if not path.exists():
                    self.missing += 1
                    continue
            except:
                self.missing += 1
                continue

            for cap in captions[vid]:
                self.video_ids.append(vid)
                self.captions.append(cap)

        print("Missing: %d videos" % self.missing)

    def __getitem__(self, idx):
        vid = str(self.video_ids[idx])
        caption = self.captions[idx]

        if self.use_lmdb:
            # This is NOT fast
            with self.env.begin() as txn:
                raw_datum = txn.get(vid.encode())
            from txt2vid.data.create_cache import get_frames
            frames = get_frames(raw_datum)
            frames = torch.Tensor(frames)
        else:
            cache = Path(self.get_cache_video_path(vid))
            assert(cache.exists())
            
            frames = []
            for frame_path in cache.iterdir():
                if frame_path.suffix != '.jpg' and frame_path.suffix != '.png':
                    continue

                frames.append(int(frame_path.stem))

            frames.sort()

            # IDK why I'm doing it like this lol whatever
            num_frames = self.random_frames
            random_frames = True
            if num_frames == 0:
                random_frames = False
                # TODO: remove constant
                num_frames = 16

            frames = pick_frames(frames, random=random_frames, num_frames=num_frames)

            flip_horiz = random.random() < 0.5

            def map_frame(path):
                path = '%s/%s.jpg' % (cache, path)
                img = Image.open(path)
                if flip_horiz:
                    img = F.hflip(img)

                if self.transform:
                    img = self.transform(img)
                return img

            frames = [ map_frame(path) for path in frames ]
            frames = torch.stack(frames)

        caption = [self.vocab(token) for token in self.vocab.tokenize(caption)]
        if caption[-1] != self.vocab(self.vocab.END):
            caption.append(self.vocab(self.vocab.END))

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
        if idx not in self.idx2word:
            return self.UNKNOWN
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
        result = u''
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

def default_transform(frame_size=None, num_channels=3):
    if len(frame_size) == 1:
        frame_size.append(frame_size[0])

    if num_channels == 3:
        transform = transforms.Compose([transforms.CenterCrop(frame_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        transform = transforms.Compose([transforms.CenterCrop(frame_size),
                                        transforms.Grayscale(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])])
    return transform

def cifar10_dataset(data=None, vocab=None, anno=None, transform=None, download=True):
    import torchvision.datasets as datasets
    return datasets.CIFAR10(data, transform=transform, download=download)

def my_dataset(data=None, vocab=None, anno=None, transform=None, random_frames=False, num_frames=16, use_lmdb=False):
    return Dataset(video_dir=data, vocab=vocab, captions=anno, transform=transform, random_frames=random_frames, num_frames=num_frames, use_lmdb=use_lmdb)

def get_loader(dset=None, batch_size=64, val=False, num_workers=4, has_captions=False):
    if has_captions:
        return torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=not val, num_workers=num_workers, collate_fn=collate_fn)
    else:
        return torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=not val, num_workers=num_workers)

def main(args):
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
