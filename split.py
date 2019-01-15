import random
from pathlib import Path
from shutil import copy

path = Path('../cooking_videos')

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

vids = list(get_videos(path))

TRAIN_SPLIT_RATIO=0.8

train_dir = path / 'train'
val_dir = path / 'val'

from util.dir import ensure_exists
ensure_exists(train_dir)
ensure_exists(val_dir)

for vid in vids:
    r = random.uniform(0, 1)
    if r <= TRAIN_SPLIT_RATIO:
        copy(vid, str(train_dir))
    else:
        copy(vid, str(val_dir))
