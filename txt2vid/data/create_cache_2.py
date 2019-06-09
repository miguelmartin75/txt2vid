import sys
from pathlib import Path

import cv2
import numpy as np
import lmdb
import tqdm

from txt2vid.data import read_video_file

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

def resize(frame, frame_size):
    return cv2.resize(frame, (frame_size, frame_size))

def main(args):

    LMDB_MAP_SIZE = 1 << 40
    env = lmdb.open(args.out, map_size=LMDB_MAP_SIZE)

    video_dir = Path(args.dir)
    videos = list(get_videos(video_dir))

    def transform(frame):
        return resize(frame, args.frame_size)

    with env.begin(write=True) as txn:
        for video in tqdm.tqdm(videos):
            vid = video.stem
            sys.stdout.flush()

            frames = [ transform(frame) for frame in read_video_file(video, convert_to_pil=False) ]
            frames = np.stack(frames)
    

            print(frames.shape)
            sys.exit(0)





if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos', required=True)
    parser.add_argument('--frame_size', type=int, default=256, help='size of frame')
    parser.add_argument('--out', type=str, default=None, help='output LMDB', required=True)

    args = parser.parse_args()
    main(args)
