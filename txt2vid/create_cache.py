from pathlib import Path

from data import read_video_file

import tqdm
import sys

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

def main(args):
    print('a')
    sys.stdout.flush()
    video_dir = Path(args.dir)
    print('b')
    sys.stdout.flush()

    print('getting vids')
    sys.stdout.flush()
    videos = list(get_videos(video_dir))
    print('got vids looping now')
    sys.stdout.flush()
    for video in tqdm.tqdm(videos):
        vid = video.stem
        sys.stdout.flush()
        for frame in read_video_file(video, vid, cache=video_dir):
            del frame

if __name__ == '__main__':
    print('hi')
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos', required=True)

    args = parser.parse_args()

    main(args)
