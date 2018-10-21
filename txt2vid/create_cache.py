from pathlib import Path

from data import read_video_file

import tqdm

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

def main(args):
    video_dir = Path(args.dir)

    videos = list(get_videos(video_dir))
    for video in tqdm.tqdm(videos):
        vid = video.stem

        for frame in read_video_file(video, vid, cache=video_dir):
            del frame

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos')

    args = parser.parse_args()

    main(args)
