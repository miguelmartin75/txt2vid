import sys
from pathlib import Path

import tqdm

from txt2vid.data import read_video_file

import cv2

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

def resize(frame, frame_size):
    return cv2.resize(frame, (frame_size, frame_size))

def main(args):
    video_dir = Path(args.dir)
    videos = list(get_videos(video_dir))

    for video in tqdm.tqdm(videos):
        vid = video.stem
        sys.stdout.flush()

        for idx, frame in enumerate(read_video_file(video, convert_to_pil=False)):
            path_to_save = '%s/%s/%d.jpg' % (video_dir, vid, idx)
            parent_path = Path(path_to_save).parent
            if not parent_path.exists():
                parent_path.mkdir()
            frame = resize(frame, args.frame_size)
            cv2.imwrite(path_to_save, frame)
            #frame.save(path_to_save)

            del frame

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos', required=True)
    parser.add_argument('--frame_size', type=int, default=256, help='size of frame')

    args = parser.parse_args()
    main(args)
