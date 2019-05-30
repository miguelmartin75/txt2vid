import sys
from pathlib import Path

import tqdm
import cv2
import numpy as np

from txt2vid.data import read_video_file, to_pil

def get_videos(directory):
    for video in directory.iterdir():
        if video.suffix != '.avi':
            continue
        yield video

def resize(frame, frame_size):
    return cv2.resize(frame, (frame_size, frame_size))

def get_all_frames(video_dir):
    videos = list(get_videos(video_dir))

    for video in tqdm.tqdm(videos):
        vid = video.stem
        sys.stdout.flush()

        frames = []
        for idx, frame in enumerate(read_video_file(video, convert_to_pil=False)):
            frame = resize(frame, args.frame_size)
            frames.append(frame)

        yield vid, frames

        for frame in frames:
            del frame

def main(args):
    video_dir = Path(args.dir)

    if args.lmdb is not None:
        # https://github.com/pytorch/pytorch/blob/master/caffe2/python/examples/lmdb_create_example.py
        from caffe2.proto import caffe2_pb2
        from caffe2.python import workspace, model_helper
        import lmdb

        LMDB_MAP_SIZE = 1 << 40 # changeme

        env = lmdb.open(args.lmdb, map_size=LMDB_MAP_SIZE)
        checksum = 0
        with env.begin(write=True) as txn:
            for vid, frames in get_all_frames(video_dir):
                tensor_protos = caffe2_pb2.TensorProtos()

                # TODO: configure
                from txt2vid.data import pick_frames

                mean = 0.5
                std = 0.5
                def transform_frame(frame):
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = np.array(frame.transpose(2, 0, 1), dtype=np.float32)
                    for ch in range(frame.shape[0]):
                        frame[ch] = (frame[ch] - mean) / std
                    return frame

                frames = pick_frames(frames, random=False, num_frames=16)
                frames = [ transform_frame(frame) for frame in frames ]

                frames = np.stack(frames)

                fixed_vid_tensor = tensor_protos.protos.add()
                fixed_vid_tensor.dims.extend(frames.shape)
                fixed_vid_tensor.data_type = 1

                flat = frames.reshape(np.prod(frames.shape))
                fixed_vid_tensor.float_data.extend(flat)
                txn.put('{}'.format(vid).encode('ascii'), tensor_protos.SerializeToString())

                checksum += np.sum(frames) 

        print("Checksum/write: {}".format(int(checksum)))

    else:
        for vid, frames in get_all_frames(video_dir):
            for idx, frame in enumerate(frames):
                path_to_save = '%s/%s/%d.jpg' % (video_dir, vid, idx)
                parent_path = Path(path_to_save).parent
                if not parent_path.exists():
                    parent_path.mkdir()

                cv2.imwrite(path_to_save, frame)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos', required=True)
    parser.add_argument('--frame_size', type=int, default=256, help='size of frame')
    parser.add_argument('--lmdb', type=str, default=None, help='save to LMDB?')

    args = parser.parse_args()
    main(args)
