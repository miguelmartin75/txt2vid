import sys
from pathlib import Path

import tqdm
import cv2
import numpy as np

from txt2vid.data import read_video_file, to_pil
from txt2vid.data.reddit_videos_json_to_pickle import from_path_to_key

def get_videos(directory):
    valid_suffixes = [ '.avi', '.mp4', '.gif', '.webm' ]
    for video in directory.iterdir():
        if not video.suffix in valid_suffixes:
            continue
        yield video

def resize(frame, frame_size):
    return cv2.resize(frame, (frame_size, frame_size))

def get_all_frames(videos):
    for video in tqdm.tqdm(videos):
        vid = from_path_to_key(video)

        frames = []
        for idx, frame in enumerate(read_video_file(video, convert_to_pil=False)):
            frame = resize(frame, args.frame_size)
            frames.append(frame)

        #if len(frames) == 0:
        #    continue

        yield vid, frames

        for frame in frames:
            del frame

def get_frames(raw_datum, frame_size=128):
    from caffe2.proto import caffe2_pb2
    tensor_protos = caffe2_pb2.TensorProtos()
    tensor_protos.ParseFromString(raw_datum)

    print('type=', tensor_protos.protos[0].data_type)
    data = tensor_protos.protos[0].int32_data
    print(data)
    data = data.reshape(16, 3, frame_size, frame_size)
    data -= 255*.5
    data /= 255*.5
    #data = data*.5
    return data

def f(video_dir, video):
    vid = from_path_to_key(video)
    frames = list(read_video_file(video, convert_to_pil=False))

    if len(frames) < 16:
        return vid

    from txt2vid.data import pick_frames
    frames = pick_frames(frames, random=False, num_frames=16)


    for idx, frame in enumerate(frames):
        frame = resize(frame, args.frame_size)

        path_to_save = '%s/%s/%d.jpg' % (video_dir, vid, idx)
        parent_path = Path(path_to_save).parent
        if not parent_path.exists():
            parent_path.mkdir()

        cv2.imwrite(path_to_save, frame)

    return vid

def main(args):
    video_dir = Path(args.dir)

    if args.lmdb is not None:
        if args.sent is not None:
            from txt2vid.util.pick import load
            sent = load(args.sent)

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

                # https://github.com/pytorch/pytorch/blob/master/caffe2/proto/caffe2.proto
                frames = pick_frames(frames, random=False, num_frames=16)
                frames = np.stack(frames)
                assert(frames.dtype == np.uint8)

                fixed_vid_tensor = tensor_protos.protos.add()
                fixed_vid_tensor.dims.extend(frames.shape)
                fixed_vid_tensor.data_type = 8

                flat = frames.reshape(np.prod(frames.shape))
                fixed_vid_tensor.int32_data.extend(flat)

               # if args.sent is not None:
               #     if args.is_labels:
               #         # TODO
               #         assert(False)
               #     else:
               #         #num_sentences = tensor_protos.protos.add()
               #         #num_sentences.data_type = 2 
               #         #num_sentences.int32_data.append(len(sent[vid]))

               #         sentence_tensor = tensor_protos.protos.add()
               #         sentence_tensor.data_type = 4
               #         for s in sent[vid]:
               #             sentence_tensor.string_data.append(s.encode())

                txn.put('{}'.format(vid).encode(), tensor_protos.SerializeToString())

        print("DONE")

    else:
        from functools import partial
        from multiprocessing import Pool
        with Pool(args.workers) as pool:
            videos = list(get_videos(video_dir))
            videos.sort()

            f_temp = partial(f, video_dir)
            processed = pool.imap_unordered(f_temp, videos)

            for vid in tqdm.tqdm(processed, total=len(videos)):
                continue

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default=None, help='location of videos', required=True)
    parser.add_argument('--sent', type=str, default=None, help='sentence or label data (if applicable)')
    parser.add_argument('--is_labels', action='store_true', default=False, help='is `sent` labels? By default it is assumed that `sent` is sentences.')
    parser.add_argument('--frame_size', type=int, default=256, help='size of frame')
    parser.add_argument('--lmdb', type=str, default=None, help='save to LMDB?')
    parser.add_argument('--workers', type=int, default=3, help='num workers')

    args = parser.parse_args()
    main(args)
