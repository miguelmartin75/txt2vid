import PIL
from PIL import Image
import random

import numpy as np

import torch
import torch.utils
import torch.utils.data
from torchvision import datasets

import cv2

# prop = [0, 1]
def linear_step(prop, a, b):
    return (a + (b-a)*prop)

def generate_frames(object_img=None, frame_size=None, repeat=False, frames=None,
                  animation_len=None, step_fn=linear_step, fromPt=None, toPt=None, idx=None):
    assert object_img is not None

    repeat_count = 1
    if repeat:
        repeat_count = int((frames+1)/animation_len)

    #aStart = np.random.randint(0, frames - repeat_count*animation_len + 1)
    aStart = 0
    aEnd = aStart + animation_len

    for i in range(frames):
        prop = (i - aStart + 1) / (aEnd - aStart + 1)
        prop = float(np.clip(prop, 0, 1))
        curr_pos = step_fn(prop, fromPt, toPt)
        curr_pos = np.array(curr_pos, dtype=np.int32)

        if repeat and i == aEnd:
            aStart += animation_len
            aEnd += animation_len

            # swap positions
            #aStart, aEnd = aEnd, aStart
            fromPt, toPt = toPt, fromPt

        pos = (curr_pos[0], curr_pos[1])
        frame = Image.new('RGB', (frame_size[0], frame_size[1]))
        frame.paste(object_img, pos)
        yield frame


def save_video(frames, path, fps, frame_size):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(path, fourcc, fps, (frame_size[0], frame_size[1]))

    for frame in frames:
        frame = np.array(frame)
        out.write(frame)
    out.release()

def generate_examples(video_dir, sentence_out, num_examples=10, fps=None, frame_size=None, num_frames=None, object_classes=None, objects=None, object_names=None):
    try:
        import os
        os.makedirs(video_dir)
    except:
        pass

    WIDTH = frame_size[0]
    HEIGHT = frame_size[1]

    corners = [ (0, 0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT) ]
    left_corners = [ corners[0], corners[2] ]
    right_corners = [ corners[1], corners[3] ]

    top_corners = [ corners[0], corners[1] ]
    bottom_corners = [ corners[2], corners[3] ]

    corner_names = [ 'top left', 'top right', 'bottom left', 'bottom right' ]

    sent_map = {}
    import tqdm
    #for i in range(5):
    for i in tqdm.tqdm(range(num_examples)):
        class_idx = np.random.randint(0, len(object_classes))
        idx = np.random.randint(0, len(objects[class_idx]))

        obj = objects[class_idx][idx]
        name = object_names[class_idx]

        bounce = True#random.randint(0, 1) # should we repeat the animation, in the reverse?
        animation_length = random.randint(int(0.1*num_frames), num_frames)
        #if bounce:
            #animation_length = #int((num_frames+1)/animation_length)
            #animation_length = random.randint(int(0.1*num_frames), num_frames/2)


        horizontal = random.randint(0, 1) # horiz or vert
        l2r_u2d = random.randint(0, 1) # if false, go reverse
        use_corners = False#random.randint(0, 1) # should we use corners or not

        a = None # from position
        b = None # to position

        sentence = '{} '.format(name)

        #if bounce:
        #    sentence += 'is moving '
        #else:
        #    sentence += 'moves '

        sentence += 'is '

        if use_corners:
            i1 = np.random.randint(0, len(corners))
            i2 = i1
            while i1 == i2:
                i2 = np.random.randint(0, len(corners))

            a = corners[i1]
            b = corners[i2]

            rn = random.randint(0, 1)
            if ((a in left_corners and b in left_corners) or (a in right_corners and b in right_corners) and rn):
                sentence += 'from '
                if a == left_corners[0] or a == right_corners[0]:
                    sentence += 'top to bottom'
                else:
                    sentence += 'bottom to top'
            elif ((a in top_corners and b in top_corners) or (a in bottom_corners and b in bottom_corners) and rn):
                sentence += 'from '
                if a == top_corners[0] or a == bottom_corners[0]:
                    sentence += 'left to right'
                else:
                    sentence += 'right to left'
            else:
                sentence += 'from {} to {}'.format(corner_names[i1], corner_names[i2])
        else:
            if horizontal:
                y = np.random.randint(0, HEIGHT)
                x1 = np.random.randint(0, int(0.1*WIDTH))
                x2 = np.random.randint(int(0.9*WIDTH), WIDTH)

                a = np.array([x1, y])
                b = np.array([x2, y])

                if l2r_u2d:
                    sentence += 'left and right'
                else:
                    sentence += 'right and left'
            else:
                x = np.random.randint(0, WIDTH)
                y1 = np.random.randint(0, int(0.1*HEIGHT))
                y2 = np.random.randint(int(0.9*HEIGHT), HEIGHT)

                a = np.array([x, y1])
                b = np.array([x, y2])

                if l2r_u2d:
                    sentence += 'top and bottom'
                else:
                    sentence += 'bottom and top'

            if not l2r_u2d:
                a, b = b, a

        a = np.array(a)
        b = np.array(b)

        a[0] = np.clip(a[0], 0, WIDTH - obj.size[0])
        a[1] = np.clip(a[1], 0, HEIGHT - obj.size[1])
        b[0] = np.clip(b[0], 0, WIDTH - obj.size[0])
        b[1] = np.clip(b[1], 0, HEIGHT - obj.size[1])

        # encode speed in sentence
        #if random.randint(0, 10) <= 8:
        #    anim_prop = animation_length / (num_frames)
        #    if anim_prop <= 0.1:
        #        sentence += ' very fast'
        #    elif anim_prop <= 0.4:
        #        sentence += ' fast'

        #if bounce:
        #    sentence += ' back and forth'
        sentence += '.'

        sent_map[i] = [ sentence ]

        frames = generate_frames(obj, FRAME_SIZE, frames=num_frames, animation_len=animation_length, repeat=bounce, fromPt=a, toPt=b, idx=i)
        save_video(frames, '{}/{}.avi'.format(video_dir, i), fps, frame_size)

    with open(sentence_out, 'wb') as out_f:
        import pickle
        pickle.dump(sent_map, out_f)


def load_mnist(train=False):
    data = datasets.MNIST('mnist_data', train=train, download=True)

    objects = {}
    object_classes = set()
    object_names = {}

    for t in data:
        img = t[0]
        clazz = int(t[1])

        img = img.resize((28,28)) # TODO: customize

        if clazz not in objects:
            objects[clazz] = []

        objects[clazz].append(img)
        object_classes.add(clazz)

    for c in object_classes:
        object_names[c] = 'digit {}'.format(c)

    return objects, object_classes, object_names 

# TODO: scale, rotation, etc.
if __name__ == '__main__':
    random.seed(300)
    np.random.seed(300)

    # PARAMETERS
    WIDTH = 64
    HEIGHT = 64
    FPS = 30
    VIDEO_LEN = 64 # number of frames

    FRAME_SIZE = np.array([WIDTH, HEIGHT])

    for train, dset, num_examples in zip([True, False], ['train', 'test'], [40000, 10000]):
        objects, object_classes, object_names = load_mnist(train)

        video_out = '/run/media/doubleu/Linux/synthetic/{}/videos'.format(dset)
        sent_out = '/run/media/doubleu/Linux/synthetic/{}/sent.pickle'.format(dset)
        print("Generating: %s to %s" % (dset, video_out))
        generate_examples(video_out, 
                          sent_out, 
                          num_examples=num_examples,
                          frame_size=FRAME_SIZE,
                          num_frames=VIDEO_LEN,
                          fps=FPS,
                          object_classes=object_classes,
                          objects=objects,
                          object_names=object_names)
