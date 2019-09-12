from pathlib import Path
import pandas

import cv2

# input: list of YT video IDs
# output: {video ID: catagory} mapping
def get_categories(video_list):
    from yt import create_yt_api, get_category, get_all_categories

    yt = create_yt_api()
    all_cats = get_all_categories(yt)
    print(all_cats)

    result = {}
    for vi in video_list:
        cat = get_category(yt, vi)
        if cat is None:
            continue

        as_str = all_cats[cat]
        print('{} = {} ({})'.format(vi, as_str, cat))
        result[vi] = (cat, as_str)
        break

    return result, all_cats

def download_yt(video):
    pass


MAX_FRAMES = 32
MAX_DESC = 60

# returns set of (video: [ sentences ]) pairs
def read_data(csv_path_or_buf, yt_clip_dir='data/YouTubeClips'):
    result = {}

    csv = pandas.read_csv(csv_path_or_buf, sep=',')

    # get english only, simple sentences
    csv = csv.loc[csv['Language'] == 'English']
    csv = csv.loc[csv['Description'].str.len() <= 60]

    csv = csv[['VideoID', 'Start', 'End', 'Description']]
    csv['VideoID'] = [ k.strip('"') for k in csv['VideoID'] ]
    csv['Description'] = [ k.strip('"') for k in csv['Description'] ]

    VIDEO_ID = 1
    START = 2
    END = 3
    DESC = 4
    result = {}

    import tqdm
    rows = list(csv.itertuples())
    for row in tqdm.tqdm(rows):
        name = row[VIDEO_ID]
        start = row[START]
        end = row[END]
        vid_len = end - start + 1

        key = name
        key += '_'
        key += str(start) 
        key += '_'
        key += str(end)

        desc = row[DESC]
        
        video_name = '%s/%s.avi' % ('data/YouTubeClips', key)
        video = cv2.VideoCapture(video_name)
        fps = video.get(cv2.CAP_PROP_FPS)
        video.release()
        num_frames = vid_len * fps
        if len(desc) > MAX_DESC or vid_len*fps < MAX_FRAMES:
            continue

        if key not in result:
            result[key] = []

        result[key].append(desc)

    video_ids = [ k for k in result ]
    got_videos = [ p.stem for p in Path(yt_clip_dir).iterdir() ]
    missing = [ k for k in set(video_ids) - set(got_videos) ]

    print('missing %d videos' % len(missing))

    # TODO: download vids
    for m in missing:
        result.pop(m)
        #download_yt(m, yt_clip_dir)

    print('have %d unique videos' % len(result))
    print('have %d samples' % len([ sent for key in result for sent in result[key] ]))

    return result

if __name__ == '__main__':
    sent_map = read_data('data/MSR.csv')

    with open('data/msr.pickle', 'wb') as out_f:
        import pickle
        pickle.dump(sent_map, out_f)
