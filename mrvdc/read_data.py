from pathlib import Path
import pandas

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
    for row in csv.itertuples():
        name = row[VIDEO_ID]

        key = (name, row[START], row[END])

        if key not in result:
            result[key] = []

        desc = row[DESC]

        if len(desc) <= 60:
            result[key].append(desc)

    video_ids = [ k for k in result ]
    got_videos = [ p.stem[0:p.stem.find('_')] for p in Path(yt_clip_dir).iterdir() ]
    missing = [ k for k in set(video_ids) - set(got_videos) ]

    print('missing %d videos' % len(missing))

    # TODO
    #for m in missing:
        #print(m)
        #download_yt(m, yt_clip_dir)

    return result

if __name__ == '__main__':
    data = read_data('data/MSR.csv')

    sent_map = {}
    for k in data:
        video_name = k[0]
        video_name += "_"
        video_name += str(k[1])
        video_name += "_"
        video_name += str(k[2])
        sent_map[video_name] = data[k]

    with open('data/msr.pickle', 'wb') as out_f:
        import pickle
        pickle.dump(sent_map, out_f)
