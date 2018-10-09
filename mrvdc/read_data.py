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

    print(csv.shape)
    
    csv = csv[['VideoID', 'Description']]

    data = [ (k.strip('"'), v) for k, v in csv.values ]
    print('Amount of data points =', len(data))
    data = set(data)
    print('Amount of unique =', len(data))


    result = {}
    for k, v in data:
        if k not in result:
            result[k] = []

        result[k].append(v)

    video_ids = [ k for k in result ]
    video_ids.sort()

    got_videos = [ p.stem[0:p.stem.find('_')] for p in Path(yt_clip_dir).iterdir() ]
    missing = set(video_ids) - set(got_videos)
    print('len', len(missing))
    print(missing)

    for m in missing:
        download_yt(m, yt_clip_dir)


    #print(video_ids[0])
    cats, all_cats = get_categories(video_ids)
    #filter_cats = [

    return result
    

if __name__ == '__main__':
    data = read_data('data/MSR.csv')
    #print(data)
