import pickle
import json
import urllib
import urllib.parse

from pathlib import Path

def from_url_to_path(url):
    parsed = urllib.parse.urlparse(url)
    return parsed.netloc + parsed.path.replace('/', '_')

def from_url_to_key(url):
    path = from_url_to_path(url)
    return from_path_to_key(path)

def from_path_to_key(video_path):
    invalid_exts = [ '.gif', '.gifv', '.mp4', '.webm' ]

    video_path = Path(video_path).name
    while Path(video_path).suffix in invalid_exts:
        video_path = Path(video_path).stem

    key = str(video_path)
    return key

def main(args):
    sent_map = None
    with open(args.urls) as json_file:
        json_obj = json.load(json_file)
        sent_map = { from_url_to_key(obj['url']): [obj['title']] for obj in json_obj }

    with open(args.out, 'wb') as out_f:
        pickle.dump(sent_map, out_f)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls', type=str, default=None, help='location of urls')
    parser.add_argument('--out', type=str, default=None, help='output path', required=True)

    args = parser.parse_args()
    main(args)
