import json
import pandas as pd
from tqdm import tqdm

dataset = 'data/k600'
metadata = {}

train_json = json.load(open(f'{dataset}/train.json'))
val_json = json.load(open(f'{dataset}/val.json'))
test_json = json.load(open(f'{dataset}/test.json'))
#train_csv = pd.read_csv(f'{dataset}/train.csv')
#val_csv = pd.read_csv(f'{dataset}/val.csv')
#test_csv = pd.read_csv(f'{dataset}/tests.csv')
train_txt = open(f'{dataset}/train.txt', 'r').read().split('\n\n')
val_txt = open(f'{dataset}/val.txt', 'r').read().split('\n\n')


def proc_csv(csv: pd.DataFrame, md: dict):
    for i, row in tqdm(csv.iterrows(), total=len(csv)):
        md[row['youtube_id']] = {
            'annotations':
            {
                'label': row['label'],
                'segment': [row['time_start'], row['time_end']]
            },
            'duration': row['time_end'] - row['time_start']
        }

    return md


def proc_json(json: dict, md: dict):
    for key, val in tqdm(json.items(), total=len(json)):
        del val['subset']
        del val['url']
        md[key] = val

    return md


def proc_txt(txt: list, md: dict):
    labels = txt[0].split('\n')[1:]

    for i, block in tqdm(enumerate(txt[1:]), total=len(txt)):
        videos = block.split('\n')[1:]

        for video in videos:
            try:
                vid_name = video[:11]
                time_start = float(video[12:18])
                time_end = float(video[19:25])
                label = labels[i]

                md[vid_name] = {
                    'annotations':
                    {
                        'label': label,
                        'segment': [time_start, time_end]
                    },
                    'duration': time_end - time_start
                }
            except:
                pass

    return md


metadata = proc_json(train_json, metadata)
metadata = proc_json(val_json, metadata)
metadata = proc_json(test_json, metadata)
#metadata = proc_csv(train_csv, metadata)
#metadata = proc_csv(val_csv, metadata)
#metadata = proc_csv(test_csv, metadata)
metadata = proc_txt(train_txt, metadata)
metadata = proc_txt(val_txt, metadata)


with open(f'{dataset}/metadata.json', 'w') as f:
    f.write(json.dumps(metadata, indent=4, sort_keys=False))
