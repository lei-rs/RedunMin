import av
import io
import gc
import json
import pathlib
import random
import numpy as np
import pandas as pd
import webdataset as wds
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool, Queue


DATASET = 'ssv2'
VID_EXT = 'webm'
outputs = Queue(maxsize=100)


class VideoMetadata:
    def __init__(self):
        self.md = {}
        self.int_labels = {}

    def get_info(self, vid_name):
        try:
            md = self.md[vid_name]
            label = md['annotations']['label']
            int_label = self.int_labels[label]
        except KeyError:
            raise Exception(f'Missing metadata for {vid_name}')
        return md, int_label


class KineticsMetadata(VideoMetadata):
    def __init__(self, dataset):
        super().__init__()
        self.md_path = pathlib.Path.cwd() / 'metadata'
        self.md = json.load(open(str(self.md_path / f'{dataset}_ann_raw.json'), 'r'))
        labels = open(str(self.md_path / f'{dataset}_labels.txt'), 'r').read().split('\n')
        self.int_labels = {label: str(i).encode('ascii') for i, label in enumerate(labels)}


class SSV2Metadata(VideoMetadata):
    def __init__(self, dataset):
        super().__init__()
        self.md_path = pathlib.Path.cwd() / 'metadata'
        train = json.load(open(str(self.md_path / f'{dataset}/train.json'), 'r'))
        val = json.load(open(str(self.md_path / f'{dataset}/val.json'), 'r'))
        test = json.load(open(str(self.md_path / f'{dataset}/tests.json'), 'r'))
        test_ans = pd.read_csv(str(self.md_path / f'{dataset}/tests-answers.csv'), header=None, index_col=0, delimiter=';')
        self.int_labels = json.load(open(str(self.md_path / f'{dataset}/labels.json'), 'r'))

        for entry in train:
            self._add_entry(entry, 'train')

        for entry in val:
            self._add_entry(entry, 'val')

        for entry in test:
            entry['label'] = test_ans.loc[int(entry['id'])].values[0]
            self._add_entry(entry, 'tests')

    def _add_entry(self, entry, subset):
        self.md[entry['id']] = {
            'annotations': {
                'label': entry['label'] if subset == 'tests' else entry['template'].replace('[', '').replace(']', ''),
                'raw_label': entry['label'] if subset != 'tests' else None,
                'placeholder': entry['placeholders'] if subset != 'tests' else None,
            },
            'subset': subset
        }


vmd = SSV2Metadata(DATASET)


def _create_sample(vid_path: pathlib.Path, vid_name: str, int_label: str):
    sample = {'__key__': f'{vid_name}'}
    num_frames = 0

    with av.open(str(vid_path)) as container:
        stream = container.streams.video[0]
        for i, frame in enumerate(container.decode(stream)):
            frame = frame.to_image()
            if np.isnan(np.asarray(frame)).any():
                raise Exception('Frame is None')
            h, w = frame.size
            if h > 240:
                frame = frame.resize((int(w * 240 / h), 240))
            frame_bytes = io.BytesIO()
            frame.save(frame_bytes, format='JPEG2000', quality=85, optimize=True)
            frame.close()
            sample[f'frame_{i:06d}.jpeg'] = frame_bytes.getvalue()
            num_frames += 1

    if num_frames == 0:
        raise Exception('No frames in video')

    sample['target.cls'] = str(int_label).encode('ascii')
    sample['num_frames'] = str(num_frames).encode('ascii')

    gc.collect()

    return sample


def _process_task(vid_path: pathlib.Path):
    vid_name = vid_path.stem
    try:
        md, int_label = vmd.get_info(vid_name)
        sample = _create_sample(vid_path, vid_name, int_label)
        outputs.put((sample, vid_name, md), block=True)

    except Exception as e:
        outputs.put((e, vid_name, None), block=True)


class VideoToWebdataset:
    def __init__(self, dataset,  verbose=False):
        self.dataset = dataset
        self.verbose = verbose

        input_path = pathlib.Path.cwd() / dataset
        output_path = pathlib.Path.cwd() / f'{dataset}_out'
        self.md_out_path = output_path / 'metadata'
        self.md_out_path.mkdir(parents=True, exist_ok=True)
        self.file_paths = list(pathlib.Path(input_path).rglob(f'*.{VID_EXT}'))
        random.shuffle(self.file_paths)

        self.error_log = ['id path error']
        self.counts = {label: 0 for label in vmd.int_labels.keys()}

        (output_path / 'train').mkdir(parents=True, exist_ok=True)
        (output_path / 'val').mkdir(parents=True, exist_ok=True)
        (output_path / 'tests').mkdir(parents=True, exist_ok=True)

        self.writers = {
            'train': wds.ShardWriter(str(output_path / f'train/shard_%06d.tar'), maxcount=5000, maxsize=2.5e9, encoder=False),
            'val': wds.ShardWriter(str(output_path / f'val/shard_%06d.tar'), maxcount=5000, maxsize=2.5e9, encoder=False),
            'tests': wds.ShardWriter(str(output_path / f'tests/shard_%06d.tar'), maxcount=5000, maxsize=2.5e9, encoder=False),
        }

    def _dump_counts(self):
        counter = {
            'missing': len(self.error_log) - 1,
            'labels': self.counts
        }

        with open(str(self.md_out_path / f'counts.json'), 'w') as f:
            json.dump(counter, f, indent=4)

    def _write_error_log(self):
        with open(str(self.md_out_path / f'errors.log'), 'w') as f:
            f.write('\n'.join(self.error_log))

    def _write_to_subset(self, sample, subset):
        self.writers[subset].write(sample)

    def _close_writers(self):
        for writer in self.writers.values():
            writer.close()

    def __call__(self, n_jobs):
        p = Pool(n_jobs - 1)
        p.map_async(_process_task, self.file_paths, chunksize=64)
        pbar = tqdm(range(len(self.file_paths)), total=len(self.file_paths), mininterval=1)

        for _ in pbar:
            sample, vid_name, md = outputs.get(block=True, timeout=100)

            if md is None:
                if self.verbose:
                    print(vid_name, sample)
                self.error_log.append(f'{vid_name} \'{str(sample)}\'')
                pbar.set_description(f'Missing {len(self.error_log) - 1} videos')

            else:
                label = md['annotations']['label']
                split = md['subset']
                self._write_to_subset(sample, split)
                self.counts[label] += 1

        p.close()
        p.join()
        self._close_writers()
        self._dump_counts()
        self._write_error_log()


if __name__ == '__main__':
    VideoToWebdataset(DATASET, verbose=True)(31)
