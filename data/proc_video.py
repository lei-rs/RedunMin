import gc
import json
import os
import pathlib
from typing import Dict

import av
import pandas as pd
from bounded_pool_executor import BoundedProcessPoolExecutor
from pydantic import BaseModel
from tqdm import tqdm

DATASET = 'ssv2'


class SampleMetadata(BaseModel):
    key: str
    cls: str
    subset: str
    annotations: dict


class VideoMetadata:
    def __init__(self):
        self.md: Dict[SampleMetadata] = {}
        self.int_labels = {}

    def get_info(self, key) -> SampleMetadata:
        try:
            md = self.md[key]
            md['key'] = key
            md['cls'] = self.int_labels[md['annotations']['label']]
            return SampleMetadata(**md)
        except KeyError:
            raise Exception(f'Missing metadata for {key}')


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
        self.md_path = pathlib.Path.cwd()
        train = json.load(open(str(self.md_path / f'{dataset}/train.json'), 'r'))
        val = json.load(open(str(self.md_path / f'{dataset}/val.json'), 'r'))
        test = json.load(open(str(self.md_path / f'{dataset}/test.json'), 'r'))
        test_ans = pd.read_csv(str(self.md_path / f'{dataset}/test-answers.csv'), header=None, index_col=0, delimiter=';')
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


def _new_output(out_path, width, height):
    output_container = av.open(out_path, 'wb')
    output_stream = output_container.add_stream(
        'h264',
        rate=30,
        options={
            'preset': 'veryslow',
            'tune': 'fastdecode'
        }
    )
    output_stream.pix_fmt = 'yuv420p'
    output_stream.width = width
    output_stream.height = height
    return output_container, output_stream


def _calc_new_dims(width, height, short_edge=224):
    if width < height:
        new_width = short_edge
        new_height = int(height * short_edge / width)
        if new_height % 2 != 0:
            new_height += 1
    else:
        new_height = short_edge
        new_width = int(width * short_edge / height)
        if new_width % 2 != 0:
            new_width += 1
    return new_width, new_height


def _resize_video(input_path: str, out_path: str):
    num_frames = 0
    input_container = av.open(input_path)
    output_container, output_stream = None, None

    for i, frame in enumerate(input_container.decode(video=0)):
        w, h = _calc_new_dims(frame.width, frame.height)
        if output_container is None:
            output_container, output_stream = _new_output(out_path, w, h)

        frame = frame.reformat(
            width=w,
            height=h,
            format='rgb24',
            interpolation=av.video.reformatter.Interpolation.LANCZOS
        )

        for packet in output_stream.encode(frame):
            output_container.mux(packet)

        num_frames += 1

    for packet in output_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    return num_frames


def _process_task(vid_path: pathlib.Path, out_path: str):
    vid_name = vid_path.stem
    try:
        md = vmd.get_info(vid_name)
        key = md.key
        sample_path = f'{out_path}/{md.subset}/{key}'
        os.makedirs(sample_path, exist_ok=True)
        assert _resize_video(
            str(vid_path),
            f'{sample_path}/vid.mp4'
        ) > 0, f'Empty video {vid_name}'
        gc.collect()
        with open(f'{sample_path}/md.json', 'w') as f:
            json.dump(md.model_dump_json(), f)
        return
    except Exception as e:
        print(f'Error processing {vid_name}: {e}')


class VideoToWebdataset:
    def __init__(self, input_path, output_path,  verbose=False):
        self.output_path = pathlib.Path(output_path)
        self.verbose = verbose

        self.md_out_path = self.output_path / 'metadata'
        self.md_out_path.mkdir(parents=True, exist_ok=True)
        self.file_paths = list(pathlib.Path(input_path).rglob('*'))
        self.error_log = ['id error']

        for path in ['train', 'val', 'tests']:
            (self.output_path / path).mkdir(parents=True, exist_ok=True)

    def _write_error_log(self):
        with open(str(self.md_out_path / f'errors.log'), 'w') as f:
            f.write('\n'.join(self.error_log))

    def start(self, n_jobs):
        executor = BoundedProcessPoolExecutor(max_workers=n_jobs)
        for path in tqdm(self.file_paths):
            executor.submit(_process_task, path, self.output_path)


if __name__ == '__main__':
    VideoToWebdataset(
        'rawdata',
        'data_out',
        verbose=True
    ).start(32)
