import gc
import io
import json
import pathlib
from multiprocessing import Pool, Queue
from typing import Dict

import av
import pandas as pd
import pickle
from pydantic import BaseModel
from tqdm import tqdm
from rand_archive import Writer


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


q = Queue(200)
vmd = SSV2Metadata('ssv2')


def _new_output(out_path, width, height):
    output_container = av.open(out_path, 'wb', 'h264')
    output_stream = output_container.add_stream(
        'h264',
        rate=30,
        options={
            'tune': 'fastdecode',
            'crf': '28',
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


def _resize_video(input_path: str):
    num_frames = 0
    input_container = av.open(input_path)
    output = io.BytesIO()
    output_container, output_stream = None, None

    for i, frame in enumerate(input_container.decode(video=0)):
        w, h = _calc_new_dims(frame.width, frame.height)
        if output_container is None:
            output_container, output_stream = _new_output(output, w, h)

        frame = frame.reformat(
            width=w,
            height=h,
            format='yuv420p',
            interpolation=av.video.reformatter.Interpolation.LANCZOS
        )

        for packet in output_stream.encode(frame):
            output_container.mux(packet)

        num_frames += 1

    for packet in output_stream.encode():
        output_container.mux(packet)

    input_container.close()
    output_container.close()
    return num_frames, output


def _process_task(vid_path: pathlib.Path):
    vid_name = vid_path.stem
    try:
        md = vmd.get_info(vid_name)
        out = _resize_video(str(vid_path))
        assert out[0] > 0, f'Empty video {vid_name}'
        gc.collect()
        return q.put([out[1].getvalue(), md], block=True)
    except Exception as e:
        return q.put([e, vid_name], block=True)


class VideoToWebdataset:
    def __init__(self, input_path, output_path,  verbose=False):
        self.output_path = pathlib.Path(output_path)
        self.verbose = verbose

        self.md_out_path = self.output_path / 'metadata'
        self.md_out_path.mkdir(parents=True, exist_ok=True)
        self.file_paths = list(pathlib.Path(input_path).rglob('*'))
        self.error_log = ['id error']

        self.writers = {
            'train': Writer(
                str(self.output_path / 'train.raa'),
                max_header_size=10_000_000,
            ),
            'val': Writer(
                str(self.output_path / 'val.raa'),
                max_header_size=1_000_000,
            ),
            'tests': Writer(
                str(self.output_path / 'tests.raa'),
                max_header_size=1_000_000,
            ),
        }

    def _write_error_log(self):
        with open(str(self.md_out_path / f'errors.log'), 'w') as f:
            f.write('\n'.join(self.error_log))

    def _write_to_subset(self, key, sample, subset):
        assert len(sample) > 0, f'Empty sample {key} {subset}'
        self.writers[subset].write(key, sample)

    def _close_writers(self):
        for writer in self.writers.values():
            writer.close()

    def start(self, n_jobs):
        p = Pool(n_jobs - 1)
        p.map_async(_process_task, self.file_paths, chunksize=64)
        pbar = tqdm(range(len(self.file_paths)), total=len(self.file_paths), mininterval=1)

        for _ in pbar:
            out = q.get(block=True, timeout=100)

            if out[1] is str:
                e, vid_name = out
                self.error_log.append(f'{vid_name} {str(e)}')
                if self.verbose:
                    print(f'Error processing {vid_name}: {str(e)}')

            else:
                raw, md = out
                sample = {
                    'md': dict(md),
                    'video': raw,
                }
                self._write_to_subset(md.key, pickle.dumps(sample), md.subset)

        p.close()
        p.join()
        self._close_writers()
        self._write_error_log()


if __name__ == '__main__':
    VideoToWebdataset(
        'rawdata',
        'data_out',
        verbose=True
    ).start(64)
