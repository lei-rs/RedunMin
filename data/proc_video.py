import gc
import json
import pathlib
import pickle
from io import BytesIO
from multiprocessing import Pool, Queue
from typing import Dict, List, Union, Any, Optional

import av
import pandas as pd
from av.video.frame import VideoFrame
from cvproc import h264_to_ndarrays
from numpy import ndarray
from pydantic import BaseModel, ConfigDict, field_serializer, model_validator
from rand_archive import Writer
from tqdm import tqdm


av.logging.set_level(av.logging.ERROR)


class Video(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    frames: List[ndarray]
    frame_count: int
    height: int
    width: int

    @classmethod
    def from_frames(cls, frames: List[ndarray]) -> 'Video':
        assert len(frames) > 0, 'Empty video'
        return cls.model_construct(
            frames=frames,
            frame_count=len(frames),
            height=frames[0].shape[0],
            width=frames[0].shape[1],
        )

    @classmethod
    def from_path(cls, source: Union[str, BytesIO]) -> 'Video':
        container = av.open(source)
        frames = []
        for frame in container.decode(video=0):
            frame = frame.reformat(
                width=224,
                height=224,
                format='yuv420p',
                interpolation=av.video.reformatter.Interpolation.LANCZOS
            )
            frames.append(frame.to_ndarray(format='rgb24'))
        return cls.from_frames(frames)

    @field_serializer('frames')
    def ser_model(self, frames: List[ndarray]) -> bytes:
        buf = BytesIO()
        container = av.open(buf, mode='w', format='h264')
        stream = container.add_stream(
            'h264',
            options={
                'preset': 'ultrafast',
                'tune': 'fastdecode',
                'crf': '28',
            }
        )
        stream.pix_fmt = 'yuv420p'
        stream.width = frames[0].shape[1]
        stream.height = frames[0].shape[0]
        for frame in frames:
            frame = VideoFrame.from_ndarray(frame, format='rgb24')
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
        container.close()
        return bytes(buf.getvalue())

    @model_validator(mode='before')
    @classmethod
    def validate_model(cls, data) -> Any:
        assert isinstance(data, dict), 'Only dict is supported'
        assert isinstance(data['frames'], bytes), 'Video is already decoded'
        try:
            frames = h264_to_ndarrays(data['frames'])
        except Exception as e:
            raise ValueError('Failed to decode video') from e
        assert len(frames) == data['frame_count'], 'Frame count mismatch'
        data['frames'] = frames
        return data


class VideoSample(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )
    key: str
    cls: str
    video: Video
    extra: Optional[dict] = None


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
        test_ans = pd.read_csv(str(self.md_path / f'{dataset}/test-answers.csv'), header=None, index_col=0,
                               delimiter=';')
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


def _process_task(vid_path: pathlib.Path):
    vid_name = vid_path.stem
    try:
        md = vmd.get_info(vid_name).model_dump()
        vid = Video.from_path(str(vid_path))
        sample = VideoSample(
            key=md.pop('key'),
            cls=md.pop('cls'),
            video=vid,
            extra=md
        )
        gc.collect()
        return q.put(sample.model_dump(mode='python'), block=True)
    except Exception as e:
        return q.put([e, vid_path], block=True)


class VideoToWebdataset:
    def __init__(self, input_path, output_path, verbose=False):
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

            if isinstance(out, dict):
                self._write_to_subset(out['key'], pickle.dumps(out), out['extra']['subset'])

            else:
                e, vid_name = out
                self.error_log.append(f'{str(vid_name)} {str(e)}')
                if self.verbose:
                    print(f'Error processing {str(vid_name)}: {str(e)}')

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
