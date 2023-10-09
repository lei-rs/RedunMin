import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Dict

import pypeln as pl
from rand_archive import Reader

import src.data.transforms as T
from .types import VideoSample
from .wrappers import Batcher


@dataclass(init=True, repr=True, frozen=True)
class DLConfig:
    data_loc: str
    batch_size: int = 1
    shuffle: bool = False
    n_frames: int = 32
    base_seed: int = 42


class DataLoader:
    def __init__(self, dataset: str, config: DLConfig):
        self.dataset = dataset
        self.config = config

        self.readers = {stage: self._build_reader(stage) for stage in ['train', 'val', 'test']}
        self.transforms: Dict[str, Optional[Callable]] = {stage: None for stage in ['train', 'val', 'test']}
        self.epoch = 0

    def _build_reader(self, stage) -> Iterable[Any]:
        reader = Reader().by_count(32)

        if self.config.data_loc.startswith('gs://'):
            reader = reader.open_gcs(f'{self.config.data_loc}/{self.dataset}/{stage}.raa').with_buffering(32)
        else:
            reader = reader.open_file(f'{self.config.data_loc}/{self.dataset}/{stage}.raa')

        return reader

    def _update_transforms(self):
        pass

    def _get_loader(self, stage) -> Iterable[Any]:
        loader = self.readers[stage]
        if stage == 'train' and self.config.shuffle:
            loader = loader.with_shuffling(self._epoch_seed())

        loader = (
            pl.sync.from_iterable(iter(loader))
            | pl.thread.map(lambda x: VideoSample(**pickle.loads(x[1])), workers=8, maxsize=32)
            | pl.sync.map(lambda x: x.sample_frames(self.config.n_frames).to_tensors())
        )

        if self.transforms[stage] is not None:
            loader = loader | pl.sync.map(lambda x: self.transforms[stage](x[-1]), workers=2, maxsize=32)

        if self.config.batch_size > 1:
            loader = (
                Batcher(loader, self.config.batch_size)
            )

        return loader

    def _epoch_seed(self) -> int:
        return self.config.base_seed + self.epoch

    def step(self):
        self.epoch += 1
        self._update_transforms()

    def train_loader(self) -> Iterable[Any]:
        return self._get_loader('train')

    def val_loader(self) -> Iterable[Any]:
        return self._get_loader('val')

    def test_loader(self) -> Iterable[Any]:
        return self._get_loader('tests')


class SSV2(DataLoader):
    def __init__(self, config: DLConfig):
        super().__init__('ssv2', config)

        self.transforms = {
            'train': T.TrivialAugment([
                T.Cutout(8, (8, 32)),
                T.FlipHorizontal(),
                T.Shear('x', (0, 1)),
                T.Shear('y', (0, 1)),
                T.Rotate((0, 360)),
            ]),
        }
