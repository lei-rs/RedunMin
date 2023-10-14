import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Dict, List, Tuple

import jax
import pypeln as pl
from rand_archive import Reader

import src.data.transforms as T
from .types import VideoSample
from .wrappers import Batcher


def default_transforms(key):
    return {
            'train': [T.TrivialAugment.default(key=key), T.Normalize.default()],
            'val': [T.Normalize.default()],
            'test': [T.Normalize.default()],
        }


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
        self.transforms: Dict[str, Optional[List[Callable]]] = default_transforms(jax.random.PRNGKey(config.base_seed))
        self.epoch = 0

    def _build_reader(self, stage) -> Iterable[Any]:
        reader = Reader().by_count(64)

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

        def to_tensors(x: Tuple[str, bytes]):
            return VideoSample(**pickle.loads(x[1])).sample_frames(self.config.n_frames).to_tensors()
        loader = (
            pl.sync.from_iterable(iter(loader))
            | pl.thread.map(to_tensors, workers=12, maxsize=32)
        )

        if self.transforms[stage] is not None:
            def do_transform(transforms, x):
                for transform in transforms:
                    x = transform(x)
                return x
            loader = loader | pl.sync.map(lambda x: do_transform(self.transforms[stage], x))

        if self.config.batch_size > 1:
            loader = pl.sync.from_iterable(Batcher(loader, 2, self.config.batch_size))

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
        return self._get_loader('test')

    def len(self, stage: str) -> int:
        raise NotImplementedError


class SSV2(DataLoader):
    def __init__(self, config: DLConfig):
        super().__init__('ssv2', config)
        self._len = {
            'train': 220_847,
            'val': 24_777,
            'test': 27_157
        }

    def len(self, stage: str) -> int:
        return self._len[stage]
