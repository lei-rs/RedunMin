import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Dict, List, Tuple

import jax
import jax.numpy as jnp
import haliax as hax
import pypeln as pl
import torch
from lightning import LightningDataModule
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


@hax.named_jit
def _collate(x):
    cls, x = x
    cls = hax.named(jnp.asarray(cls), 'batch')
    x = hax.named(jnp.asarray(x), ('batch', 'temporal', 'channels', 'height', 'width'))
    return cls, x


@dataclass(init=True, repr=True, frozen=True)
class DLConfig:
    data_loc: str
    put_fn: Callable
    batch_size: int = 1
    shuffle: bool = False
    n_frames: int = 32
    shard: Optional[Tuple[int, int]] = None


class DataLoader(LightningDataModule):
    def __init__(self, dataset: str, config: DLConfig, *, key):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.key, key_t = jax.random.split(key)

        self._len: Dict[str, int] = {}
        self.readers: Dict[str, Reader] = {}
        self.transforms: Dict[str, List[Callable]] = {}

    def _build_reader(self, stage) -> Iterable[Any]:
        reader = Reader().by_count(64)

        if self.config.data_loc.startswith('gs://'):
            reader = reader.open_gcs(f'{self.config.data_loc}/{self.dataset}/{stage}.raa').with_buffering(32)
        else:
            reader = reader.open_file(f'{self.config.data_loc}/{self.dataset}/{stage}.raa')

        if self.config.shard is not None and stage != 'test':
            reader = reader.with_sharding(*self.config.shard)

        return reader

    def _get_loader(self, stage) -> Iterable[Any]:
        loader = self.readers[stage]
        if stage == 'train' and self.config.shuffle:
            loader = loader.with_shuffling(self._fork_seed())

        def to_tensors(x: Tuple[str, bytes]):
            return VideoSample(**pickle.loads(x[1])).sample_frames(self.config.n_frames).to_tensors()
        loader = (
            iter(loader)
            | pl.thread.map(to_tensors, workers=12, maxsize=24)
        )

        if stage in self.transforms:
            def do_transform(x):
                cls, x = x
                for transform in self.transforms[stage]:
                    with jax.default_device(jax.devices("cpu")[0]):
                        x = transform(x)
                return cls, x
            loader = loader | pl.sync.map(do_transform)

        if self.config.batch_size > 1:
            loader = (
                Batcher(loader, 2, self.config.batch_size)
                | pl.thread.map(_collate, workers=2, maxsize=4)
            )

        return loader

    def _fork_seed(self) -> int:
        self.key, key = jax.random.split(self.key)
        return int(key[0])

    def setup(self, stage: str) -> None:
        if len(self.transforms) == 0:
            self.transforms = default_transforms(self.key)
        if len(self.readers) == 0:
            self.readers = {stage: self._build_reader(stage) for stage in ['train', 'val', 'test']}

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return self.config.put_fn(batch)

    def train_dataloader(self) -> Iterable:
        dl = self._get_loader('train')
        if 'train' in self._len:
            dl.__len__ = lambda: self._len['train']
        return dl

    def val_dataloader(self) -> Iterable:
        dl = self._get_loader('val')
        if 'val' in self._len:
            dl.__len__ = lambda: self._len['val']
        return dl

    def state_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.key = state_dict['key']


class SSV2(DataLoader):
    def __init__(self, config: DLConfig, *, key):
        super().__init__('ssv2', config, key=key)
        self._len = {
            'train': 220_847,
            'val': 24_777,
            'test': 27_157
        }
