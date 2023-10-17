import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Dict, Tuple

import equinox as eqx
import jax
import pypeln as pl
from haliax import NamedArray
from jax import Array
from lightning import LightningDataModule
from rand_archive import Reader

from .types import VideoSample
from .wrappers import Batcher


@dataclass(init=True, repr=True, frozen=True)
class DLConfig:
    data_loc: str
    collate_put: Callable
    transforms: Dict[str, Callable]
    batch_size: int = 1
    shuffle: bool = False
    n_frames: int = 32
    shard: Optional[Tuple[int, int]] = None


class DataLoader(LightningDataModule):
    def __init__(self, dataset: str, config: DLConfig, *, key):
        super().__init__()
        self.dataset = dataset
        self.cfg = config
        self.key, key_t = jax.random.split(key)

        self._len: Dict[str, int] = {}
        self.readers: Dict[str, Reader] = {}

    def _build_reader(self, stage) -> Iterable[Any]:
        reader = Reader().by_count(64)

        if self.cfg.data_loc.startswith('gs://'):
            reader = reader.open_gcs(f'{self.cfg.data_loc}/{self.dataset}/{stage}.raa').with_buffering(32)
        else:
            reader = reader.open_file(f'{self.cfg.data_loc}/{self.dataset}/{stage}.raa')

        if self.cfg.shard is not None and stage != 'test':
            reader = reader.with_sharding(*self.cfg.shard)

        return reader

    def _get_loader(self, stage) -> Iterable[Any]:
        loader = self.readers[stage]
        if stage == 'train' and self.cfg.shuffle:
            loader = loader.with_shuffling(int(self._fork_seed()[0]))

        def _decode(x: Tuple[str, bytes]):
            return VideoSample(**pickle.loads(x[1])).sample_frames(self.cfg.n_frames)

        def _to_tensors(x: VideoSample):
            return x.to_tensors()

        loader = (
            iter(loader)
            | pl.thread.map(_decode, workers=8, maxsize=24)
            | pl.thread.map(_to_tensors, workers=2, maxsize=6)
        )

        if self.cfg.batch_size > 1:
            loader = (
                Batcher(loader, self.cfg.batch_size)
                | pl.thread.map(lambda x: self.cfg.collate_put(*x), workers=1, maxsize=12)
            )

        if stage in self.cfg.transforms:
            def do_transform(batch_vid: Array):
                key = self._fork_seed()
                batch_key = jax.random.split(key, self.cfg.batch_size)
                return eqx.filter_vmap(
                    lambda v, k: self.cfg.transforms[stage](v, key=k),
                    in_axes=(0, 0),
                    out_axes=0
                )(batch_vid, batch_key)
            loader = loader | pl.sync.map(lambda batch: eqx.tree_at(
                lambda x: x[1].array,
                batch,
                do_transform(batch[1].array)
            ))

        return loader

    def _fork_seed(self):
        self.key, key = jax.random.split(self.key)
        return key

    def setup(self, stage: str) -> None:
        if len(self.readers) == 0:
            self.readers = {stage: self._build_reader(stage) for stage in ['train', 'val', 'test']}

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
