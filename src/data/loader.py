import pickle
from dataclasses import dataclass
from typing import Any, Iterable, Callable, Optional, Dict

import jax
import jax.numpy as jnp
import pypeln as pl
from jax.experimental import mesh_utils
from jax.sharding import PositionalSharding
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

        self.sharding = PositionalSharding(mesh_utils.create_device_mesh(
            (8,),
            jax.devices('tpu'))
        )

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

        to_tensors = lambda x: VideoSample(**pickle.loads(x[1])).sample_frames(self.config.n_frames).to_tensors()
        loader = (
            pl.sync.from_iterable(iter(loader))
            | pl.thread.map(to_tensors, workers=12, maxsize=32)
        )

        if self.config.batch_size > 1:
            loader = pl.sync.from_iterable(Batcher(loader, 2, self.config.batch_size))

        def put_device(x):
            x = (jnp.asarray(x[0]), jnp.asarray(x[1]))
            return (
                jax.device_put(x[0], self.sharding),
                jax.device_put(x[1], self.sharding.reshape((8, 1, 1, 1, 1))),
            )
        loader = loader | pl.thread.map(put_device, workers=4, maxsize=32)

        if self.transforms[stage] is not None:
            loader = loader | pl.sync.map(lambda x: self.transforms[stage](x))

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
        key = jax.random.key(config.base_seed)
        dummy = jnp.ones((config.batch_size, config.n_frames, 3, 224, 224), dtype=jnp.uint8)
        self.transforms = {
            'train': T.TrivialAugment([
                T.FlipHorizontal(),
                T.Shear('x', (0, 0.99)),
                T.Shear('y', (0, 0.99)),
                T.Rotate((0, 135.0)),
                T.Invert(),
            ], key=key),
        }
        self.transforms['train'].warmup(dummy)
