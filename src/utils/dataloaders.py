import random
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Dict

import numpy as np
import torchvision.transforms as T
from lightning import LightningDataModule
from torch import Tensor, from_numpy, float32
from torch.nn import Module, Sequential
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES
from torchdata.dataloader2 import DistributedReadingService, DataLoader2, ReadingServiceInterface
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper
from torchvision.transforms import InterpolationMode

from .callable import SampleFrames, DecodeFrames
from .dispatcher import dispatch_worker


norm_info = {
    'ssv2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            distributed: bool,
            prefetch_count: int,
            test: bool,
            path: str,
            seed: int = 0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.distributed = distributed
        self.prefetch_count = prefetch_count
        self.test = test
        self.seed = seed

        shards = {task: list((Path(path) / task).glob('*.tar')) for task in ['train', 'val', 'test']}
        self.shards = {task: list(map(str, shards[task])) for task in ['train', 'val', 'test']}

        self.current_stage: Optional[str] = None
        self.current_epoch = 0

        # Initialized once, updated per stage
        self.cpu_transforms: Dict[str, Optional[List[Callable]]] = {
            'train': None,
            'val': None,
            'test': None
        }
        self.gpu_transforms: Dict[str, Optional[Module]] = {
            'train': None,
            'val': None,
            'test': None
        }
        self.train_loader: Optional[DataLoader2] = None
        self.val_loader: Optional[DataLoader2] = None
        self.test_loader: Optional[DataLoader2] = None

    @staticmethod
    def _batch(batch: List[Tuple[int, Tensor]]) -> Tuple[Tensor, Tensor]:
        targets, frames = map(list, zip(*batch))
        targets = from_numpy(np.asarray(targets))
        frames = from_numpy(np.stack([frame.numpy() for frame in frames]))
        return targets, frames

    def _update_cpu_transform(self) -> List[Callable]:
        # If you want to update per stage
        pass

    def _update_gpu_transform(self) -> Module:
        # If you want to update per stage
        pass

    def _get_datapipe(self, task: str, transforms: List[Callable]) -> IterDataPipe:
        dp: IterDataPipe = IterableWrapper(self.shards[task]).wds().read()
        if task == 'train':
            dp = dp.shuffle(buffer_size=10)
        if self.distributed:
            dp = (
                dp
                .sharding_filter(SHARDING_PRIORITIES.DISTRIBUTED)
                .sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING)
            )
        dp = dp.seq(transforms)
        dp = dp.batch(batch_size=self.batch_size, drop_last=True, wrapper_class=self._batch)
        return dp

    def _define_reading_service(self) -> ReadingServiceInterface:
        rs: Optional[ReadingServiceInterface] = None
        if self.distributed:
            rs = DistributedReadingService()
        return rs

    def _get_dataloader(self) -> DataLoader2:
        if self.current_stage == 'train':
            random.Random(self.seed + self.current_epoch).shuffle(self.shards['train'])
        transforms = self.cpu_transforms[self.current_stage]
        dp = self._get_datapipe(self.current_stage, transforms)
        return DataLoader2(dp, reading_service=self._define_reading_service())

    def stage(self, stage: str):
        if stage == 'fit':
            self.current_epoch += 1

        if type(self)._update_cpu_transform is not BaseDataModule._update_cpu_transform:
            self.cpu_transforms = self._update_cpu_transform()
        if type(self)._update_gpu_transform is not BaseDataModule._update_gpu_transform:
            self.gpu_transforms = self._update_gpu_transform()

    def on_after_batch_transfer(self, batch: Tuple[Tensor, Tensor], dataloader_idx: int) -> Tuple[Tensor, Tensor]:
        target, frames = batch
        if self.current_stage in self.gpu_transforms:
            transform = self.gpu_transforms[self.current_stage]
            for i in range(len(frames)):
                frames[i] = transform(frames[i])
        return target, frames

    def train_dataloader(self) -> DataLoader2:
        self.current_stage = 'train'
        return self._get_dataloader()

    def val_dataloader(self) -> DataLoader2:
        self.current_stage = 'val'
        if self.val_loader is None:
            self.val_loader = self._get_dataloader()
        return self.val_loader

    def test_dataloader(self) -> DataLoader2 | None:
        if self.test:
            self.current_stage = 'test'
            if self.test_loader is None:
                self.test_loader = self._get_dataloader()
            return self.test_loader


class SSv2(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_frames = 30
        self.short_side = 240
        self.crop_size = 224

        self.cpu_transforms = {
            'train': [
                SampleFrames(self.num_frames, 'random'),
                DecodeFrames(),
                T.RandomCrop(self.short_side),
                T.Resize(size=self.crop_size, interpolation=InterpolationMode.BILINEAR, antialias=True),
            ],
            'val': [
                SampleFrames(self.num_frames, 'uniform'),
                DecodeFrames(),
            ]
        }
        self.gpu_transforms = {
            'train': Sequential(
                T.RandomHorizontalFlip(),
                T.RandAugment(),
                T.ConvertImageDtype(float32),
                T.Normalize(*norm_info['ssv2']),
            ),
            'val': Sequential(
                T.CenterCrop(self.short_side),
                T.Resize(self.crop_size),
                T.ConvertImageDtype(float32),
                T.Normalize(*norm_info['ssv2']),
            )
        }
