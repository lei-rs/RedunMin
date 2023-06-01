import os
from pathlib import Path
from typing import List, Callable, Optional, Tuple, Dict, Set

import numpy as np
import torch
import torchvision.transforms as T
from lightning import LightningDataModule
from torch import Tensor
from torch.nn import Module, Sequential
from torchdata.dataloader2 import DistributedReadingService, DataLoader2, ReadingServiceInterface
from torchdata.datapipes.iter import IterDataPipe, IterableWrapper

from .callable import SampleFrames, DecodeFrames

norm_info = {
    'ssv2': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
}


class BaseDataModule(LightningDataModule):
    def __init__(
            self,
            batch_size: int,
            distributed: bool,
            prefetch_count: int,
            tasks: Set[str],
            path: str,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.distributed = distributed
        self.prefetch_count = prefetch_count
        self.tasks = tasks
        self.path = Path(path)
        self.shards: Dict[str, IterDataPipe] = {}

        # Trackers
        self.current_epoch = 0
        self.current_stage = None

        # Initialized per stage
        self.cpu_transforms: Optional[List[Callable]] = None
        self.gpu_transform: Optional[Module] = None
        self.dp: Optional[IterDataPipe] = None

        # Initialized once, updated per stage
        self.train_loader: Optional[DataLoader2] = None
        self.val_loader: Optional[DataLoader2] = None
        self.test_loader: Optional[DataLoader2] = None

    @staticmethod
    def _batch(batch: List[Tuple[str, int, Tensor]]) -> Tuple[List, Tensor, List]:
        keys, targets, frames = map(list, zip(*batch))
        targets = torch.from_numpy(np.asarray(targets))
        return keys, targets, frames

    def _get_cpu_transform(self) -> List[Callable]:
        # Implement per dataset
        pass

    def _get_gpu_transform(self) -> Module:
        # Implement per dataset
        pass

    def _get_datapipe(self) -> IterDataPipe:
        dp: IterDataPipe = self.shards[self.current_stage].read()
        if self.current_stage == 'train':
            dp = dp.shuffle(buffer_size=1000)
        if self.distributed > 1:
            dp = dp.sharding_filter()
        dp = dp.spdp(transforms=self.cpu_transforms)
        dp = dp.batch(batch_size=self.batch_size, drop_last=True, wrapper_class=self._batch)
        if self.distributed > 1:
            dp = dp.fullsync()
        return dp.prefetch(self.prefetch_count)

    def _define_reading_service(self) -> ReadingServiceInterface:
        rs: Optional[ReadingServiceInterface] = None
        if self.distributed:
            rs = DistributedReadingService()
        return rs

    def prepare_data(self):
        shards = {task: list((self.path / task).glob('*.tar')) for task in self.tasks}
        self.shards = {task: IterableWrapper(shards[task]).map(str) for task in self.tasks}
        if 'train' in self.tasks:
            self.shards['train'] = self.shards['train'].shuffle(buffer_size=len(shards['train']))

    def setup(self, stage: str):
        if stage not in self.tasks:
            return

        if stage == 'fit':
            self.current_epoch += 1

        self.current_stage = stage

        if type(self)._get_cpu_transform != BaseDataModule._get_cpu_transform:
            self.cpu_transforms = self._get_cpu_transform()
        if type(self)._get_gpu_transform != BaseDataModule._get_gpu_transform:
            self.gpu_transform = self._get_gpu_transform()

        self.dp = self._get_datapipe()

    def on_after_batch_transfer(self, batch: Tuple[List, Tensor, List], dataloader_idx: int
                                ) -> Tuple[List, Tensor, List]:
        key, target, frames = batch
        if self.gpu_transform is not None:
            frames = [self.gpu_transform(frame) for frame in frames]
            frames = torch.stack(frames)
        return key, target, frames

    def train_dataloader(self) -> DataLoader2:
        if self.train_loader is None:
            self.train_loader = DataLoader2(self.dp, reading_service=self._define_reading_service())
        self.train_loader.seed(self.current_epoch)
        return self.train_loader

    def val_dataloader(self) -> DataLoader2:
        if self.val_loader is None:
            self.val_loader = DataLoader2(self.dp, reading_service=self._define_reading_service())
        return self.val_loader

    def test_dataloader(self) -> DataLoader2:
        if 'test' in self.tasks:
            if self.test_loader is None:
                self.test_loader = DataLoader2(self.dp, reading_service=self._define_reading_service())
            return self.test_loader


class SSv2(BaseDataModule):
    def __init__(
            self,
            batch_size: int,
            distributed: bool,
            prefetch_count: int,
            path: str,
            tasks: Set[str] = {'fit'},
    ):
        super().__init__(
            batch_size=batch_size,
            distributed=distributed,
            prefetch_count=prefetch_count,
            tasks=tasks,
            path=path,
        )
        self.num_frames = 30
        self.short_side = 240
        self.crop_size = 224

    def _get_cpu_transform(self) -> List[Callable]:
        if self.current_stage == 'fit':
            transform = [
                SampleFrames(self.num_frames, 'random'),
                DecodeFrames(),
                T.RandomCrop(self.short_side),
                T.Resize(self.crop_size),
                T.RandomHorizontalFlip(),
            ]
        else:
            transform = [
                SampleFrames(self.num_frames, 'uniform'),
                DecodeFrames(),
                T.CenterCrop(self.short_side),
                T.Resize(self.crop_size),
            ]
        return transform

    def _get_gpu_transform(self) -> Module:
        transform = None
        if self.current_stage == 'fit':
            transform = Sequential(
                T.RandAugment(),
                T.Normalize(*norm_info['ssv2']),
            )
        return transform
