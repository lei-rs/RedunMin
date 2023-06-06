import random
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
from torch.utils.data.datapipes.iter.sharding import SHARDING_PRIORITIES


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
            test: bool,
            path: str,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.distributed = distributed
        self.prefetch_count = prefetch_count
        self.test = test

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
    def _batch(batch: List[Tuple[str, int, Tensor]]) -> Tuple[List, Tensor, List]:
        keys, targets, frames = map(list, zip(*batch))
        targets = torch.from_numpy(np.asarray(targets))
        return keys, targets, frames

    def _update_cpu_transform(self) -> List[Callable]:
        # If you want to update per stage
        pass

    def _update_gpu_transform(self) -> Module:
        # If you want to update per stage
        pass

    def _get_datapipe(self, task: str, transforms: List[Callable]) -> IterDataPipe:
        dp: IterDataPipe = IterableWrapper(self.shards[task]).read()
        if self.distributed > 1:
            dp = dp.sharding_round_robin_dispatch(SHARDING_PRIORITIES.MULTIPROCESSING).sharding_filter()
        if task == 'train':
            dp = dp.shuffle(buffer_size=1000)
        dp = dp.spdp(transforms=transforms)
        dp = dp.batch(batch_size=self.batch_size, drop_last=True, wrapper_class=self._batch)
        if self.distributed > 1:
            dp = dp.fullsync()
        return dp.prefetch(self.prefetch_count)

    def _define_reading_service(self) -> ReadingServiceInterface:
        rs: Optional[ReadingServiceInterface] = None
        if self.distributed:
            rs = DistributedReadingService()
        return rs

    def stage(self, stage: str):
        if stage == 'fit':
            self.current_epoch += 1

        if type(self)._update_cpu_transform is not BaseDataModule._update_cpu_transform:
            self.cpu_transforms = self._update_cpu_transform()
        if type(self)._update_gpu_transform is not BaseDataModule._update_gpu_transform:
            self.gpu_transforms = self._update_gpu_transform()

    def on_after_batch_transfer(self, batch: Tuple[List, Tensor, List], dataloader_idx: int) -> Tuple[List, Tensor, Tensor]:
        key, target, frames = batch
        transform = self.gpu_transforms[self.current_stage]
        if transform is not None:
            frames = [transform(frame) for frame in frames]
        frames = torch.stack(frames)
        return key, target, frames

    def train_dataloader(self) -> DataLoader2:
        self.current_stage = 'train'
        random.seed(self.current_epoch)
        random.shuffle(self.shards['train'])
        transforms = self.cpu_transforms['train']
        dp = self._get_datapipe('train', transforms)
        self.train_loader = DataLoader2(dp, reading_service=self._define_reading_service())
        self.train_loader.seed(self.current_epoch)
        return self.train_loader

    def val_dataloader(self) -> DataLoader2:
        self.current_stage = 'val'
        if self.val_loader is None:
            transforms = self.cpu_transforms['val']
            dp = self._get_datapipe('val', transforms)
            self.val_loader = DataLoader2(dp, reading_service=self._define_reading_service())
        return self.val_loader

    def test_dataloader(self) -> DataLoader2 | None:
        if self.test:
            self.current_stage = 'test'
            if self.test_loader is None:
                transforms = self.cpu_transforms['test']
                dp = self._get_datapipe('test', transforms)
                self.test_loader = DataLoader2(dp, reading_service=self._define_reading_service())
            return self.test_loader


class SSv2(BaseDataModule):
    def __init__(
            self,
            batch_size: int,
            distributed: bool,
            prefetch_count: int,
            path: str,
            test: bool = False,
    ):
        super().__init__(
            batch_size=batch_size,
            distributed=distributed,
            prefetch_count=prefetch_count,
            test=test,
            path=path,
        )
        self.num_frames = 30
        self.short_side = 240
        self.crop_size = 224

        self.cpu_transforms = {
            'train': [
                SampleFrames(self.num_frames, 'random'),
                DecodeFrames(),
            ],
            'val': [
                SampleFrames(self.num_frames, 'uniform'),
                DecodeFrames(),
            ]
        }
        self.gpu_transforms = {
            'train': Sequential(
                    T.RandomCrop(self.short_side),
                    T.Resize(self.crop_size),
                    T.RandomHorizontalFlip(),
                    T.RandAugment(),
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(*norm_info['ssv2']),
            ),
            'val': Sequential(
                    T.CenterCrop(self.short_side),
                    T.Resize(self.crop_size),
                    T.ConvertImageDtype(torch.float32),
                    T.Normalize(*norm_info['ssv2']),
            )
        }
