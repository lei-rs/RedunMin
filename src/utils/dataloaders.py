import os
from typing import List, Callable, Iterator, Tuple
from typing import Optional

import numpy as np
import torchvision.transforms as T
from lightning import Fabric
from torch import Tensor, from_numpy
from torchdata.dataloader2 import MultiProcessingReadingService, DataLoader2
from torchdata.datapipes.iter import IterDataPipe
from torchdata.datapipes.iter import IterableWrapper

from callable import FramesToTensor, SampleFrames
from datapipes import SingleWorkerDataset


class Dataloader:
    def __init__(
            self,
            shards: List[str],
            batch_size: int,
            shuffle: bool = True,
            transforms: Optional[List[Callable]] = None,
            num_workers: int = 1,
            prefetch_count: int = 10,
            global_seed: Optional[int] = None
    ):
        assert batch_size > 0
        assert num_workers >= 1
        assert prefetch_count >= 1

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.transforms = transforms
        self.num_workers = num_workers
        self.prefetch_count = prefetch_count

        if global_seed is None:
            self.global_seed = np.random.randint(0, 2 ** 32 - 1)
        else:
            self.global_seed = global_seed

        self.shards = IterableWrapper(shards).shuffle()
        self.dl = DataLoader2(self.get_single_worker_dataset(), reading_service=self.get_reading_service())

    def get_single_worker_dataset(self) -> IterDataPipe:
        dp = self.shards.sharding_filter()
        dp = SingleWorkerDataset(dp, self.transforms)
        return dp

    def get_reading_service(self) -> MultiProcessingReadingService:
        if self.num_workers > 1:
            worker_prefetch_cnt = int(self.prefetch_count * self.batch_size / self.num_workers)
            return MultiProcessingReadingService(num_workers=self.num_workers,
                                                 worker_prefetch_cnt=worker_prefetch_cnt,
                                                 main_prefetch_cnt=1)
        else:
            raise NotImplementedError

    @staticmethod
    def numpy_batcher(batch_list: List[Tensor]) -> Tensor:
        np_data: np.ndarray = np.empty(len(batch_list), dtype=object)
        for i, ar in enumerate(batch_list):
            np_data[i] = ar.numpy()
        return from_numpy(np.stack(np_data))

    def get_wrapped_dl(self) -> IterDataPipe:
        dl = IterableWrapper(self.dl)
        if self.shuffle:
            dl = dl.shuffle(buffer_size=self.prefetch_count * self.batch_size)
        else:
            dl = dl.prefetch(self.prefetch_count * self.batch_size)
        if self.batch_size > 1:
            dl = dl.batch(self.batch_size, drop_last=True)
        return dl

    def __iter__(self) -> Iterator[Tuple[str, int, Tensor]]:
        return iter(self.get_wrapped_dl())


class DataModule:
    def __init__(
            self,
            dataset: str,
            path: str,
            frame_sampler: SampleFrames,
            batch_size: int,
            num_workers: int = 1,
            prefetch_count: int = 10,
            crop_size: int = 224
    ):
        super().__init__()
        self.dataset = dataset
        self.path = path
        self.frame_sampler = frame_sampler
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.prefetch_count = prefetch_count
        self.crop_size = crop_size

        self.frame_to_tensor = FramesToTensor()

        # Not Initialized
        self.shard_path = None
        self.worker_transform = None
        self.gpu_transform = None

    def setup_stage(self, stage: str):
        shard_path = os.listdir(os.path.join(self.path, self.dataset, stage))
        self.shard_path = [os.path.join(self.path, self.dataset, stage, p) for p in shard_path]

        if stage == 'train':
            self.worker_transform = [
                self.frame_sampler,
                self.frame_to_tensor,
                T.RandomCrop(self.crop_size),
                T.RandomHorizontalFlip(),
            ]
        elif stage in ('val', 'test'):
            self.worker_transform = [
                self.frame_sampler,
                self.frame_to_tensor,
                T.CenterCrop(self.crop_size),
            ]

    def get_dataloader(self) -> Dataloader:
        assert self.shard_path is not None and self.worker_transform is not None, 'Setup method has not been called!'
        return Dataloader(
            shards=self.shard_path,
            batch_size=self.batch_size,
            transforms=self.worker_transform,
            num_workers=min(self.num_workers, len(self.shard_path)),
            prefetch_count=self.prefetch_count
        )

    @staticmethod
    def transfer_batch_to_device(batch: Tuple[str, int, Tensor], fabric: Fabric) -> Tuple[str, int, Tensor]:
        return batch[0], batch[1], fabric.to_device(batch[2])

    def on_after_batch_transfer(self, batch: Tuple[str, int, Tensor]) -> Tuple[str, int, Tensor]:
        if self.gpu_transform is not None:
            batch = self.gpu_transform(batch[2])
        return batch
