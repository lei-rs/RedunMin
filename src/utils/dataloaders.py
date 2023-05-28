from typing import Optional

import lightning as L
import numpy as np
from torch import Tensor, from_numpy
from torchdata.dataloader2 import SequentialReadingService, MultiProcessingReadingService, DataLoader2
from torchdata.datapipes.iter import IterableWrapper

from datapipes import *


class Dataloader:
    def __init__(self, shards: List[str],
                 batch_size: int,
                 shuffle: bool = True,
                 transforms: Optional[List[Callable]] = None,
                 num_workers: int = 1,
                 prefetch_count: int = 10,
                 global_seed: Optional[int] = None):
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
        np.random.seed(self.global_seed)
        self.shards = IterableWrapper(shards).shuffle()
        self.dl = DataLoader2(self.get_single_worker_dataset(), reading_service=self.get_reading_service())

    def get_single_worker_dataset(self) -> IterDataPipe:
        dp = self.shards.sharding_filter()
        dp = SingleWorkerDataset(dp, self.transforms)
        return dp

    def get_reading_service(self) -> SequentialReadingService | MultiProcessingReadingService:
        if self.num_workers > 1:
            worker_prefetch_cnt = int(self.prefetch_count * self.batch_size / self.num_workers)
            return MultiProcessingReadingService(num_workers=self.num_workers,
                                                 worker_prefetch_cnt=worker_prefetch_cnt,
                                                 main_prefetch_cnt=1)
        else:
            return SequentialReadingService()

    @staticmethod
    def numpy_batcher(batch_list: List[np.array]) -> Tensor:
        np_data: np.ndarray = np.empty(len(batch_list), dtype=object)
        for i, ar in enumerate(batch_list):
            np_data[i] = ar
        return from_numpy(np.stack(np_data))

    def get_wrapped_dl(self) -> IterDataPipe:
        dl = IterableWrapper(self.dl)
        if self.shuffle:
            dl = dl.shuffle(buffer_size=self.prefetch_count * self.batch_size)
        else:
            dl = dl.prefetch(self.prefetch_count * self.batch_size)
        if self.batch_size > 1:
            dl = dl.batch(self.batch_size)
        return dl

    def __iter__(self) -> Tensor:
        for x in self.get_wrapped_dl():
            yield x


class DataModule(L.LightningDataModule):
    pass
