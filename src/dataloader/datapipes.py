from queue import Queue
from typing import Callable

import av
import numpy as np
from torch.utils.data import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe

from .executor import LazyThreadPoolExecutor


class QueueIterator:
    def __init__(self, queue: Queue):
        self.queue = queue

    def __iter__(self):
        while True:
            item = self.queue.get(block=True)
            if item is StopIteration:
                break
            yield item


class ThreadedTransform(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, func: Callable, n_workers: int = 1, queue_size: int = 16):
        super().__init__()
        self.datapipe = datapipe
        self.func = func
        self.n_workers = n_workers
        self.queue_size = queue_size

    def __iter__(self):
        executor = LazyThreadPoolExecutor(self.n_workers, self.queue_size)
        executor.map(self.func, self.datapipe)
        yield from executor


@functional_datapipe('read_video')
class VideoReader(ThreadedTransform):
    def __init__(self, paths: IterDataPipe[str], n_workers: int = 1, queue_size: int = 8):
        super().__init__(paths, self._get_tensor, n_workers=n_workers, queue_size=queue_size)

    @staticmethod
    def _get_tensor(path):
        container = av.open(path)
        video_tensor = []
        for frame in container.decode(video=0):
            video_tensor.append(frame.to_ndarray(format='rgb24'))
        container.close()
        return np.asarray(video_tensor)
