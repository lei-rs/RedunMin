from multiprocessing.shared_memory import ShareableList
from typing import List, Dict, Callable, Iterator, Tuple

from pytarrs import GroupedPyReader
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe


@functional_datapipe('seq')
class Sequential(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, callables: List[Callable]):
        self.datapipe = datapipe
        self.callables = callables

    def _call(self, item):
        for module in self.callables:
            item = module(item)
        return item

    def __iter__(self) -> Iterator[IterDataPipe]:
        for sample in self.datapipe:
            yield self._call(sample)


@functional_datapipe('read_tar')
class TarToWDS(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[str]):
        self.datapipe = datapipe

    @staticmethod
    def proc_sample(sample):
        key = sample.pop('__key__')
        target = sample.pop('__target__')
        frames = sample.pop('frames')
        frames = ShareableList(frames, name=key)
        return [key, target, frames]

    def __iter__(self) -> Iterator[IterDataPipe[Tuple[str, bytes]]]:
        for path in self.datapipe:
            for sample in GroupedPyReader(path):
                yield self.proc_sample(sample)


class QueuePopper(IterDataPipe):
    def __init__(self, queue):
        self.queue = queue

    def __iter__(self):
        while True:
            sample = self.queue.get()
            if sample is StopIteration:
                raise StopIteration
            yield tuple(sample)
