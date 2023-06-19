from typing import List, Dict, Callable, Iterator, Tuple

from numpy import frombuffer
from torch import Tensor, from_numpy
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, FileOpener

from .constants import NUM_FRAMES, TARGET


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
        yield from self.datapipe.map(self._call)


@functional_datapipe('wds')
class TarToWDS(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe):
        self.datapipe = datapipe

    def __iter__(self) -> Iterator[IterDataPipe]:
        dp = FileOpener(self.datapipe, mode='rb').load_from_tar().webdataset()
        yield from dp


@functional_datapipe('read')
class ReadStream(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe):
        self.datapipe = datapipe

    @staticmethod
    def _decode(item):
        key, value = item
        if key.endswith('.cls') or key.endswith(NUM_FRAMES):
            value = int(value.read())
        elif key.endswith('.jpeg'):
            value = bytes(value.read())
        return key, value

    def _decode_sample(self, sample: Dict) -> Dict:
        sample['__key__'] = sample['__key__'].split('/')[-1]
        for i in sample.items():
            k, v = self._decode(i)
            sample[k] = v
        return sample

    def __iter__(self) -> Iterator[IterDataPipe]:
        yield from self.datapipe.map(self._decode_sample)


class SplitWDSample(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict]):
        self.datapipe = datapipe

    @staticmethod
    def _split(sample: Dict):
        key = sample.pop('__key__')
        target = sample.pop(TARGET)
        return key, target, sample

    def __iter__(self) -> Iterator[IterDataPipe[Tuple[str, int, Dict]]]:
        yield from self.datapipe.map(self._split).unzip(3)


@functional_datapipe('spdp')
class SingleProcDP(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict], transforms: List[Callable] = None):
        keys, targets, images = SplitWDSample(datapipe)
        if transforms is not None:
            images = Sequential(images, transforms)
        self.pipe_out = keys.zip(targets, images)

    def __iter__(self) -> Iterator[Tuple[str, int, Tensor]]:
        yield from self.pipe_out
