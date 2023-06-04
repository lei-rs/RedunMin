from typing import List, Dict, Callable, Iterator, Tuple

from torch import Tensor
from torchdata.datapipes import functional_datapipe
from torchdata.datapipes.iter import IterDataPipe, FileOpener

from .constants import NUM_FRAMES, TARGET


class Sequential(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, callables: List[Callable]):
        self.datapipe = datapipe
        self.callables = callables

    def _call(self, item):
        for module in self.callables:
            item = module(item)
        return item

    def __iter__(self) -> Iterator[IterDataPipe]:
        return iter(self.datapipe.map(self._call))


@functional_datapipe('read')
class ReadToMemory(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe):
        self.datapipe = datapipe

    @staticmethod
    def _decode(item):
        key, value = item
        if key.endswith('.cls') or key.endswith(NUM_FRAMES):
            value = int(value.read())
        elif key.endswith('.jpeg'):
            value = value.read()
        return key, value

    @staticmethod
    def _split_key(sample: Dict) -> Dict:
        sample['__key__'] = sample['__key__'].split('/')[-1]
        return sample

    def __iter__(self) -> Iterator[IterDataPipe]:
        dp = FileOpener(self.datapipe, mode='rb').load_from_tar().map(self._decode).webdataset()
        for sample in dp:
            yield self._split_key(sample)


class SplitWDSample(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict]):
        self.datapipe = datapipe

    @staticmethod
    def _split(sample: Dict):
        key = sample.pop('__key__')
        target = sample.pop(TARGET)
        return key, target, sample

    def __iter__(self) -> Iterator[IterDataPipe[Tuple[str, int, Dict]]]:
        return iter(self.datapipe.map(self._split).unzip(3))


@functional_datapipe('spdp')
class SingleProcDP(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict], transforms: List[Callable] = None):
        keys, targets, images = SplitWDSample(datapipe)
        if transforms is not None:
            images = Sequential(images, transforms)
        self.pipe_out = keys.zip(targets, images)

    def __iter__(self) -> Iterator[Tuple[str, int, Tensor]]:
        return iter(self.pipe_out)
