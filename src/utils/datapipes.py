from typing import List, Dict, Callable, Iterable

from torchdata.datapipes.iter import IterDataPipe, FileOpener

from constants import NUM_FRAMES, TARGET


def decode_md(item):
    key, value = item
    if key.endswith('.cls') or key.endswith(NUM_FRAMES):
        value = int(value.read())
    return key, value


def decode_wd_key(sample: Dict) -> Dict:
    sample['__key__'] = sample['__key__'].split('/')[-1]
    return sample


class Sequential(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe, callables: List[Callable]):
        self.datapipe = datapipe
        self.callables = callables

    def _call(self, item):
        for module in self.callables:
            item = module(item)
        return item

    def __iter__(self):
        return iter(self.datapipe.map(self._call))


class SplitWDSample(IterDataPipe):
    def __init__(self, datapipe: IterDataPipe[Dict]):
        self.datapipe = datapipe

    @staticmethod
    def _split(sample: Dict):
        key = sample.pop('__key__')
        target = sample.pop(TARGET)
        return key, target, sample

    def __iter__(self):
        return iter(self.datapipe.map(self._split).unzip(3))


class SingleWorkerDataset(IterDataPipe):
    def __init__(self, shards: Iterable[str], transforms: List[Callable] = None):
        raw_data = FileOpener(shards, mode='rb').load_from_tar().map(decode_md).webdataset().map(decode_wd_key)
        keys, targets, images = SplitWDSample(raw_data)
        if transforms is not None:
            images = Sequential(images, transforms)
        self.pipe_out = keys.zip(targets, images)

    def __iter__(self):
        return iter(self.pipe_out)
