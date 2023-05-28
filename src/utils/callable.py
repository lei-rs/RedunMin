import io
from typing import Dict, Literal

import numpy as np
from PIL import Image
from torch import Tensor

NORM = {
  'imagenet': {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225]
    }
}


def gfn(idx: int):
    """
    Get frame name from index
    :param idx:
    :return str:
    """
    return f'.frame_{idx:06d}.jpeg'


class ToDevice:
    def __init__(self, device: str = 'cpu'):
        self.device = device

    def __call__(self, x: Tensor):
        return x.to(self.device)


class SampleFrames:
    def __init__(self, num_frames: int, mode: Literal['random', 'uniform'] = 'random'):
        self.frames_out = num_frames
        self.mode = mode

    def __call__(self, sample: Dict):
        num_frames = sample['.num_frames']

        if self.frames_out > num_frames:
            return sample

        indices = np.linspace(0, num_frames, self.frames_out + 1, dtype=int)
        if self.mode == 'random':
            indices = [np.random.randint(indices[i], indices[i + 1]) for i in range(self.frames_out)]
        elif self.mode == 'uniform':
            indices = indices[:-1]
        ret = {gfn(i): sample[gfn(idx)] for i, idx in enumerate(indices)}
        ret['.num_frames'] = self.frames_out
        return ret


class FramesToArray:
    def __call__(self, sample: Dict):
        images = []
        for i in range(sample['.num_frames']):
            stream = io.BytesIO(sample[gfn(i)].read())
            images.append(np.asarray(Image.open(stream), dtype=np.uint8))
        return np.transpose(np.asarray(images), (0, 3, 1, 2))
