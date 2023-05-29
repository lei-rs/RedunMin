import io
from typing import Dict, Literal, List

import numpy as np
from torch import Tensor, from_numpy

from .constants import NUM_FRAMES


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

    def __call__(self, x: Tensor) -> Tensor:
        return x.to(self.device)


class SampleFrames:
    def __init__(self, num_frames: int, mode: Literal['random', 'uniform'] = 'random'):
        self.frames_out = num_frames
        self.mode = mode

    def __call__(self, sample: Dict) -> Dict:
        num_frames = sample[NUM_FRAMES]

        if self.frames_out > num_frames:
            return sample

        indices = np.linspace(0, num_frames, self.frames_out + 1, dtype=int)
        if self.mode == 'random':
            indices = [np.random.randint(indices[i], indices[i + 1]) for i in range(self.frames_out)]
        elif self.mode == 'uniform':
            indices = indices[:-1]
        ret = {gfn(i): sample[gfn(idx)] for i, idx in enumerate(indices)}
        ret[NUM_FRAMES] = self.frames_out
        return ret


class ReadFrames:
    def __call__(self, sample: Dict) -> List:
        images = []
        for i in range(sample[NUM_FRAMES]):
            stream = io.BytesIO(sample[gfn(i)].read())
            ar = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            images.append(ar)
        return images
