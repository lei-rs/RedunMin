from typing import Dict, Literal

import numpy as np
from imagecodecs import jpeg_decode
from numpy import stack, uint8
from torch import Tensor, from_numpy

from .constants import NUM_FRAMES, FRAME


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
            for i in range(num_frames, self.frames_out):
                sample[FRAME.format(i)] = sample[FRAME.format(num_frames - 1)]
            num_frames = self.frames_out

        indices = np.linspace(0, num_frames, self.frames_out + 1, dtype=int)
        if self.mode == 'random':
            indices = [np.random.randint(indices[i], indices[i + 1]) for i in range(self.frames_out)]
        elif self.mode == 'uniform':
            indices = indices[:-1]
        ret = {FRAME.format(i): sample[FRAME.format(idx)] for i, idx in enumerate(indices)}
        ret[NUM_FRAMES] = self.frames_out
        return ret


class DecodeFrames:
    def __call__(self, sample: Dict) -> Tensor:
        images = [bytes(sample[FRAME.format(i)]) for i in range(sample[NUM_FRAMES])]
        return from_numpy(stack([jpeg_decode(image) for image in images], dtype=uint8)).permute(0, 3, 1, 2).contiguous()
