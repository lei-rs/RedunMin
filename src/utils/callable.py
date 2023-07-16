from multiprocessing.shared_memory import ShareableList
from typing import Literal, Tuple, List
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from imagecodecs import jpeg_decode


def copy_shared_list(sample: List):
    key, target, l = sample
    ar = np.copy(l)
    l.shm.close()
    l.shm.unlink()
    return key, target, ar


class SampleFrames:
    def __init__(self, num_frames: int, mode: Literal['random', 'uniform'] = 'random'):
        self.frames_out = num_frames
        self.mode = mode

    @staticmethod
    def _sample(frames_out: int, frames: NDArray, mode: Literal['random', 'uniform']) -> NDArray:
        num_frames = len(frames)
        if frames_out > num_frames:
            last = [frames[-1]] * (frames_out - num_frames)
            return np.append(frames, last)

        indices = np.linspace(0, num_frames, frames_out + 1, dtype=int)
        if mode == 'random':
            indices = np.random.randint(indices[:-1], indices[1:], frames_out)
        elif mode == 'uniform':
            indices = indices[:-1]
        return frames[indices]

    def __call__(self, sample: Tuple[str, int, NDArray]) -> Tuple[str, int, NDArray]:
        key, target, frames = sample
        frames = self._sample(self.frames_out, frames, self.mode)
        return key, target, frames


class DecodeFrames:
    def __init__(self):
        self.pool = None

    def _start_pool(self):
        self.pool = ThreadPoolExecutor(4)

    def __call__(self, sample: Tuple[str, int, NDArray[bytes]]) -> Tuple[str, int, NDArray]:
        if self.pool is None:
            self._start_pool()
        key, target, frames = sample
        frames = list(self.pool.map(jpeg_decode, frames))
        frames = np.asarray(frames, dtype=np.uint8).transpose((3, 0, 1, 2))
        return key, target, frames
