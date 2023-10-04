from typing import Tuple, Union

import numpy as np
from numpy import ndarray
import numba
from torch import Tensor


@numba.jit(nopython=True, cache=True, parallel=True, nogil=True)
def _horizontal_flip(video: ndarray):
    for t in numba.prange(video.shape[0] * video.shape[1]):
        video[t // video.shape[1], t % video.shape[1]] = np.fliplr(video[t // video.shape[1], t % video.shape[1]])
    return video


def _cutout(video: Tensor, num_crops: int, crop_size_range: tuple, fill: int = 0):
    T, C, H, W = video.shape
    for _ in range(num_crops):
        crop_height = np.random.randint(crop_size_range[0], crop_size_range[1] + 1)
        crop_width = np.random.randint(crop_size_range[0], crop_size_range[1] + 1)

        coord = (
            np.random.randint(H - crop_height + 1),
            np.random.randint(W - crop_width + 1),
        )

        video[:, :, coord[0]:coord[0] + crop_height, coord[1]:coord[1] + crop_width] = fill

    return video


class Cutout:
    def __init__(self, num_crops: int, crop_size_range: tuple, fill: int = 0):
        self.num_crops = num_crops
        self.crop_size_range = crop_size_range
        self.fill = fill

    def __call__(self, video: Union[Tensor, Tuple[Tensor, ...]]) -> Union[Tensor, Tuple[Tensor, ...]]:
        if isinstance(video, tuple):
            vid = _cutout(video[-1], self.num_crops, self.crop_size_range, self.fill)
            return *video[:-1], vid
        return _cutout(video, self.num_crops, self.crop_size_range, self.fill)


class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, video: Union[ndarray, Tuple[ndarray, ...]]) -> Union[ndarray, Tuple[ndarray, ...]]:
        if np.random.random() < self.p:
            if isinstance(video, tuple):
                return *video[:-1], _horizontal_flip(video[-1])
            return _horizontal_flip(video)
        return video
