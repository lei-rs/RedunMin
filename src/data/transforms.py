import random
from functools import partial
from typing import Tuple, Union, List, Literal

import jax
import jax.numpy as jnp
from jax.numpy import ndarray
from jax.scipy.ndimage import map_coordinates


class Augment:
    @staticmethod
    def _t():
        raise NotImplementedError

    def _call(self, video: ndarray, *, key=None) -> ndarray:
        raise NotImplementedError

    def __call__(self, video: Union[ndarray, Tuple[ndarray, ...]], *, key=None) -> Union[ndarray, Tuple[ndarray, ...]]:
        if isinstance(video, tuple):
            vid = self._call(video[-1], key=key)
            return *video[:-1], vid
        return self._call(video, key=key)

    def update_strength(self, strength: float):
        raise NotImplementedError


class Cutout(Augment):
    def __init__(self, max_num_crops: int, crop_size_range: tuple, fill: int = 0):
        self.max_num_crops = max_num_crops
        self.num_crops = max_num_crops // 2
        self.crop_size_range = crop_size_range
        self.fill = fill

    @staticmethod
    def _cutout(video: ndarray, num_crops: int, crop_size_range: tuple, *, key, fill: int = 0):
        T, C, H, W = video.shape
        key = jax.random.split(key, 4)
        crop_height = jax.random.randint(
            key[0],
            [num_crops],
            crop_size_range[0],
            crop_size_range[1] + 1,
        )
        crop_width = jax.random.randint(
            key[1],
            [num_crops],
            crop_size_range[0],
            crop_size_range[1] + 1,
        )
        coord_h = jax.random.randint(
            key[2],
            [num_crops],
            0,
            H - crop_height
        )
        coord_w = jax.random.randint(
            key[3],
            [num_crops],
            0,
            W - crop_width
        )
        for i in range(num_crops):
            video.at[:, :, coord_h[0]:coord_h[0] + crop_height[i], coord_w[1]:coord_w[1] + crop_width[i]].set(fill)

        return video

    def _call(self, video: ndarray, *, key) -> ndarray:
        return self._cutout(
            video,
            self.num_crops,
            self.crop_size_range,
            key=key,
            fill=self.fill
        )

    def update_strength(self, strength: float):
        self.num_crops = int(self.max_num_crops * strength)


class FlipHorizontal(Augment):
    @staticmethod
    @jax.jit
    def _t(video: ndarray):
        return video[:, :, :, ::-1]

    def _call(self, video: ndarray, *, key) -> ndarray:
        if len(video.shape) > 4:
            return jax.vmap(self._t)(video)
        return self._t(video)

    def update_strength(self, strength: float):
        return


class Shear(Augment):
    def __init__(self, axis: Literal['x', 'y'], factor_range: tuple):
        self.axis = axis
        self.factor_range = factor_range
        self.factor = sum(factor_range) / 2

        self._t = jax.jit(partial(self._t, axis=axis))

    @staticmethod
    def _t(video: ndarray, axis: Literal['x', 'y'], factor: float):
        T, C, H, W = video.shape
        y_coords, x_coords = jnp.mgrid[:H, :W]
        if axis == 'x':
            x_coords = x_coords + y_coords * factor
        elif axis == 'y':
            y_coords = y_coords + x_coords * factor
        else:
            raise ValueError(f'Axis must be one of x or y, got {axis}')
        video_reshaped = video.reshape(T * C, H, W)
        func = partial(map_coordinates, coordinates=[x_coords, y_coords], order=0)
        sheared = jax.vmap(func)(video_reshaped)
        return sheared.reshape(T, C, H, W)

    def _call(self, video: ndarray, *, key) -> ndarray:
        if len(video.shape) > 4:
            return jax.vmap(partial(self._t, factor=self.factor))(video)
        return self._t(video, factor=self.factor)

    def update_strength(self, strength: float):
        self.factor = (self.factor_range[1] - self.factor_range[0]) * strength


class Rotate(Augment):
    def __init__(self, angle_range: tuple):
        self.angle_range = angle_range
        self.angle = sum(angle_range) / 2

    @staticmethod
    @jax.jit
    def _t(video: ndarray, angle: float):
        T, C, H, W = video.shape
        y_coords, x_coords = jnp.mgrid[:H, :W]
        angle = jnp.deg2rad(angle)
        x_coords = x_coords - W / 2
        y_coords = y_coords - H / 2
        x_coords = x_coords * jnp.cos(angle) - y_coords * jnp.sin(angle)
        y_coords = x_coords * jnp.sin(angle) + y_coords * jnp.cos(angle)
        x_coords = x_coords + W / 2
        y_coords = y_coords + H / 2
        video_reshaped = video.reshape(T * C, H, W)
        func = partial(map_coordinates, coordinates=[x_coords, y_coords], order=0)
        rotated = jax.vmap(func)(video_reshaped)
        return rotated.reshape(T, C, H, W)

    def _call(self, video: ndarray, *, key) -> ndarray:
        if len(video.shape) > 4:
            return jax.vmap(partial(self._t, angle=self.angle))(video)
        return self._t(video, self.angle)

    def update_strength(self, strength: float):
        self.angle = (self.angle_range[1] - self.angle_range[0]) * strength


class Invert(Augment):
    @staticmethod
    @jax.jit
    def _t(video: ndarray):
        return 255 - video

    def _call(self, video: ndarray, *, key) -> ndarray:
        if len(video.shape) > 4:
            return jax.vmap(self._t)(video)
        return self._t(video)

    def update_strength(self, strength: float):
        return


class TrivialAugment:
    def __init__(self, augments: List[Augment], bins: int = 20, *, key, debug=False):
        self.augments = augments
        self.strengths = jnp.linspace(0, 1, bins)
        self.key = key
        self.debug = debug

    def __call__(self, video: ndarray) -> ndarray:
        self.key, key_aug = jax.random.split(self.key)
        aug = random.choice(self.augments)
        strength = random.choice(self.strengths)
        aug.update_strength(strength)
        if self.debug:
            print(f'Augment: {aug.__class__.__name__}, Strength: {strength}')
        return aug(video, key=key_aug)

    def warmup(self, video: ndarray):
        for strength in self.strengths:
            for aug in self.augments:
                aug.update_strength(strength)
                aug(video, key=self.key)

    @classmethod
    def default(cls, *, key):
        augs = [
            FlipHorizontal(),
            Shear('x', (0, 0.99)),
            Shear('y', (0, 0.99)),
            Rotate((0, 135.0)),
            Invert(),
        ]
        return cls(augs, key=key)


class Normalize(Augment):
    def __init__(self, mean: ndarray, std: ndarray, dtype: jnp.dtype = jnp.bfloat16):
        self.mean = mean.astype(dtype)
        self.std = std.astype(dtype)

    @staticmethod
    @jax.jit
    def _t(video: ndarray, mean: ndarray, std: ndarray, dtype: jnp.dtype):
        video = video.astype(dtype)
        return (video - mean) / std

    def _call(self, video: ndarray, *, key=None) -> ndarray:
        return self._t(video, self.mean, self.std)

    @classmethod
    def default(cls):
        return cls(
            jnp.array([0.485, 0.456, 0.406]),
            jnp.array([0.229, 0.224, 0.225])
        )

    def warmup(self, video: ndarray):
        self._call(video)