from functools import partial
from typing import Literal, List

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array
from jax.numpy import ndarray
from jax.scipy.ndimage import map_coordinates


class Augment(eqx.Module):
    def __call__(self, video: ndarray, strength: float) -> ndarray:
        raise NotImplementedError


class FlipHorizontal(Augment):
    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, strength: float = None) -> ndarray:
        return video[:, :, :, ::-1]


class ShearAxis(eqx.Enumeration):
    x = 'x'
    y = 'y'


class Shear(Augment):
    factor_range: tuple[float, float] = eqx.static_field(default=(0, 0.99))
    axis: Literal['x', 'y'] = eqx.static_field(default='x')

    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, strength: float) -> ndarray:
        factor = (self.factor_range[1] - self.factor_range[0]) * strength + self.factor_range[0]
        T, C, H, W = video.shape
        y_coords, x_coords = jnp.mgrid[:H, :W]
        if self.axis == 'x':
            x_coords = x_coords + y_coords * factor
        elif self.axis == 'y':
            y_coords = y_coords + x_coords * factor
        else:
            raise ValueError(f'Axis must be one of x or y, got {self.axis}')
        video_reshaped = video.reshape(T * C, H, W)
        func = partial(map_coordinates, coordinates=[x_coords, y_coords], order=0)
        sheared = jax.vmap(func)(video_reshaped)
        return sheared.reshape(T, C, H, W)


class Rotate(Augment):
    angle_range: tuple[float, float] = eqx.static_field(default=(0, 135))

    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, strength: float) -> ndarray:
        angle = (self.angle_range[1] - self.angle_range[0]) * strength + self.angle_range[0]
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


class Invert(Augment):
    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, strength: float = None) -> ndarray:
        return 255 - video


class TrivialAugment(eqx.Module):
    strengths: Array
    augments: List[Augment] = eqx.static_field()

    def __init__(self, augments: List[Augment] = None, bins: int = 20):
        if augments is None:
            augments = [
                FlipHorizontal(),
                Shear(axis='x'),
                Shear(axis='y'),
                Rotate(),
                Invert(),
            ]
        self.augments = augments
        self.strengths = jnp.linspace(0, 1, bins)

    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, *, key) -> ndarray:
        key_a, key_s = jax.random.split(key)
        aug_idx = jax.random.choice(key_a, len(self.augments))
        strength = jax.random.choice(key_s, self.strengths)
        return jax.lax.switch(
            aug_idx,
            self.augments,
            video,
            strength
        )

    def warmup(self, video: ndarray):
        for strength in self.strengths:
            for aug in self.augments:
                aug(jnp.copy(video), strength=strength)
        self(jnp.copy(video), key=jax.random.PRNGKey(0))


class Normalize(eqx.Module):
    mean: Array = jnp.array([0.485, 0.456, 0.406])
    std: Array = jnp.array([0.229, 0.224, 0.225])

    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, *, key=None) -> ndarray:
        video = video.astype(self.mean.dtype)
        mean = jax.lax.broadcast_in_dim(self.mean, video.shape, (1,))
        std = jax.lax.broadcast_in_dim(self.std, video.shape, (1,))
        return (video - mean) / std

    def warmup(self, video: ndarray):
        self(jnp.copy(video))

    def astype(self, dtype: jnp.dtype) -> 'Normalize':
        return eqx.tree_at(
            lambda x: (x.mean, x.std),
            self,
            (self.mean.astype(dtype), self.std.astype(dtype))
        )


class Sequential(eqx.Module):
    transforms: List[eqx.Module] = eqx.static_field()

    @eqx.filter_jit(donate='all')
    def __call__(self, video: ndarray, *, key=None):
        for transform in self.transforms:
            video = transform(video, key=key)
        return video

    def warmup(self, video: ndarray):
        for transform in self.transforms:
            video = transform.warmup(jnp.copy(video))
        self(jnp.copy(video), key=jax.random.PRNGKey(0))
