import numpy as np

from src.data.transforms import *


jax.config.update("jax_default_device", jax.devices('cpu')[0])


def test_bench_flip_horizontal(benchmark):
    test = jnp.asarray(np.random.randint(0, 255, (32, 32, 3, 224, 224), dtype=np.uint8))
    flip = FlipHorizontal()
    flip(test, key=jax.random.PRNGKey(0))
    benchmark(flip, test, key=jax.random.PRNGKey(0))


def test_bench_shear(benchmark):
    test = jnp.asarray(np.random.randint(0, 255, (32, 32, 3, 224, 224), dtype=np.uint8))
    shear = Shear('x', (0, 1))
    shear(test, key=jax.random.PRNGKey(0))
    benchmark(shear, test, key=jax.random.PRNGKey(0))


def test_bench_rotate(benchmark):
    test = jnp.asarray(np.random.randint(0, 255, (32, 32, 3, 224, 224), dtype=np.uint8))
    rotate = Rotate((0, 360))
    rotate(test, key=jax.random.PRNGKey(0))
    benchmark(rotate, test, key=jax.random.PRNGKey(0))


def test_bench_invert(benchmark):
    test = jnp.asarray(np.random.randint(0, 255, (32, 32, 3, 224, 224), dtype=np.uint8))
    invert = Invert()
    invert(test, key=jax.random.PRNGKey(0))
    benchmark(invert, test, key=jax.random.PRNGKey(0))


def test_bench_trivial_aug(benchmark):
    test = jnp.asarray(np.random.randint(0, 255, (32, 32, 3, 224, 224), dtype=np.uint8))
    key = jax.random.PRNGKey(0)
    aug = TrivialAugment([
        FlipHorizontal(),
        Shear('x', (0, 0.99)),
        Shear('y', (0, 0.99)),
        Rotate((0, 135.0)),
        Invert(),
    ], key=key)
    aug.warmup(test)
    benchmark(aug, test)
