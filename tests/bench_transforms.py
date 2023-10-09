import numpy as np

from src.data.transforms import *


def test_bench_cutout(benchmark):
    test = jnp.array(np.random.randint(0, 255, (32, 3, 224, 224), dtype=np.uint8))
    cutout = Cutout(6, (8, 32))
    cutout(test, key=jax.random.PRNGKey(0))
    benchmark(cutout, test, key=jax.random.PRNGKey(0))


def test_bench_flip_horizontal(benchmark):
    test = jnp.array(np.random.randint(0, 255, (32, 3, 224, 224), dtype=np.uint8))
    flip = FlipHorizontal()
    flip(test, key=jax.random.PRNGKey(0))
    benchmark(flip, test, key=jax.random.PRNGKey(0))


def test_bench_shear(benchmark):
    test = jnp.array(np.random.randint(0, 255, (32, 3, 224, 224), dtype=np.uint8))
    shear = Shear('x', (0, 1))
    shear(test, key=jax.random.PRNGKey(0))
    benchmark(shear, test, key=jax.random.PRNGKey(0))


def test_bench_rotate(benchmark):
    test = jnp.array(np.random.randint(0, 255, (32, 3, 224, 224), dtype=np.uint8))
    rotate = Rotate((0, 360))
    rotate(test, key=jax.random.PRNGKey(0))
    benchmark(rotate, test, key=jax.random.PRNGKey(0))


def test_bench_invert(benchmark):
    test = jnp.array(np.random.randint(0, 255, (32, 3, 224, 224), dtype=np.uint8))
    invert = Invert()
    invert(test, key=jax.random.PRNGKey(0))
    benchmark(invert, test, key=jax.random.PRNGKey(0))
