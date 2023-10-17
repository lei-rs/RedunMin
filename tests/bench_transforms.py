import jax.random

from src.data.transforms import *

X = jax.random.randint(
    jax.random.PRNGKey(0),
    (32, 32, 3, 224, 224),
    0,
    255,
    jnp.uint8
)


def test_bench_trivial_augment(benchmark):
    t = TrivialAugment()
    key = jax.random.split(jax.random.PRNGKey(0), 32)
    t.warmup(X[0])
    v_t = eqx.filter_vmap(
        lambda v, k: t(v, key=k),
        in_axes=(0, 0),
        out_axes=0
    )
    benchmark(lambda: jax.block_until_ready(v_t(jnp.copy(X), key)))


def test_bench_flip_horizontal(benchmark):
    flip = FlipHorizontal()
    flip(jnp.copy(X))
    benchmark(lambda: jax.block_until_ready(flip(jnp.copy(X))))


def test_bench_shear(benchmark):
    shear = jax.jit(Shear())
    shear(jnp.copy(X), strength=1)
    benchmark(lambda: jax.block_until_ready(shear(jnp.copy(X), strength=1)))


def test_bench_rotate(benchmark):
    rotate = jax.jit(Rotate())
    rotate(jnp.copy(X), strength=1)
    benchmark(lambda: jax.block_until_ready(rotate(X, strength=1)))


def test_bench_invert(benchmark):
    invert = jax.jit(Invert())
    invert(jnp.copy(X))
    benchmark(lambda: jax.block_until_ready(invert(jnp.copy(X))))


def test_bench_normalize(benchmark):
    n = Normalize()
    n.warmup(X)
    n = jax.jit(n.__call__)
    benchmark(lambda: jax.block_until_ready(n(jnp.copy(X))))
