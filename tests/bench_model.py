import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrand
import optax
from haliax import NamedArray
from jax.nn import one_hot
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from jax_smi import initialise_tracking
from transformers import ViTConfig as HFViTConfig
from transformers.models.vit.modeling_flax_vit import FlaxViTEncoder

from src.model.lq import LQViT, LQViTConfig

Batch = hax.Axis(name='batch', size=32)
Pos = hax.Axis(name='position', size=196 * 8)
Embed = hax.Axis(name='embed', size=768)


initialise_tracking()


def cross_entropy(logits: NamedArray, labels: NamedArray, num_classes: int) -> NamedArray:
    labels = labels.astype(jnp.int32)
    return optax.softmax_cross_entropy(
        logits.array,
        one_hot(labels.array, num_classes)
    ).mean()


@hax.named_jit
def make_model(mapping):
    key = jrand.PRNGKey(0)
    cfg = LQViTConfig()
    model = LQViT.init(cfg, key=key).astype(jnp.bfloat16)
    return hax.shard_with_axis_mapping(model, mapping)


@hax.named_jit
def forward(model, x, key, mapping):
    with hax.axis_mapping(mapping):
        return model(x, key=key)


@hax.named_jit
@eqx.filter_value_and_grad
def grad_forward(model, x, key, mapping):
    y_pred = forward(model, x, key, mapping)
    return cross_entropy(
        y_pred,
        hax.named(jnp.zeros(Batch.size, jnp.int32), 'batch'),
        400
    )


def test_bench_baseline_encoder_ddp(benchmark):
    key = jrand.PRNGKey(0)

    mesh = Mesh(jax.devices('tpu'), 'data')
    x_sharding = NamedSharding(mesh, PartitionSpec('data', None, None))

    x = jnp.ones((Batch.size, Pos.size, Embed.size), dtype=jnp.bfloat16)
    model = FlaxViTEncoder(HFViTConfig(), jnp.bfloat16)

    x = jax.device_put(x, x_sharding)
    params = model.init(key, x)

    fwd = jax.jit(model.apply)
    forward = lambda: jax.block_until_ready(fwd(params, x))
    forward()
    benchmark(forward)


def test_bench_encoder_ddp(benchmark):
    mesh = Mesh(jax.devices('tpu'), ('data', 'model'))
    compute_axis_mapping = {'batch': 'data'}

    with mesh:
        key = jrand.PRNGKey(1)
        x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping)
        model = make_model({}).vit_encoder
        fwd = lambda: jax.block_until_ready(forward(model, x, key=key, mapping=compute_axis_mapping))
        fwd()
        benchmark(fwd)


def test_bench_encoder_fsdp(benchmark):
    mesh = Mesh(jax.devices('tpu'), 'data')
    compute_axis_mapping = {'batch': 'data'}
    param_axis_mapping = {'embed': 'data'}

    with mesh:
        key = jrand.PRNGKey(1)
        x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping)
        model = make_model(param_axis_mapping).vit_encoder
        fwd = lambda: jax.block_until_ready(forward(model, x, key=key, mapping=compute_axis_mapping))
        fwd()
        benchmark(fwd)


def test_bench_model_fsdp(benchmark):
    mesh = Mesh(jax.devices('tpu'), 'data')
    compute_axis_mapping = {'batch': 'data'}
    param_axis_mapping = {'embed': 'data'}

    with mesh:
        key = jrand.PRNGKey(1)
        x = hax.named(
            jnp.ones((32, 32, 3, 224, 224), dtype=jnp.bfloat16),
            ('batch', 'temporal', 'channels', 'height', 'width')
        )
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping)
        model = make_model(param_axis_mapping)
        fwd = lambda: jax.block_until_ready(forward(model, x, key=key, mapping=compute_axis_mapping))
        fwd()
        benchmark(fwd)


def test_bench_model_fsdp_grad(benchmark):
    mesh = Mesh(jax.devices('tpu'), 'data')
    compute_axis_mapping = {'batch': 'data'}
    param_axis_mapping = {'embed': 'data'}

    with mesh:
        key = jrand.PRNGKey(1)
        x = hax.named(
            jnp.ones((32, 32, 3, 224, 224), dtype=jnp.bfloat16),
            ('batch', 'temporal', 'channels', 'height', 'width')
        )
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping)
        model = make_model(param_axis_mapping)
        fwd = lambda: jax.block_until_ready(grad_forward(model, x, key=key, mapping=compute_axis_mapping))
        fwd()
        benchmark(fwd)
