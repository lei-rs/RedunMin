import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jrand
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec, NamedSharding
from transformers import ViTConfig as HFViTConfig
from transformers.models.vit.modeling_flax_vit import FlaxViTEncoder

from src.model.lq import LQViT, LQViTConfig

Batch = hax.Axis(name='batch', size=64)
Pos = hax.Axis(name='position', size=196 * 8)
Embed = hax.Axis(name='embed', size=768)


@hax.named_jit
def make_model(mapping):
    key = jrand.PRNGKey(0)
    cfg = LQViTConfig()
    model = LQViT.init(cfg, key=key).astype(jnp.bfloat16).vit_encoder
    return hax.shard_with_axis_mapping(model, mapping)


@hax.named_jit
def forward(model, x, key, mapping):
    with hax.axis_mapping(mapping):
        return model(x, key=key)


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
    mesh = mesh_utils.create_device_mesh((4, 2), jax.devices('tpu'), contiguous_submeshes=True)
    mesh = Mesh(mesh, ('data', 'model'))
    compute_axis_mapping = {'batch': 'data'}
    param_axis_mapping = {'embed': 'model'}

    with mesh:
        key = jrand.PRNGKey(1)
        x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping)
        model = make_model(param_axis_mapping)
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
        model = make_model(param_axis_mapping)
        fwd = lambda: jax.block_until_ready(forward(model, x, key=key, mapping=compute_axis_mapping))
        fwd()
        benchmark(fwd)
