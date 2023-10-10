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


def make_model() -> LQViT:
    key = jrand.PRNGKey(0)
    cfg = LQViTConfig(
        32,
        8,
        400
    )
    return LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        'gs://redunmin-us/vit/vit-base-16-224.safetensors',
        cfg,
        key=key,
        dtype=jnp.bfloat16
    )


def test_bench_baseline_encoder(benchmark):
    key = jrand.PRNGKey(0)

    mesh = mesh_utils.create_device_mesh((4, 2), jax.devices('tpu'), contiguous_submeshes=True)
    mesh = Mesh(mesh, ('data', 'model'))
    model_sharding = NamedSharding(mesh, PartitionSpec('model', None))
    x_sharding = NamedSharding(mesh, PartitionSpec('data', None, None))

    x = jnp.ones((8, 196 * 8, 768), dtype=jnp.bfloat16)
    model = FlaxViTEncoder(HFViTConfig(), jnp.bfloat16)

    x = jax.device_put(x, x_sharding)

    params = model.init(key, x)
    forward = jax.jit(model.apply)
    forward(params, x)
    benchmark(forward, params, x)


def test_bench_encoder_ddp(benchmark):
    key = jrand.PRNGKey(0)

    mesh = mesh_utils.create_device_mesh((4, 2), jax.devices('tpu'), contiguous_submeshes=True)
    mesh = Mesh(mesh, ('data', 'model'))
    model_sharding = NamedSharding(mesh, PartitionSpec('model', None))

    cfg = LQViTConfig(32, 8, 400)
    model = LQViT.init(cfg, key=key).astype(jnp.bfloat16).vit_encoder
    x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)

    x = hax.shard_with_axis_mapping(x, {'batch': 'data'}, mesh)
    model = jax.device_put(model, model_sharding)

    forward = jax.jit(model.__call__)
    forward(x, key=key)
    benchmark(forward, x, key=key)


def test_bench_encoder_fsdp(benchmark):
    key = jrand.PRNGKey(0)

    mesh = Mesh(jax.devices('tpu'), 'data')

    cfg = LQViTConfig(32, 8, 400)
    model = LQViT.init(cfg, key=key).astype(jnp.bfloat16).vit_encoder
    x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)

    x = hax.shard_with_axis_mapping(x, {'batch': 'data'}, mesh)
    model = hax.shard_with_axis_mapping(model, {'embed': 'data'}, mesh)

    forward = jax.jit(model.__call__)
    forward(x, key=key)
    benchmark(forward, x, key=key)
