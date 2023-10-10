import jax
import numpy as np
import haliax as hax
import jax.numpy as jnp
import jax.random as jrand
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from src.model.lq import LQViT, LQViTConfig


Batch = hax.Axis(name='batch', size=128)
Temporal = hax.Axis(name='time_in', size=32)
Spatial = hax.Axis(name='spatial', size=(224 ** 2 // 16 ** 2))
Embed = hax.Axis(name='embed', size=768)

compute_axis_mapping = {"batch": 'data'}
param_axis_mapping = {"embed": 'model'}


def test_bench_model(benchmark):
    key = jrand.PRNGKey(0)
    cfg = LQViTConfig(
        32,
        8,
        400
    )
    mesh = Mesh(np.array(jax.devices('tpu')).reshape((4, 2)), ('data', 'model'))
    model = LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        'gs://redunmin-us/vit/vit-base-16-224.safetensors',
        cfg,
        key=key,
    )
    model = hax.shard_with_axis_mapping(model, param_axis_mapping, mesh)
    x = hax.ones((Batch, Temporal, Spatial, Embed), dtype=jnp.float32)
    x = hax.shard_with_axis_mapping(x, compute_axis_mapping, mesh)
    model(x, key=key)
    benchmark(model, x, key=key)
