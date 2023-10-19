import os

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jax_rand
from haliax import NamedArray
from jax.nn import one_hot
from jax.sharding import Mesh
from jax_smi import initialise_tracking
from optax import softmax_cross_entropy
from tqdm import tqdm

import src.data.transforms as T
from src.data.loader import DLConfig, SSV2
from src.model.lq import LQViTConfig, LQViT

compute_axis_mapping = {'batch': 'data'}
param_axis_mapping = {'embed': 'data'}
mesh = Mesh(jax.local_devices(backend='tpu'), 'data')


TRAIN_AUG = T.Sequential([
    T.TrivialAugment(),
    T.Normalize().astype(jnp.bfloat16),
])

@jax.jit
def collate_put(cls, vid):
    cls = hax.named(jnp.stack(cls), 'batch')
    vid = hax.named(jnp.stack(vid), ('batch', 'temporal', 'channels', 'height', 'width'))
    with mesh:
        cls = hax.shard_with_axis_mapping(cls, compute_axis_mapping)
        vid = hax.shard_with_axis_mapping(vid, compute_axis_mapping)
    return cls, vid


@hax.named_jit(donate_args=(False, True, True))
def cross_entropy(model: LQViT, targets: NamedArray, x: NamedArray, *, key, num_classes: 174) -> NamedArray:
    with hax.axis_mapping(compute_axis_mapping):
        raw_logits = model(x, key=key)
    targets = targets.astype(jnp.int32)
    return softmax_cross_entropy(
        raw_logits.array,
        one_hot(targets.array, num_classes)
    ).mean()


key = jax_rand.PRNGKey(0)

m_cfg = LQViTConfig(n_classes=174)
model = hax.shard_with_axis_mapping(
    LQViT.init(m_cfg, key=key).astype(jnp.bfloat16),
    param_axis_mapping,
    mesh=mesh,
)

config = DLConfig(
    data_loc='gs://redunmin',
    collate_put=collate_put,
    transforms={
        'train': TRAIN_AUG
    },
    batch_size=32,
)
loader = SSV2(config, key=key)
loader.setup('train')
initialise_tracking()
log_path = f'{os.environ["HOME"]}/tmp/tensorboard'
os.makedirs(log_path, exist_ok=True)
with jax.profiler.trace(log_path):
    for i, x in enumerate(tqdm(loader.train_dataloader())):
        with mesh:
            jax.block_until_ready(cross_entropy(model, x[0], x[1], key=key, num_classes=174))
            #jax.block_until_ready(x)
