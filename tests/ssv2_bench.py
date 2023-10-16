from typing import Tuple

import haliax as hax
import jax
import jax.random as jax_rand
from haliax import NamedArray
from jax.sharding import Mesh
from tqdm import tqdm

from src.data.loader import DLConfig, SSV2

compute_axis_mapping = {'batch': 'data'}
local_mesh = Mesh(jax.local_devices(backend='tpu'), 'data')


def put(x: Tuple[NamedArray, NamedArray]) -> Tuple[NamedArray, NamedArray]:
    with local_mesh:
        return hax.shard_with_axis_mapping(x, compute_axis_mapping)


config = DLConfig(
    data_loc='gs://redunmin',
    put_fn=lambda x: x,
    batch_size=32,
)
loader = SSV2(config, key=jax_rand.PRNGKey(0))
loader.setup('train')
for i, x in enumerate(tqdm(loader.train_dataloader())):
    jax.block_until_ready(x)
