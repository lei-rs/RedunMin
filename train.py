from functools import partial
from typing import List, Tuple

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jax_rand
import optax
from lightning import Trainer
from haliax import NamedArray
from jax import Array
from jax.numpy import bfloat16
from jax.sharding import Mesh
from jax.nn import one_hot

from src.data.loader import DLConfig, SSV2
from src.model.lq import LQViTConfig, LQViT
from src.trainer import TrainConfig, TrainModule


Batch = hax.Axis(name='batch', size=16)

compute_axis_mapping = {"batch": "data"}
global_mesh = Mesh(jax.devices('tpu'), 'data')
local_mesh = Mesh(jax.local_devices(backend='tpu'), 'data')


def collate_and_put(x: Tuple[List[Array], List[Array]]) -> Tuple[NamedArray, NamedArray]:
    cls, x = x
    cls = hax.named(jnp.asarray(cls), 'cls')
    x = hax.named(jnp.asarray(x), ('batch', 'temporal', 'channels', 'height', 'width'))
    cls = hax.shard_with_axis_mapping(cls, compute_axis_mapping, local_mesh)
    x = hax.shard_with_axis_mapping(x, compute_axis_mapping, local_mesh)
    return cls, x


def cross_entropy(logits: NamedArray, labels: NamedArray, num_classes: int):
    labels = labels.astype(jnp.int32)
    return jnp.mean(
        optax.softmax_cross_entropy(
            logits.array,
            one_hot(labels.array, num_classes)
        )
    )


if __name__ == '__main__':
    key = jax_rand.PRNGKey(0)
    key, key_dl, key_model, key_trainer = jax_rand.split(key, 4)

    dl_cfg = DLConfig(
        data_loc='gs://redunmin-us',
        collate_fn=collate_and_put,
        batch_size=Batch.size,
        shuffle=True,
        n_frames=32,
    )
    dl = SSV2(dl_cfg, key=key_dl)

    model_cfg = LQViTConfig()
    model = LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        f'{dl_cfg.data_loc}/vit/vit-base-16-224.safetensors',
        model_cfg,
        key=key_model,
        dtype=bfloat16,
    )

    train_cfg = TrainConfig(
        loss_fn=partial(cross_entropy, num_classes=model_cfg.n_classes),
        optim=optax.adamw(
            1e-4,
            weight_decay=1e-2,
        ),
    )

    tm = TrainModule(train_cfg, model, key=key_trainer)
    trainer = Trainer(max_epochs=100)
    trainer.fit(tm, datamodule=dl)
