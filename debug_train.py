from functools import partial
from typing import Tuple

import haliax as hax
import jax
import jax.numpy as jnp
import jax.random as jax_rand
import optax
from haliax import NamedArray
from jax.nn import one_hot
from jax.numpy import bfloat16
from jax.sharding import Mesh
from jax_smi import initialise_tracking
from lightning import Trainer

import src.trainer as trainer
from src.data.loader import DLConfig, SSV2
from src.model.lq import LQViTConfig, LQViT
from src.trainer import TrainConfig, TrainModule

Batch = hax.Axis(name='batch', size=64)
compute_axis_mapping = {'batch': 'data'}
param_axis_mapping = {'embed': 'data'}
global_mesh = Mesh(jax.devices('tpu'), 'data')


def put(x: Tuple[NamedArray, NamedArray]) -> tuple[NamedArray, NamedArray]:
    x1, x2 = x
    '''with global_mesh:
        return (
            hax.shard_with_axis_mapping(x1, compute_axis_mapping),
            hax.shard_with_axis_mapping(x2, compute_axis_mapping),
        )'''
    return jax.device_put(x1), jax.device_put(x2)


def cross_entropy(model: LQViT, targets: NamedArray, x: NamedArray, *, key, num_classes: int) -> NamedArray:
    raw_logits = model(x, key=key)
    targets = targets.astype(jnp.int32)
    return optax.softmax_cross_entropy(
        raw_logits.array,
        one_hot(targets.array, num_classes)
    ).mean()


def init_model(*args, **kwargs):
    model = LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        f'{dl_cfg.data_loc}/vit/vit-base-16-224.safetensors',
        *args,
        **kwargs,
    )
    return hax.shard_with_axis_mapping(model, param_axis_mapping)


if __name__ == '__main__':
    trainer.DEBUG = True
    key = jax_rand.PRNGKey(0)
    key, key_dl, key_model, key_trainer = jax_rand.split(key, 4)

    dl_cfg = DLConfig(
        data_loc='gs://redunmin-us',
        put_fn=put,
        batch_size=Batch.size,
        shuffle=True,
        n_frames=32,
    )
    dl = SSV2(dl_cfg, key=key_dl)

    model_cfg = LQViTConfig()
    model_init = partial(init_model, model_cfg, key=key_model, dtype=bfloat16)

    train_cfg = TrainConfig(
        batch_size=Batch.size,
        loss_fn=partial(cross_entropy, num_classes=model_cfg.n_classes),
        global_mesh=global_mesh,
        compute_axis_mapping=compute_axis_mapping,
        param_axis_mapping=param_axis_mapping,
        optim=optax.adamw(
            1e-4,
            weight_decay=1e-2,
        ),
    )
    tm = TrainModule(model_init, train_cfg, key=key_trainer)
    trainer = Trainer(max_epochs=100)
    initialise_tracking()
    jax.profiler.start_trace('.tmp/', create_perfetto_trace=True)
    trainer.fit(tm, datamodule=dl)
    jax.profiler.stop_trace()
