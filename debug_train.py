from functools import partial

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
import src.data.transforms as T
from src.data.loader import DLConfig, SSV2
from src.model.lq import LQViTConfig, LQViT
from src.trainer import TrainConfig, TrainModule

Batch = hax.Axis(name='batch', size=72)
compute_axis_mapping = {'batch': 'data'}
param_axis_mapping = {'embed': 'data'}
mesh = Mesh(jax.devices('tpu'), 'data')


TRAIN_AUG = T.Sequential([
    T.TrivialAugment(),
    T.Normalize().astype(jnp.bfloat16),
])


@hax.named_jit
def collate_put(cls, vid):
    cls = hax.named(jnp.stack(cls), 'batch')
    vid = hax.named(jnp.stack(vid), ('batch', 'temporal', 'channels', 'height', 'width'))
    with mesh:
        cls = hax.shard_with_axis_mapping(cls, compute_axis_mapping)
        vid = hax.shard_with_axis_mapping(vid, compute_axis_mapping)
    return cls, vid


def cross_entropy(model: LQViT, targets: NamedArray, x: NamedArray, *, key, num_classes: int) -> NamedArray:
    raw_logits = model(x, key=key)
    targets = targets.astype(jnp.int32)
    return optax.softmax_cross_entropy(
        raw_logits.array,
        one_hot(targets.array, num_classes)
    ).mean()


def init_model(*args, dtype=bfloat16, **kwargs):
    return LQViT.init(*args, **kwargs).astype(dtype)
    '''return LQViT.from_pretrained(
        'google/vit-base-patch16-224',
        f'{dl_cfg.data_loc}/vit/vit-base-16-224.safetensors',
        *args,
        **kwargs,
    )'''


if __name__ == '__main__':
    trainer.DEBUG = True
    key = jax_rand.PRNGKey(0)
    key, key_dl, key_model, key_trainer = jax_rand.split(key, 4)

    dl_cfg = DLConfig(
        data_loc='gs://redunmin-us',
        collate_put=collate_put,
        transforms={
            'train': TRAIN_AUG
        },
        batch_size=Batch.size,
        shuffle=True,
        n_frames=32,
    )
    dl = SSV2(dl_cfg, key=key_dl)

    model_cfg = LQViTConfig(
        t_dims=(32, 4),
        n_classes=174
    )
    init_model = partial(init_model, model_cfg, key=key_model, dtype=bfloat16)

    train_cfg = TrainConfig(
        batch_size=Batch.size,
        loss_fn=partial(cross_entropy, num_classes=model_cfg.n_classes),
        mesh=mesh,
        compute_axis_mapping=compute_axis_mapping,
        param_axis_mapping=param_axis_mapping,
        optim=optax.adamw(
            1e-4,
            weight_decay=1e-2,
        ),
    )

    initialise_tracking()
    tm = TrainModule(init_model, train_cfg, key=key_trainer)
    trainer = Trainer(max_epochs=100)
    trainer.fit(tm, datamodule=dl)
