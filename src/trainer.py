from dataclasses import dataclass
from functools import partial
from typing import Tuple, Callable, Dict, List, Any

import equinox as eqx
import haliax as hax
import jax
import jax.numpy as jnp
import pypeln as pl
from jax.numpy import ndarray
from jax.sharding import Mesh
from tqdm import tqdm

from .data.loader import DataLoader
from .model.lq import LQViT

compute_axis_mapping = {"batch": "data"}
param_axis_mapping = {"embed": "data"}


def _calc_loss(model, x, y, loss_fn, *, key):
    pred_y = model(x, key=key)
    return loss_fn(pred_y, y)


def _calc_loss_partial(diff_model, static_model, x, y, loss_fn, *, key):
    model = eqx.combine(diff_model, static_model)
    pred_y = model(x, key=key)
    return loss_fn(pred_y, y)


@hax.named_jit
def _init_optimizer(optimizer: Any, model: eqx.Module):
    model = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(model)
    return hax.shard_with_axis_mapping(opt_state, param_axis_mapping)


@dataclass(init=True, repr=True, frozen=True)
class TrainerConfig:
    max_epochs: int
    loss_fn: Callable
    optim: Any
    metrics: Dict[str, Callable] = None
    distributed: bool = False

    def build_mesh(self):
        if self.distributed:
            jax.distributed.initialize()
        return Mesh(
            jax.devices('tpu'),
            'data'
        )

    @staticmethod
    def filter_spec(epoch):
        return None


class Trainer:
    def __init__(self, config: TrainerConfig, model: LQViT, data: DataLoader, *, key=None):
        self.config = config
        self.mesh = config.build_mesh()
        self.model = hax.shard_with_axis_mapping(
            model,
            param_axis_mapping,
            self.mesh
        )
        self.optim = config.optim
        self.dl = data
        self.opt_state = None

        self.epoch = 0
        self.key = key

    @staticmethod
    @hax.named_jit
    def _data_put(x: Tuple[List[ndarray], ...], mesh: Mesh):
        cls, x = x
        cls = hax.named(jnp.asarray(cls), 'cls')
        x = hax.named(jnp.asarray(x), ('batch', 'temporal', 'channels', 'height', 'width'))
        cls = hax.shard_with_axis_mapping(cls, compute_axis_mapping, mesh)
        x = hax.shard_with_axis_mapping(x, compute_axis_mapping, mesh)
        return cls, x

    def get_loss_fn(self, x, y, *, key):
        fs = self.config.filter_spec(self.epoch)
        if fs is not None:
            diff_model, static_model = eqx.partition(self.model, fs)
            loss_fn = partial(
                eqx.filter_value_and_grad(_calc_loss_partial),
                diff_model,
                static_model,
                x,
                y,
                self.config.loss_fn,
                key=key
            )
        else:
            loss_fn = partial(
                eqx.filter_value_and_grad(_calc_loss),
                self.model,
                x,
                y,
                self.config.loss_fn,
                key=key
            )
        return loss_fn

    @staticmethod
    @hax.named_jit
    def train_step(model, optim, opt_state, calc_grad_loss):
        with hax.axis_mapping(compute_axis_mapping):
            loss, grads = calc_grad_loss()
        updates, opt_state = optim.update(grads, opt_state, params=model)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    @staticmethod
    @hax.named_jit
    def val_step(model, x, y, metrics):
        with hax.axis_mapping(compute_axis_mapping):
            return {k: m(model(x), y) for k, m in metrics.items()}

    def train_loop(self):
        tl = (
            self.dl.train_loader()
                | pl.sync.map(partial(self._data_put, mesh=self.mesh))
            )
        for cls, x in tqdm(tl, total=self.dl.len('train')):
            print(x.dtype)
            self.key, key_m = jax.random.split(self.key)
            loss, self.model, self.opt_state = self.train_step(
                self.model,
                self.optim,
                self.opt_state,
                self.get_loss_fn(x, cls, key=key_m)
            )
            tqdm.write(f'Loss: {loss}')

    def val_loop(self):
        vl = (
            self.dl.val_loader()
                | pl.sync.map(partial(self._data_put, mesh=self.mesh))
            )
        for cls, x in vl:
            loss = self.val_step(
                self.model,
                self.get_loss_fn(x, cls)
            )
            msg = 'Val: '
            for k, v in loss.items():
                msg += f'{k}: {v} '
            tqdm.write(msg)

    def train(self):
        with self.mesh:
            self.opt_state = _init_optimizer(self.optim, self.model)
            while self.epoch < self.config.max_epochs:
                self.train_loop()
                self.val_loop()
                self.epoch += 1
                self.dl.step()
