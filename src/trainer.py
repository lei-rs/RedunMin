from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Callable

import equinox as eqx
import haliax as hax
import jax
import jax.tree_util as jtu
from haliax import named_jit
from jax.sharding import Mesh
from lightning import LightningModule
from optax import GradientTransformation

from src.model.lq import LQViT


DEBUG = False


@dataclass(init=True, repr=True, frozen=True)
class TrainConfig:
    batch_size: int
    optim: GradientTransformation
    loss_fn: Callable
    global_mesh: Mesh
    compute_axis_mapping: Dict[str, str]
    param_axis_mapping: Dict[str, str]
    dist: bool = False


class TrainModule(LightningModule):
    def __init__(self, model: LQViT, cfg: TrainConfig, *, key):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.key = key

        self.opt_state = None

        self.automatic_optimization = False
        self.global_step_ = 0

    @property
    def global_step(self) -> int:
        return self.global_step_

    @cached_property
    def filter_spec(self):
        fs = jtu.tree_map(lambda _: True, self.model)
        return eqx.tree_at(
            lambda x: x.vit_encoder,
            fs,
            jtu.tree_map(lambda _: False, self.model.vit_encoder)
        )

    @cached_property
    def _forward_fn(self):
        @eqx.filter_value_and_grad
        def _forward(diff_model, static_model, loss_fn, target, x, *, key):
            with hax.axis_mapping(self.cfg.compute_axis_mapping):
                model = eqx.combine(diff_model, static_model)
                y_pred = model(x, key=key)
                return loss_fn(y_pred, target)
        return _forward

    @cached_property
    def _backward_fn(self):
        def _backward(model, opt_state, grads, params):
            updates, opt_state = self.cfg.optim.update(grads, opt_state, params=params)
            model = eqx.apply_updates(model, updates)
            return model, opt_state
        return _backward

    @cached_property
    def _train_step_fn(self):
        def train_step(model, opt_state, *batch, **batch_kwargs):
            diff_model, static_model = eqx.partition(model, self.filter_spec)
            loss, grads = self._forward_fn(
                diff_model,
                static_model,
                self.cfg.loss_fn,
                *batch,
                **batch_kwargs,
            )
            model, opt_state = self._backward_fn(
                model,
                opt_state,
                grads,
                diff_model
            )
            return loss, model, opt_state
        return named_jit(train_step, donate_args=(True, True))

    @cached_property
    def _val_step_fn(self):
        def val_step(target, x) -> Dict[str, Any]:
            with hax.axis_mapping(self.cfg.compute_axis_mapping):
                y_pred = self.model(x, key=self.key)
            ret = {
                'val_loss': self.cfg.loss_fn(y_pred, target),
            }
            return ret
        return named_jit(val_step)

    def on_train_epoch_start(self):
        self.model = self.model.set_inference(False)

    def training_step(self, batch, batch_idx):
        self.key, key = jax.random.split(self.key)

        with self.cfg.global_mesh:
            loss, self.model, self.opt_state = self._train_step_fn(
                self.model,
                self.opt_state,
                *batch,
                key=key,
            )

        loss = float(jax.block_until_ready(loss))
        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=False,
            batch_size=self.cfg.batch_size
        )

        self.global_step_ += 1

    def on_validation_epoch_start(self):
        self.model = self.model.set_inference(True)

    '''def validation_step(self, batch, batch_idx):
        self.model = self.model.set_inference(True)
        metrics = self._val_step_fn(*batch)
        metrics = {k: float(jax.block_until_ready(v)) for k, v in metrics.items()}
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            batch_size=self.cfg.batch_size
        )
        return metrics['val_loss']'''

    def configure_optimizers(self):
        @named_jit(out_axis_resources=self.cfg.param_axis_mapping)
        def _init_model_optim():
            trainable, _ = eqx.partition(self.model, self.filter_spec)
            trainable = eqx.filter(trainable, eqx.is_inexact_array)
            return self.model, self.cfg.optim.init(trainable)
        with self.cfg.global_mesh:
            self.model, self.opt_state = _init_model_optim()
        if DEBUG:
            print(jax.debug.visualize_array_sharding(self.model.pos_embed.array[0]))
