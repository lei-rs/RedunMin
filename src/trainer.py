from dataclasses import dataclass
from functools import cached_property
from typing import Any, Dict, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from haliax import NamedArray, named_jit
from jax.nn import one_hot
from lightning import LightningModule
from optax import GradientTransformation


@dataclass(init=True, repr=True, frozen=True)
class TrainConfig:
    optim: GradientTransformation
    loss_fn: Callable
    dist: bool = False


class TrainModule(LightningModule):
    def __init__(self, model: eqx.Module, cfg: TrainConfig, *, key):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.key = key

        self.automatic_optimization = False
        self.global_step_ = 0
        self.opt_state = None

    @property
    def global_step(self) -> int:
        return self.global_step_

    @staticmethod
    def _forward(diff_model, static_model, loss_fn, target, x, *, key):
        model = eqx.combine(diff_model, static_model)
        y_pred = model(x, key=key)
        return loss_fn(y_pred, target)

    @staticmethod
    def _all_reduce(grads: NamedArray):
        return jax.lax.pmean(grads.array, axis_name='batch')

    @cached_property
    def _update_fn(self):
        def update(params, grads):
            updates, opt_state = self.cfg.optim.update(
                grads,
                self.opt_state,
                params=params
            )
            model = eqx.apply_updates(self.model, updates)
            return model, opt_state
        return named_jit(update, donate_args=(True, True))

    @cached_property
    def _train_step_fn(self):
        def train_step(*batch, **batch_kwargs):
            diff_model, static_model = self.partition_params(self.model)
            loss, grads = eqx.filter_value_and_grad(self._forward)(
                diff_model,
                static_model,
                self.cfg.loss_fn,
                *batch,
                **batch_kwargs,
            )
            if self.dist:
                grads = self._all_reduce(grads)
                model, opt_state = eqx.filter_pmap(
                    self._update_fn,
                    axis_name='batch',
                )(diff_model, grads)
            else:
                model, opt_state = self._update_fn(diff_model, grads)
            return loss, model, opt_state

        return named_jit(
            train_step,
            axis_resources=self.parameter_axis_mapping,
            out_axis_resources=self.parameter_axis_mapping,
            donate_args=(True, True),
        )

    @cached_property
    def _val_step_fn(self):
        def val_step(model, batch) -> Dict[str, Any]:
            target, x = batch
            y_pred = model(x)
            ret = {
                'val_loss': self.cfg.loss_fn(y_pred, target),
            }
            return ret
        return named_jit(val_step, donate_args=(False, True))

    def training_step(self, batch, batch_idx):
        self.model = self.model.set_inference(False)
        self.key, key = jax.random.split(self.key)
        loss, self.model, self.opt_state = self._train_step_fn(
            self.model,
            self.opt_state,
            *batch,
            key=key,
        )
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss.item()

    def validation_step(self, batch, batch_idx):
        self.model = self.model.set_inference(True)
        metrics = self._val_step_fn(self.model, batch)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return metrics['val_loss'].item()

    def configure_optimizers(self):
        self.opt_state = self.cfg.optim.init(self.model)
