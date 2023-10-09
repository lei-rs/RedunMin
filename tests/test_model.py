from unittest import TestCase

import jax.numpy as jnp
import jax.random as jrand
import haliax as hax

from src.model.lq import LQViT, LQViTConfig
from src.model.vit_encoder import ViTConfig

Batch = hax.Axis(name='batch', size=1)
Temporal = hax.Axis(name='time_in', size=32)
Spatial = hax.Axis(name='spatial', size=(224 ** 2 // 16 ** 2))
Embed = hax.Axis(name='embed', size=768)


class TestLQViT(TestCase):
    def test_lqvit_load(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig(
            32,
            8,
            400
        )
        LQViT.from_pretrained(
            'google/vit-base-patch16-224',
            'data/vit_vit-base-16-224.safetensors',
            cfg,
            key=key,
        )

    def test_lqvit_forward(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig(
            32,
            8,
            400
        )
        cfg.add_vit_cfg(ViTConfig())
        slf = LQViT.init(cfg, key=key)
        x = hax.ones((Batch, Temporal, Spatial, Embed), dtype=jnp.float32)
        slf(x, key=key)
