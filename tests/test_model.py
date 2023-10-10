from unittest import TestCase

import haliax as hax
import jax.numpy as jnp
import jax.random as jrand

from src.model.lq import LQViT, LQViTConfig

Batch = hax.Axis(name='batch', size=1)
Temporal = hax.Axis(name='time_in', size=32)
Spatial = hax.Axis(name='spatial', size=(224 ** 2 // 16 ** 2))
Pos = hax.Axis(name='position', size=196 * 8)
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
            'gs://redunmin-us/vit/vit-base-16-224.safetensors',
            cfg,
            key=key,
        )

    def test_lqvit_forward(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig(32, 8, 400)
        model = LQViT.init(cfg, key=key).astype(jnp.bfloat16)
        x = hax.ones((Batch, Temporal, Spatial, Embed), dtype=jnp.bfloat16)
        model(x, key=key)

    def test_encoder_fowrad(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig(32, 8, 400)
        model = LQViT.init(cfg, key=key).astype(jnp.bfloat16).vit_encoder
        x = hax.ones((Batch, Pos, Embed), dtype=jnp.bfloat16)
        model(x, key=key)