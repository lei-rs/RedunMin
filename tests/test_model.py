from unittest import TestCase

import jax.random as jrand

from src.model.lq import LQViT, LQViTConfig


class TestLQViT(TestCase):
    def test_load_lqvit(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig(
            32,
            8,
            400
        )
        model = LQViT.from_pretrained(
            'google/vit-base-patch16-224',
            'data/vit_vit-base-16-224.safetensors',
            cfg,
            key=key,
        )
        