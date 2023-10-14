from unittest import TestCase

import haliax as hax
import jax.numpy as jnp
import jax.random as jrand
from google.cloud import storage
from safetensors.numpy import load

from src.model.lq import LQViT, LQViTConfig

Batch = hax.Axis(name='batch', size=1)
Temporal = hax.Axis(name='temporal', size=32)
Spatial = hax.Axis(name='spatial', size=(224 ** 2 // 16 ** 2))
Pos = hax.Axis(name='position', size=196 * 8)
Embed = hax.Axis(name='embed', size=768)


def get_sd():
    client = storage.Client()
    bucket = client.get_bucket('redunmin-us')
    blob = bucket.blob('vit/vit-base-16-224.safetensors')
    return load(blob.download_as_bytes())


class TestLQViT(TestCase):
    def test_load(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig()
        LQViT.from_pretrained(
            'google/vit-base-patch16-224',
            'gs://redunmin-us/vit/vit-base-16-224.safetensors',
            cfg,
            key=key,
        )

    def test_forward(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig()
        model = LQViT.init(cfg, key=key).astype(jnp.bfloat16)
        x = hax.ones((Batch, Temporal, cfg.Channels, cfg.Height, cfg.Width), dtype=jnp.bfloat16)
        model(x, key=key)

    def test_pretrained_forward(self):
        key = jrand.PRNGKey(0)
        cfg = LQViTConfig()
        model = LQViT.from_pretrained(
            'google/vit-base-patch16-224',
            'gs://redunmin-us/vit/vit-base-16-224.safetensors',
            cfg,
            key=key,
        )
        model = model.astype(jnp.bfloat16)
        x = hax.ones((Batch, Temporal, cfg.Channels, cfg.Height, cfg.Width), dtype=jnp.bfloat16)
        model(x, key=key)