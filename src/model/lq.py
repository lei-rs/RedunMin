from typing import Optional, Dict

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax_dataclasses as jdc
from google.cloud import storage
from haliax import NamedArray, Axis
from haliax.jax_utils import named_call, maybe_rng_split
from safetensors.numpy import load
from transformers import ViTConfig as HFViTConfig

from .levanter.safetensor import STSerde
from .vit import ViTConfig, ViTEncoder


@jdc.pytree_dataclass(init=True, repr=True)
class LQViTConfig:
    t_in: int
    t_out: int
    n_classes: int
    vit_config: ViTConfig = ViTConfig()

    TIn = property(lambda self: Axis(name='time_in', size=self.t_in))
    TOut = property(lambda self: Axis(name='time', size=self.t_out))
    Spatial = property(lambda self: Axis(name='spatial', size=self.vit_config.n_patches))
    Cls = property(lambda self: Axis(name='cls', size=self.n_classes))

    def wrap_vit_cfg(self, cfg: ViTConfig):
        with jdc.copy_and_mutate(cfg) as cfg:
            cfg.n_patches *= self.t_out
        with jdc.copy_and_mutate(self) as slf:
            slf.vit_config = cfg
            return slf


class LQAttention(eqx.Module):
    config: LQViTConfig
    queries: NamedArray
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    out_proj: hnn.Linear

    @staticmethod
    def init(config: LQViTConfig, *, key) -> 'LQAttention':
        c = config.vit_config
        use_bias = c.qkv_bias
        Embed = c.Embed
        k_q, k_k, k_v, k_out = jrand.split(key, 4)
        queries = hax.random.normal(k_q, (c.Heads, config.TOut, c.HeadSize)) * 0.02
        k_proj = hnn.Linear.init(In=Embed, Out=(c.Heads, c.HeadSize), key=k_k, use_bias=use_bias)
        v_proj = hnn.Linear.init(In=Embed, Out=(c.Heads, c.HeadSize), key=k_v, use_bias=use_bias)
        out_proj = hnn.Linear.init(In=(c.Heads, c.HeadSize), Out=Embed, key=k_out, use_bias=use_bias)
        return LQAttention(
            config=config,
            queries=queries,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        c = self.config

        x = x.rearrange((..., "spatial", "time_in", "embed"))
        k = self.k_proj(x).rearrange((..., "heads", "time_in", "head_size"))
        v = self.v_proj(x).rearrange((..., "heads", "time_in", "head_size"))

        attn_output = hnn.attention.dot_product_attention(
            QPos=c.TOut,
            KPos=c.TIn,
            KeySize=c.vit_config.HeadSize,
            query=self.queries,
            key=k,
            value=v,
        )

        attn_output = attn_output.rearrange((..., "time", "heads", "head_size"))
        attn_output = self.out_proj(attn_output)
        attn_output = hax.flatten_axes(attn_output, ("spatial", "time"), "position")
        return attn_output


class GAPCls(eqx.Module):
    linear: hnn.Linear

    @staticmethod
    def init(Embed: Axis, Cls: Axis, *, key) -> 'GAPCls':
        key = jrand.split(key, 1)
        linear = hnn.Linear.init(Out=Cls, In=Embed, key=key, use_bias=True)
        return GAPCls(
            linear=linear
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        x = hax.mean(x, axis="position")
        return self.linear(x)


class LQViT(eqx.Module, STSerde):
    config: LQViTConfig

    lq_atten: LQAttention
    vit_encoder: ViTEncoder
    cls_head: GAPCls

    @staticmethod
    def init(config: LQViTConfig, *, key) -> 'LQViT':
        assert config.vit_config is not None, "Missing config for ViT"
        lq_key, vit_key, cls_key = jrand.split(key, 3)

        lq_atten = LQAttention.init(config, key=lq_key)
        vit_encoder = ViTEncoder.init(config.vit_config, key=vit_key)
        cls_head = GAPCls.init(config.vit_config.Embed, config.Cls, key=cls_key)

        return LQViT(
            config=config,
            lq_atten=lq_atten,
            vit_encoder=vit_encoder,
            cls_head=cls_head,
        )

    @staticmethod
    def from_pretrained(name: str, path: str, config: LQViTConfig, *, key, dtype=None) -> 'LQViT':
        vit_cfg = ViTConfig.from_hf_config(HFViTConfig.from_pretrained(name))
        config = config.wrap_vit_cfg(vit_cfg)
        slf = LQViT.init(config, key=key)

        if path.startswith('gs://'):
            client = storage.Client()
            bucket = client.get_bucket(path.split('/')[2])
            blob = bucket.blob('/'.join(path.split('/')[3:]))
            sd = load(blob.download_as_bytes())
        else:
            sd = load(path)

        slf.vit_encoder.from_state_dict(sd, prefix='encoder')
        if dtype is not None:
            slf = slf.astype(dtype)
        return

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        key_atten, key_encoder, key_cls = maybe_rng_split(key, 3)
        x = self.lq_atten(x, key=key_atten)
        x = self.vit_encoder(x, key=key_encoder)
        x = self.cls_head(x, key=key_cls)
        x = hnn.softmax(x, axis="cls")
        return x

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'lq_atten': 'lq_atten',
            'vit_encoder': 'encoder',
            'cls_head': 'cls_head',
        }

    def astype(self, dtype: jnp.dtype) -> 'LQViT':
        cast_fn = lambda x: x.astype(dtype) if isinstance(x, jnp.ndarray) else x
        return jax.tree_map(cast_fn, self)
