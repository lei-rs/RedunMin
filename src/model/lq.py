from typing import Optional, Dict, Tuple

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
import jax.random as jrand
import jax_dataclasses as jdc
from google.cloud import storage
from haliax import NamedArray, Axis
from haliax.jax_utils import named_call
from safetensors.numpy import load
from transformers import ViTConfig as HFViTConfig

from .levanter.safetensor import Serialize
from .vit import ViTConfig, ViTEncoder


@jdc.pytree_dataclass(init=True, repr=True)
class LQViTConfig:
    t_dims: Tuple[int, ...] = (32, 8)
    n_classes: int = 400
    vit_config: ViTConfig = ViTConfig()

    Temporal = property(lambda self: Axis(name='temporal', size=self.t_dims[-1]))
    Spatial = property(lambda self: Axis(name='spatial', size=self.vit_config.n_patches_per_frame()))

    Height = property(lambda self: Axis(name='height', size=self.vit_config.image_size))
    Width = property(lambda self: Axis(name='width', size=self.vit_config.image_size))
    Channels = property(lambda self: Axis(name='channels', size=3))

    Cls = property(lambda self: Axis(name='cls', size=self.n_classes))

    def wrap_vit_cfg(self, cfg: ViTConfig):
        with jdc.copy_and_mutate(cfg) as cfg:
            cfg.n_patches *= self.t_dims[-1]
        with jdc.copy_and_mutate(self) as slf:
            slf.vit_config = cfg
            return slf


class PatchEmbeddings(eqx.Module):
    conv: hnn.Conv

    @staticmethod
    def init(cfg: LQViTConfig, *, key):
        patch = cfg.vit_config.patch_size
        assert cfg.vit_config.image_size % patch == 0, 'Image size must be divisible by patch size'
        conv = hnn.Conv.init(
            (cfg.Height, cfg.Width),
            cfg.Channels,
            cfg.vit_config.Embed,
            (patch, patch),
            key=key,
            stride=(patch, patch),
        )
        return PatchEmbeddings(
            conv=conv,
        )

    @named_call
    def __call__(self, x: NamedArray) -> NamedArray:
        x = x.rearrange((..., 'height', 'width', 'channels'))
        x = self.conv(x)
        x = x.flatten_axes(('height', 'width'), 'spatial')
        x = x.rename({'channels': 'embed'})
        return x


class RevN(eqx.Module):
    Norm: Axis = eqx.static_field()

    affine: Optional[NamedArray] = None
    bias: Optional[NamedArray] = None
    eps: float = 1e-5

    @staticmethod
    def init(Norm: Axis, Embed: Axis, *, key=None, affine=True, eps=1e-5) -> 'RevN':
        if affine:
            affine = hax.ones(Embed)
            bias = hax.zeros(Embed)
        return RevN(
            Norm=Norm,
            affine=affine,
            bias=bias,
            eps=eps,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        mu = hax.mean(x, self.Temporal)
        var = hax.var(x, self.Temporal)
        x = (x - mu) / hax.sqrt(var + self.eps)
        if self.affine is not None:
            x = x * self.affine + self.bias
        return x, mu, var

    @named_call
    def reverse(self, x: NamedArray, mu: NamedArray, sigma: NamedArray, *, key=None) -> NamedArray:
        if self.affine is not None:
            x = (x - self.bias) / self.affine
        x = x * hax.sqrt(sigma + self.eps) + mu
        return x


class RegPool(eqx.Module):
    queries: NamedArray
    q_proj: hnn.Linear
    kv_proj: hnn.Linear
    out_proj: hnn.Linear

    TIn: Axis = eqx.static_field()
    TOut: Axis = eqx.static_field()
    HeadSize: Axis = eqx.static_field()

    @staticmethod
    def init(cfg: LQViTConfig, t_in: int, t_out: int, *, key, use_bias=True) -> 'RegPool':
        TIn = Axis(name='temporal', size=t_in)
        TOut = Axis(name='temp_out', size=t_out)
        Embed = cfg.vit_config.Embed
        Heads = cfg.vit_config.Heads
        HeadSize = cfg.vit_config.HeadSize

        key, k_q = jrand.split(key, 2)
        queries = hax.random.normal(k_q, (TOut, Embed))

        k_q, k_kv, k_out = jrand.split(key, 3)
        KV = Axis(name='kv', size=2)
        q_proj = hnn.Linear.init(In=Embed, Out=(Heads, HeadSize), key=k_q, use_bias=use_bias)
        kv_proj = hnn.Linear.init(In=Embed, Out=(KV, Heads, HeadSize), key=k_kv, use_bias=use_bias)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=use_bias)

        return RegPool(
            queries=queries,
            q_proj=q_proj,
            kv_proj=kv_proj,
            out_proj=out_proj,
            TIn=TIn,
            TOut=TOut,
            HeadSize=HeadSize,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        x = x.rearrange((..., 'spatial', 'temporal', 'embed'))
        q = hax.broadcast_to(
            self.queries.rename({'temp_out': 'temporal'}),
            x.axes[:-2],
            enforce_no_extra_axes=False
        )
        x = hax.concatenate('temporal', [x, q])
        return self.pool(x)

    @named_call
    def pool(self, x: NamedArray, *, key=None) -> NamedArray:
        q = x['temporal', -self.TOut.size:].rename({'temporal': 'temp_out'})
        kv = x['temporal', :self.TIn.size]
        q = self.q_proj(q)
        kv = self.kv_proj(kv)
        k, v = kv.unbind('kv')
        attn_output = hnn.attention.dot_product_attention(
            self.TOut,
            self.TIn,
            self.HeadSize,
            q,
            k,
            v,
        )
        attn_output = attn_output.rearrange((..., 'temp_out', 'heads', 'head_size'))
        attn_output = self.out_proj(attn_output)
        attn_output = attn_output.rename({'temp_out': 'temporal'})
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


class LQViT(eqx.Module, Serialize):
    config: LQViTConfig

    patch_embed: PatchEmbeddings
    pos_embed: NamedArray
    pool: RegPool
    vit_encoder: ViTEncoder
    cls_head: GAPCls

    @staticmethod
    def init(config: LQViTConfig, *, key) -> 'LQViT':
        assert config.vit_config is not None, "Missing config for ViT"
        patch_key, pe_key, lq_key, vit_key, cls_key = jrand.split(key, 5)

        TFirst = Axis(name='temporal', size=config.t_dims[0])

        patch_embed = PatchEmbeddings.init(config, key=patch_key)
        pos_embed = hax.random.normal(
            pe_key,
            (
                TFirst,
                config.Spatial,
                config.vit_config.Embed
            )
        )
        pool = RegPool.init(config, config.t_dims[0], config.t_dims[-1], key=lq_key)
        vit_encoder = ViTEncoder.init(config.vit_config, key=vit_key)
        cls_head = GAPCls.init(config.vit_config.Embed, config.Cls, key=cls_key)

        return LQViT(
            config=config,
            patch_embed=patch_embed,
            pos_embed=pos_embed,
            pool=pool,
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
        return slf

    @named_call
    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        _, key_vit = jrand.split(key)
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pool(x)
        x = x.flatten_axes(('spatial', 'temporal'), 'position')
        x = self.vit_encoder(x, key=key_vit)
        x = self.cls_head(x)
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
