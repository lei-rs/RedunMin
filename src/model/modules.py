from typing import Callable, Optional, Dict

import equinox as eqx
import haliax.nn as hnn
import jax.random as jrand
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call

from .levanter.safetensor import Serialize


class FeedForward(eqx.Module, Serialize):
    to_hidden: hnn.Linear
    from_hidden: hnn.Linear

    Hidden: Axis = eqx.static_field()

    act: Callable = eqx.static_field(default=hnn.gelu)
    dropout: Optional[hnn.Dropout] = eqx.static_field(default=None)
    inference: bool = eqx.static_field(default=False)
    sd_key_map: Optional[Dict[str, str]] = eqx.static_field(default=None)

    @staticmethod
    def init(
            In: Axis,
            Hidden: Axis,
            act: Callable,
            *,
            key,
            use_bias: bool = False,
            dropout: float = 0.1,
            out_first: bool = False,
            sd_key_map=None,
    ) -> 'FeedForward':
        k_to_embed, k_from_embed = jrand.split(key)

        to_hidden = hnn.Linear.init(Out=Hidden, In=In, key=k_to_embed, use_bias=use_bias, out_first=out_first)
        from_hidden = hnn.Linear.init(Out=In, In=Hidden, key=k_from_embed, use_bias=use_bias, out_first=out_first)
        if dropout > 0:
            dropout = hnn.Dropout(dropout, broadcast_axes=Hidden)
        else:
            dropout = None

        return FeedForward(
            to_hidden=to_hidden,
            from_hidden=from_hidden,
            act=act,
            Hidden=Hidden,
            dropout=dropout,
            inference=False,
            sd_key_map=sd_key_map,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        k_dropout = jrand.split(key, 1)
        x = self.to_hidden(x)
        x = self.act(x)
        if self.dropout is not None:
            x = self.dropout(x, key=k_dropout, inference=self.inference)
        x = self.from_hidden(x)
        return x

    def set_inference(self, inference: bool):
        self.inference = inference


class Attention(eqx.Module, Serialize):
    qkv_proj: hnn.Linear
    out_proj: hnn.Linear

    Pos: Axis = eqx.static_field()
    KeyPos: Axis = eqx.static_field()
    HeadSize: Axis = eqx.static_field()

    @staticmethod
    def init(
        Pos: Axis,
        KeyPos: Axis,
        Embed: Axis,
        Heads: Axis,
        HeadSize: Axis,
        *,
        key,
        use_bias: bool = False,
    ) -> 'Attention':
        k_qkv, k_out = jrand.split(key, 2)
        QKV = Axis(name="qkv", size=3)
        qkv_proj = hnn.Linear.init(In=Embed, Out=(QKV, Heads, HeadSize), key=k_qkv, use_bias=use_bias)
        out_proj = hnn.Linear.init(In=(Heads, HeadSize), Out=Embed, key=k_out, use_bias=use_bias)

        return Attention(
            qkv_proj=qkv_proj,
            out_proj=out_proj,
            Pos=Pos,
            KeyPos=KeyPos,
            HeadSize=HeadSize,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        qkv = self.qkv_proj(x).rearrange((..., "qkv", "heads", "head_size"))
        q, k, v = qkv.unbind("qkv")
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
        attn_output = hnn.attention.dot_product_attention(
            self.Pos,
            self.KeyPos,
            self.HeadSize,
            q,
            k,
            v,
        )
        attn_output = attn_output.rearrange((..., "position", "heads", "head_size"))
        attn_output = self.out_proj(attn_output)
        return attn_output
