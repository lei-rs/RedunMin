from dataclasses import dataclass
from functools import partial
from typing import Callable, Union, Optional, Dict

import equinox as eqx
import haliax.nn as hnn
import jax.random as jrand
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, maybe_rng_split, shaped_rng_split
from haliax.nn.scan import Stacked
from transformers.models.vit import ViTConfig as HFViTConfig

from .levanter.flash_attention import flash_attention
from .levanter.safetensor import (
    StateDict,
    STSerde,
    apply_prefix,
    unstack_state_dict,
    stack_state_dict,
    unflatten_linear_layers,
    flatten_linear_layers,
)

TRAINING = False

ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}


def update_train_flag(mode: bool):
    global TRAINING
    TRAINING = mode


@dataclass
class ViTConfig:
    n_patches: int = (224 ** 2 // 16 ** 2)
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = 'gelu'
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    patch_size: int = 16
    qkv_bias: bool = True
    dropout = 0.1
    flash_attention: bool = False

    Pos = property(lambda self: Axis(name="position", size=self.n_patches))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_size))
    Layers = property(lambda self: Axis(name="layers", size=self.num_hidden_layers))
    Heads = property(lambda self: Axis(name="heads", size=self.num_attention_heads))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_size))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_size // self.num_attention_heads))

    @classmethod
    def from_hf_config(cls, hf_config: HFViTConfig, flash_attention: bool = False) -> 'ViTConfig':
        return ViTConfig(
            n_patches=(hf_config.image_size ** 2 // hf_config.patch_size ** 2),
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=hf_config.hidden_act,
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            initializer_range=hf_config.initializer_range,
            layer_norm_eps=hf_config.layer_norm_eps,
            patch_size=hf_config.patch_size,
            qkv_bias=hf_config.qkv_bias,
            flash_attention=flash_attention,
        )


class ViTMLP(eqx.Module, STSerde):
    to_hidden: hnn.Linear
    from_hidden: hnn.Linear
    act: Callable = eqx.static_field()
    dropout: float = eqx.static_field()

    Embed: Axis = eqx.static_field()

    @staticmethod
    def init(
            Embed: Axis,
            Mlp: Axis,
            activation_fn:
            Union[str, Callable],
            *,
            key,
            use_bias: bool = False,
            dropout: float = 0.1,
    ) -> 'ViTMLP':
        k_to_embed, k_from_embed = jrand.split(key)

        to_hidden = hnn.Linear.init(Out=Mlp, In=Embed, key=k_to_embed, use_bias=use_bias, out_first=True)
        from_hidden = hnn.Linear.init(Out=Embed, In=Mlp, key=k_from_embed, use_bias=use_bias, out_first=True)

        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn

        return ViTMLP(
            to_hidden=to_hidden,
            from_hidden=from_hidden,
            act=act,
            dropout=dropout,
            Embed=Embed,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_dropout = maybe_rng_split(key, 1)
        hidden = self.to_hidden(x)
        hidden = self.act(hidden)
        if self.dropout > 0:
            hidden = hnn.dropout(
                hidden,
                self.dropout,
                broadcast_axes=self.Embed,
                key=k_dropout,
                inference=not TRAINING,
            )
        hidden = self.from_hidden(hidden)
        return hidden

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'to_hidden': 'intermediate.dense',
            'from_hidden': 'output.dense',
        }


class ViTAttention(eqx.Module, STSerde):
    config: ViTConfig = eqx.static_field()
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    out_proj: hnn.Linear

    flash_attention: bool = eqx.static_field()

    @staticmethod
    def init(config: ViTConfig, *, key) -> 'ViTAttention':
        use_bias = config.qkv_bias
        Embed = config.Embed
        k_q, k_k, k_v, k_out = jrand.split(key, 4)
        q_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_q, use_bias=use_bias)
        k_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_k, use_bias=use_bias)
        v_proj = hnn.Linear.init(In=Embed, Out=(config.Heads, config.HeadSize), key=k_v, use_bias=use_bias)
        out_proj = hnn.Linear.init(In=(config.Heads, config.HeadSize), Out=Embed, key=k_out, use_bias=use_bias)
        return ViTAttention(
            config=config,
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            out_proj=out_proj,
            flash_attention=config.flash_attention,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        c = self.config

        k_attn = maybe_rng_split(key, 1)
        q = self.q_proj(x).rearrange((..., "heads", "position", "head_size"))
        k = self.k_proj(x).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(x).rearrange((..., "heads", "position", "head_size"))
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})

        if self.flash_attention:
            attn_output = flash_attention(
                c.Pos,
                c.KeyPos,
                c.HeadSize,
                q,
                k,
                v,
                dropout=c.dropout,
                inference=False,
                key=k_attn,
            )
        else:
            attn_output = hnn.attention.dot_product_attention(
                c.Pos,
                c.KeyPos,
                c.HeadSize,
                q,
                k,
                v,
            )

        attn_output = attn_output.rearrange((..., "position", "heads", "head_size"))
        attn_output = self.out_proj(attn_output)
        return attn_output

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'q_proj': 'attention.query',
            'k_proj': 'attention.key',
            'v_proj': 'attention.value',
            'out_proj': 'output.dense',
        }

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        d = {}
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'attention.query'),
                state_dict,
                self.q_proj,
                True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'attention.key'),
                state_dict,
                self.k_proj,
                True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'attention.value'),
                state_dict,
                self.v_proj,
                True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'output.dense'),
                state_dict,
                self.out_proj,
                True
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'attention.query'),
                self.q_proj,
                True
            )
        )
        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'attention.key'),
                self.k_proj,
                True
            )
        )
        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'attention.value'),
                self.v_proj,
                True
            )
        )
        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'output.dense'),
                self.out_proj,
                True
            )
        )

        state_dict.update(my_dict)
        return state_dict


class VitEncoderLayer(eqx.Module, STSerde):
    config: ViTConfig = eqx.static_field()
    attention: ViTAttention
    mlp: ViTMLP
    ln1: hnn.LayerNorm
    ln2: hnn.LayerNorm

    @staticmethod
    def init(config: ViTConfig, *, key) -> 'VitEncoderLayer':
        k_attn, key_mlp = jrand.split(key, 2)
        attention = ViTAttention.init(config, key=k_attn)
        mlp = ViTMLP.init(config.Embed, config.Mlp, config.hidden_act, key=key_mlp)
        ln1 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps)
        ln2 = hnn.LayerNorm.init(config.Embed, eps=config.layer_norm_eps)
        return VitEncoderLayer(
            config=config,
            attention=attention,
            mlp=mlp,
            ln1=ln1,
            ln2=ln2,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        k_attn, k_mlp = jrand.split(key, 2)
        x = self.ln1(x)
        attn_output = self.attention(x, key=key)
        x = x + attn_output
        x = self.ln2(x)
        mlp_output = self.mlp(x, key=k_mlp)
        x = x + mlp_output
        return x

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'attention': 'attention',
            'mlp': None,
            'ln1': 'layernorm_before',
            'ln2': 'layernorm_after',
        }


class ViTEncoder(eqx.Module, STSerde):
    config: ViTConfig = eqx.static_field()
    layers: Stacked[VitEncoderLayer]

    @staticmethod
    def init(config: ViTConfig, *, key) -> 'ViTEncoder':
        layers = Stacked.init(config.Layers, VitEncoderLayer)(
            config,
            key=shaped_rng_split(key, config.num_hidden_layers),
        )
        return ViTEncoder(
            config=config,
            layers=layers,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        keys = jrand.split(key, self.config.num_hidden_layers)
        x = self.layers.fold(x, key=keys)
        return x

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'layers': 'layer',
        }

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None):
        stacked = stack_state_dict(state_dict, prefix=apply_prefix(prefix, "layer"))
        out = super().from_state_dict(stacked, prefix=prefix)
        return out

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_state_dict: StateDict = {}
        super().update_state_dict(my_state_dict, prefix=prefix)

        stacked_dict = unstack_state_dict(my_state_dict, prefix=apply_prefix(prefix, "layer"))
        state_dict.update(stacked_dict)

        return state_dict
