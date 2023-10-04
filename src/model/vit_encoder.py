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

from serialize.safetensor import (
    StateDict,
    STSerde,
    unflatten_linear_layers,
    flatten_linear_layers,
    apply_prefix, stack_state_dict, unstack_state_dict
)


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": partial(hnn.gelu, approximate=False),
    "gelu_new": partial(hnn.gelu, approximate=True),
    "quick_gelu": hnn.quick_gelu,
}


@dataclass(frozen=True)
class ViTConfig:
    len: int = 8 * (224 ** 2 // 16 ** 2)
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

    Pos = property(lambda self: Axis(name="position", size=self.len))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_size))
    Layers = property(lambda self: Axis(name="layers", size=self.num_hidden_layers))
    Heads = property(lambda self: Axis(name="heads", size=self.num_attention_heads))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_size))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_size // self.num_attention_heads))

    @classmethod
    def from_hf_config(cls, hf_config: HFViTConfig) -> 'ViTConfig':
        return ViTConfig(
            len=(hf_config.image_size ** 2 // hf_config.patch_size ** 2),
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
        )


class ViTMLP(eqx.Module, STSerde):
    to_embed: hnn.Linear
    from_embed: hnn.Linear
    act: Callable = eqx.static_field()

    @staticmethod
    def init(
            Embed: Axis,
            Mlp: Axis,
            activation_fn:
            Union[str, Callable],
            *,
            key,
            use_bias: bool = False
    ) -> 'ViTMLP':
        k_to_embed, k_from_embed = jrand.split(key)
        to_embed = hnn.Linear.init(Out=Mlp, In=Embed, key=k_to_embed, use_bias=use_bias)
        from_embed = hnn.Linear.init(Out=Embed, In=Mlp, key=k_from_embed, use_bias=use_bias)
        if isinstance(activation_fn, str):
            activation_fn = ACT2FN[activation_fn]
        act = activation_fn
        return ViTMLP(
            to_embed=to_embed,
            from_embed=from_embed,
            act=act,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_to_embed, k_from_embed = jrand.split(key)
        hidden = self.to_embed(x, key=k_to_embed)
        hidden = self.act(hidden)
        hidden = self.from_embed(hidden, key=k_from_embed)
        return hidden

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'to_embed': 'intermediate.dense',
            'from_embed': 'output.dense',
        }


class ViTAttention(eqx.Module, STSerde):
    config: ViTConfig = eqx.static_field()
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    out_proj: hnn.Linear

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
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        key_q, key_k, key_v, key_o = maybe_rng_split(key, 4)

        q = self.q_proj(x, key=key_q).rearrange((..., "heads", "position", "head_size"))
        k = self.k_proj(x, key=key_k).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(x, key=key_v).rearrange((..., "heads", "position", "head_size"))

        c = self.config
        attn_output = hnn.attention.dot_product_attention(c.Pos, c.KeyPos, c.HeadSize, q, k, v,)
        attn_output = attn_output.rearrange((..., "position", "heads", "head_size"))
        attn_output = self.out_proj(attn_output, key=key_o)
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
        mlp = ViTMLP.init(config.Mlp, config.Embed, config.hidden_act, key=key_mlp)
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
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        k_attn, k_mlp = maybe_rng_split(key, 2)
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
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        keys = maybe_rng_split(key, self.config.num_hidden_layers) if key is not None else None
        x = self.layers.fold(x, keys=keys)
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


if __name__ == '__main__':
    from transformers import ViTModel
    import jax.numpy as jnp

    hf_model = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
    config = ViTConfig.from_hf_config(hf_model.config)
    sd = hf_model.state_dict()
    del hf_model
    for k, v in sd.items():
        sd[k] = jnp.array(v.numpy())

    key = jrand.PRNGKey(0)
    ViTEncoder.init(config, key=key).from_state_dict(
        state_dict=sd,
        prefix='encoder',
    )
