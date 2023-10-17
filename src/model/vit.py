from typing import Callable, Optional, Dict

import equinox as eqx
import haliax.nn as hnn
import jax.random as jax_rand
import jax_dataclasses as jdc
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn.scan import Stacked
from transformers.models.vit import ViTConfig as HFViTConfig

from .levanter.serialize import (
    StateDict,
    Serialize,
    apply_prefix,
    unstack_state_dict,
    stack_state_dict,
    unflatten_linear_layers,
    flatten_linear_layers,
)


ACT2FN: Dict[str, Callable] = {
    "relu": hnn.relu,
    "silu": hnn.silu,
    "swish": hnn.swish,
    "gelu": hnn.quick_gelu,
}


@jdc.pytree_dataclass(init=True, repr=True)
class ViTConfig:
    n_patches: int = (224 ** 2 // 16 ** 2) * 8
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: Callable = hnn.quick_gelu
    hidden_dropout_prob: float = 0.1
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    patch_size: int = 16
    qkv_bias: bool = True
    dropout: float = 0.1
    image_size: int = 224

    Pos = property(lambda self: Axis(name="position", size=self.n_patches))
    KeyPos = property(lambda self: self.Pos.alias("key_position"))
    Embed = property(lambda self: Axis(name="embed", size=self.hidden_size))
    Layers = property(lambda self: Axis(name="layers", size=self.num_hidden_layers))
    Heads = property(lambda self: Axis(name="heads", size=self.num_attention_heads))
    Mlp = property(lambda self: Axis(name="mlp", size=self.intermediate_size))
    HeadSize = property(lambda self: Axis(name="head_size", size=self.hidden_size // self.num_attention_heads))

    @classmethod
    def from_hf_config(cls, hf_config: HFViTConfig):
        return ViTConfig(
            n_patches=(hf_config.image_size ** 2 // hf_config.patch_size ** 2),
            hidden_size=hf_config.hidden_size,
            num_hidden_layers=hf_config.num_hidden_layers,
            num_attention_heads=hf_config.num_attention_heads,
            intermediate_size=hf_config.intermediate_size,
            hidden_act=ACT2FN[hf_config.hidden_act],
            hidden_dropout_prob=hf_config.hidden_dropout_prob,
            initializer_range=hf_config.initializer_range,
            layer_norm_eps=hf_config.layer_norm_eps,
            patch_size=hf_config.patch_size,
            qkv_bias=hf_config.qkv_bias,
            dropout=hf_config.hidden_dropout_prob,
            image_size=hf_config.image_size,
        )

    def n_patches_per_frame(self) -> int:
        return self.image_size ** 2 // self.patch_size ** 2


class ViTMLP(Serialize, eqx.Module):
    to_hidden: hnn.Linear
    from_hidden: hnn.Linear

    Embed: Axis = eqx.static_field()

    act: Callable = eqx.static_field(default=hnn.gelu)
    dropout: Optional[hnn.Dropout] = eqx.static_field(default=None)
    inference: bool = eqx.field(default=False)

    @staticmethod
    def init(
            Embed: Axis,
            Mlp: Axis,
            *,
            key,
            act: Callable = hnn.gelu,
            use_bias: bool = False,
            dropout: float = 0.1,
    ) -> 'ViTMLP':
        k_to_embed, k_from_embed = jax_rand.split(key, 2)

        to_hidden = hnn.Linear.init(Out=Mlp, In=Embed, key=k_to_embed, use_bias=use_bias)
        from_hidden = hnn.Linear.init(Out=Embed, In=Mlp, key=k_from_embed, use_bias=use_bias)
        if dropout > 0:
            dropout = hnn.Dropout(dropout, broadcast_axes=Embed)
        else:
            dropout = None

        return ViTMLP(
            to_hidden=to_hidden,
            from_hidden=from_hidden,
            Embed=Embed,
            act=act,
            dropout=dropout,
            inference=False,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key) -> NamedArray:
        _, k_dropout = jax_rand.split(key, 2)
        x = self.to_hidden(x)
        x = self.act(x)
        x = self.dropout(x, key=k_dropout, inference=self.inference)
        x = self.from_hidden(x)
        return x

    def set_inference(self, inference: bool) -> 'ViTMLP':
        return eqx.tree_at(
            lambda x: x.inference,
            self,
            inference
        )

    def _state_dict_key_map(self) -> Optional[Dict[str, Optional[str]]]:
        return {
            'to_hidden': 'intermediate.dense',
            'from_hidden': 'output.dense',
        }

    def from_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> 'ViTMLP':
        d = {}
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'intermediate.dense'),
                state_dict,
                self.to_hidden,
                True
            )
        )
        d.update(
            unflatten_linear_layers(
                apply_prefix(prefix, 'output.dense'),
                state_dict,
                self.from_hidden,
                True
            )
        )
        return super().from_state_dict(d, prefix)

    def update_state_dict(self, state_dict: StateDict, prefix: Optional[str] = None) -> StateDict:
        my_dict: StateDict = {}
        super().update_state_dict(my_dict, prefix=prefix)

        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'intermediate.dense'),
                self.to_hidden,
                True
            )
        )
        my_dict.update(
            flatten_linear_layers(
                apply_prefix(prefix, 'output.dense'),
                self.from_hidden,
                True
            )
        )

        state_dict.update(my_dict)
        return state_dict


class ViTAttention(Serialize, eqx.Module):
    config: ViTConfig = eqx.static_field()
    q_proj: hnn.Linear
    k_proj: hnn.Linear
    v_proj: hnn.Linear
    out_proj: hnn.Linear

    @staticmethod
    def init(config: ViTConfig, *, key) -> 'ViTAttention':
        use_bias = config.qkv_bias
        Embed = config.Embed
        k_q, k_k, k_v, k_out = jax_rand.split(key, 4)
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
        c = self.config
        q = self.q_proj(x).rearrange((..., "heads", "position", "head_size"))
        k = self.k_proj(x).rearrange((..., "heads", "position", "head_size"))
        v = self.v_proj(x).rearrange((..., "heads", "position", "head_size"))
        k = k.rename({"position": "key_position"})
        v = v.rename({"position": "key_position"})
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


class VitEncoderLayer(Serialize, eqx.Module):
    config: ViTConfig = eqx.static_field()
    attention: ViTAttention
    mlp: ViTMLP
    ln1: hnn.LayerNorm
    ln2: hnn.LayerNorm

    @staticmethod
    def init(config: ViTConfig, *, key) -> 'VitEncoderLayer':
        k_attn, key_mlp = jax_rand.split(key, 2)
        attention = ViTAttention.init(config, key=k_attn)
        mlp = ViTMLP.init(config.Embed, config.Mlp, key=key_mlp, act=config.hidden_act, use_bias=True)
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
        _, k_mlp = jax_rand.split(key, 2)
        x = self.ln1(x)
        attn_output = self.attention(x)
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


class ViTEncoder(Serialize, eqx.Module):
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
        keys = jax_rand.split(key, self.config.num_hidden_layers)
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
