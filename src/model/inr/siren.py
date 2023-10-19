from typing import Tuple

import equinox as eqx
import haliax as hax
import haliax.nn as hnn
import jax
import jax.numpy as jnp
from haliax import Axis, NamedArray
from haliax.jax_utils import named_call, shaped_rng_split
from haliax.nn import Stacked


class SirenLayer(eqx.Module):
    linear: hnn.Linear
    omega: float = eqx.static_field()

    @staticmethod
    def init(in_f: Axis, out_f: Axis, *, key, bias=True, omega=30.) -> 'SirenLayer':
        _, key_l = jax.random.split(key)
        linear = hnn.Linear.init(in_f, out_f, key=key, use_bias=bias)
        linear = eqx.tree_at(
            lambda x: x.weight,
            linear,
            hax.random.uniform(
                key_l,
                (in_f, out_f),
                minval=-1 / in_f.size,
                maxval=1 / in_f.size,
            )
        )
        return SirenLayer(
            linear=linear,
            omega=omega,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        x = hax.sin(self.linear(x))
        return x


class Siren(eqx.Module):
    first: SirenLayer
    layers: Stacked[SirenLayer]
    last: hnn.Linear

    Layer: Axis = eqx.static_field()

    @staticmethod
    def init(
            in_f: Axis,
            out_f: Axis,
            hidden_size: int,
            n_layers,
            *,
            key,
            first_omega=30.,
            hidden_omega=30.
    ) -> 'Siren':
        hidden_in = Axis(name='hidden_in', size=hidden_size)
        hidden_out = Axis(name='hidden_out', size=hidden_size)
        layer = Axis(name='layer', size=n_layers - 1)

        hidden_omega = jnp.asarray([hidden_omega] * layer.size)

        key_f, key_s, key_l = jax.random.split(key, 3)
        key_s = shaped_rng_split(key_s, (layer.size,))

        first = SirenLayer.init(in_f, hidden_in, key=key_f, omega=first_omega)
        make_layers = lambda k, o: SirenLayer.init(
            hidden_in,
            hidden_out,
            key=k,
            omega=o
        )
        layers = hax.vmap(make_layers, layer)(key_s, hidden_omega)
        last = hnn.Linear.init(hidden_in, out_f, key=key_l)
        return Siren(
            first=first,
            layers=layers,
            last=last,
            Layer=layer,
        )

    @named_call
    def __call__(self, x: NamedArray, *, key=None) -> NamedArray:
        def _scan(_x: NamedArray, _layer: SirenLayer) -> Tuple[NamedArray, None]:
            _x = _layer(_x)
            return _x.rename({'hidden_out': 'hidden_in'}), None

        x = self.first(x)
        x, _ = hax.scan(_scan, self.Layer)(x, self.layers)
        return self.last(x)
