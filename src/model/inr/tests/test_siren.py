from haliax import Axis

from src.model.inr.siren import SirenLayer, Siren
import jax
import haliax as hax
import unittest


class TestSirenLayer(unittest.TestCase):
    def test_init(self):
        In = Axis(name='in', size=8)
        Out = Axis(name='out', size=4)
        SirenLayer.init(In, Out, key=jax.random.PRNGKey(0))

    def test_call(self):
        In = Axis(name='in', size=8)
        Out = Axis(name='out', size=4)
        x = hax.random.normal(jax.random.PRNGKey(0), (Axis('temp', 8), In))
        s = SirenLayer.init(In, Out, key=jax.random.PRNGKey(0))
        self.assertEquals(s(x).axes, (Axis('temp', 8), Out))


class TestSiren(unittest.TestCase):
    def test_init(self):
        In = Axis(name='in', size=8)
        Out = Axis(name='out', size=4)
        Siren.init(In, Out, hidden_size=16, n_layers=2, key=jax.random.PRNGKey(0))

    def test_call(self):
        In = Axis(name='in', size=8)
        Out = Axis(name='out', size=4)
        x = hax.random.normal(jax.random.PRNGKey(0), (Axis('temp', 8), In))
        s = Siren.init(In, Out, hidden_size=16, n_layers=2, key=jax.random.PRNGKey(0))
        self.assertEquals(s(x).axes, (Axis('temp', 8), Out))
