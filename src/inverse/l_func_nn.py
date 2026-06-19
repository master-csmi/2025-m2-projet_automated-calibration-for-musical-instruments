import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Callable



class LFuncNN(eqx.Module):
    layers: list
    activation: Callable = eqx.static_field()

    def __init__(self, layer_sizes, activation=jax.nn.tanh, key=None):

        if key is None:
            key = jax.random.PRNGKey(0)

        keys = jax.random.split(key, len(layer_sizes) - 1)
        self.layers = [
            eqx.nn.Linear(i, o, key=k)
            for i, o, k in zip(layer_sizes[:-1], layer_sizes[1:], keys)
        ]
        self.activation = activation

    def __call__(self, y):
        x = jnp.array([y])

        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            
        # sortie positive (biais physique à vérifier)
        g = jax.nn.softplus(self.layers[-1](x)[0])
        return jax.nn.relu(y) * g