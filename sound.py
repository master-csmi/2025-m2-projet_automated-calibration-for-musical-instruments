from dataclasses import dataclass
import numpy as np
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import matplotlib.pyplot as plt

@dataclass
class Data(eqx.Module):
    S: jnp.ndarray
    Z_T_star: jnp.ndarray
    c: float
    S_star: jnp.ndarray
    
def mesh_points(gauss_point: jnp.ndarray):
    N=gauss_point.shape[0]
    dx = 1.0 / N
    xR = gauss_point[1:-1] - dx / 2
    xL = gauss_point[1:-1] + dx / 2
    return xL, xR