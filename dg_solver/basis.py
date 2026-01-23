import jax
import jax.numpy as jnp

def phi_at(x, xL, xR):
    h = xR - xL
    xi = 2.0 * (x - xL) / h - 1.0
    phi0 = 0.5 * (1.0 - xi)
    phi1 = 0.5 * (1.0 + xi)
    return jnp.stack([phi0, phi1])

vphi_at = jax.vmap(phi_at, in_axes=(0, None, None))