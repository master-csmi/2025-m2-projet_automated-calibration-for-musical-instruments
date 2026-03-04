import jax.numpy as jnp

def local_mass_inv_system(h):
    M_inv = (6.0 / h) * jnp.array([[2/3, -1/3],
                                   [-1/3, 2/3]])
    return M_inv, M_inv