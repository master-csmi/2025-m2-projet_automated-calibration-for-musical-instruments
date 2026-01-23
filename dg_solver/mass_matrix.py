import jax.numpy as jnp

def local_mass_inv_system(h, S_cell, c=1.0, S_star=1.0):
    M_ref = (h / 6.0) * jnp.array([[2., 1.],
                                  [1., 2.]])
    Mp = (S_cell / (c * S_star)) * M_ref
    Mv = (S_star / (c * S_cell)) * M_ref
    return jnp.linalg.inv(Mp), jnp.linalg.inv(Mv)