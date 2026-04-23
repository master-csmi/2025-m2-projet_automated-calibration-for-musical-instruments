import jax.numpy as jnp

def local_mass_inv_system(h):
    M_ref = (h / 6.0) * jnp.array([[2., 1.],
                                  [1., 2.]])

    return jnp.linalg.inv(M_ref), jnp.linalg.inv(M_ref)