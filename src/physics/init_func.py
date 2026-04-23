import jax.numpy as jnp

# ------------------------------------------------------------------------------------------------------------------------------
#                                           Initial compactly supported bump function
# ------------------------------------------------------------------------------------------------------------------------------


def init_func(x, L, phi0=1.0):
    xi = 4.0 * (x - 0.5 * L) / L

    inside = xi**2 < 1.0

    # safe denominator (never zero)
    denom = jnp.where(inside, 1.0 - xi**2, 1.0)

    bump = (phi0 / 4.0) * jnp.exp(1.0 - 1.0 / denom)

    return jnp.where(inside, bump, 0.0)

def init_func_const(x, L):
    
    return 0.0