import jax.numpy as jnp

# Function of the difference of pressures at the lips or the reed
def F_p(p1, p2):
    return jnp.sign(p1 - p2) * jnp.sqrt(jnp.abs(p1 - p2))

# Function describing the opening of the player's lips
def l(y):
    return y

