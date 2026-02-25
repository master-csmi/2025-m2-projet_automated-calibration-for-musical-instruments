import jax.numpy as jnp

# Right Hand Side of the ODE for phi at right BC
def phi_rhs(pR, alpha, Z):
    return -jnp.sqrt(alpha)/ (Z) * pR

def pressure_func(delta_p):
     return jnp.sqrt(delta_p)*jnp.sign(delta_p)

def l(y):
    return y  # simple linear opening