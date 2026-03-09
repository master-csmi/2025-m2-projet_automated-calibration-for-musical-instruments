import jax.numpy as jnp

# Right Hand Side of the ODE for phi at right BC
def phi_rhs(pR, alpha, Z):
    return -jnp.sqrt(alpha)/ (Z) * pR

def pressure_func(delta_p):
     return jnp.sqrt(delta_p)*jnp.sign(delta_p)

def l(y):
    return jnp.maximum(0, y)

def compute_v_bc_left(y, y_t, p_in, zeta, gamma, eps, kappa, omega_r):

    return zeta * l(y) * pressure_func(gamma - p_in) + eps * kappa / omega_r *  y_t