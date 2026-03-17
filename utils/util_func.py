import jax.numpy as jnp

# Right Hand Side of the ODE for phi at right BC
def phi_rhs(pR, alpha, Z):
    return -jnp.sqrt(alpha)/ (Z) * pR

# Right Hand Side of the ODE for reed dynamics at left BC
def reed_rhs(y, z, p_in, eps, gamma, omega_r, Q_r):
    
    dy = z
    
    dz = (
        omega_r**2 * (eps * (gamma - p_in) - y + 1)
        - (omega_r / Q_r) * z
    )
    
    return dy, dz

def pressure_func(delta_p):
     return jnp.sqrt(jnp.abs(delta_p))*jnp.sign(delta_p)

def l(y):
    return jnp.maximum(y, 0.0) 

def compute_v_bc_left(y, y_t, p_in, zeta, gamma, eps, kappa, omega_r):

    return zeta * l(y) * pressure_func(gamma - p_in) + eps * kappa / omega_r *  y_t