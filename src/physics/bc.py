import jax.numpy as jnp
from dataclasses import dataclass
from src.utils.util_func import ReedOpening 
from src.inverse.grad_pt_fixe import solve_wplus_newton as solve_wplus_fixed_point

l=ReedOpening()  # fonction d'ouverture de l'anche, à calibrer
@dataclass(frozen=True)
class BC:
    type: str

    
    
def apply_bc_right_impedance(
    u_tilde_cells, phi, beta, Z, alpha,
    S_cells, S_star, c
):

    S_R = S_cells[-1]

    p_tilde_R = u_tilde_cells[-1, 0, 1]
    v_tilde_R = u_tilde_cells[-1, 1, 1]

    f = S_star / S_R

    K = (beta / Z) * (S_star / S_R)**2
    C = (S_star / (c * S_R)) * jnp.sqrt(alpha)

    w_plus = v_tilde_R + f * p_tilde_R

    ratio = K / f

    w_minus = ((ratio - 1.0) / (ratio + 1.0)) * w_plus \
              - (2.0 * C / (ratio + 1.0)) * phi

    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = (w_plus - w_minus) / (2.0 * f)

    ghost_R = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])

    return ghost_R

def apply_bc_left_dynamic(u_cells, S_cells, c, S_star,v_bc_tilde,
                           zeta, gamma, eps, kappa, omega_r,y, dt_y):
    # cette version entraine une approximation de l'invariant entrant
    # la version fixed itère selon la méthode présentée dans le rapport   
    S_L = S_cells[0]
    f = S_star / S_L

    p_tilde_L = u_cells[0, 0, 0]
    v_tilde_L = u_cells[0, 1, 0]
    

    # Neumann exact : ghost symétrique en p, antisymétrique en v
    # ce qui impose v_interface = v_tilde_bc exactement
    w_minus = v_tilde_L - f * p_tilde_L
    w_plus  = 2.0 * v_bc_tilde - w_minus

    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = 0.5 * (w_plus - w_minus) / f


    ghost_L = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])

    return ghost_L


def apply_bc_left_dynamic_fixed_implicit(
    u_cells, S_cells, c, S_star,
    zeta, gamma, eps, kappa, omega_r,
    y, y_t, opening
):
    S_L = S_cells[0]
    k_char = S_star / S_L

    p_tilde_L = u_cells[0, 0, 0]
    v_tilde_L = u_cells[0, 1, 0]

    w_minus = v_tilde_L - k_char * p_tilde_L
    w0 = v_tilde_L + k_char * p_tilde_L

    ell_y = opening(y)

    w_plus = solve_wplus_fixed_point(
        w0, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r, ell_y, y_t
    )

    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = (w_plus - w_minus) / (2.0 * k_char)

    ghost_L = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])

    return ghost_L



def apply_bc(u_tilde_cells, phi, beta,v_bc_tilde, Z, alpha, S_cells, c, S_star, zeta, gamma, eps, kappa, omega_r, y, dt_y):

    " Fonction d'application des conditions aux limites, avec dynamique d'anche à gauche "
    # u_tilde_cells: (N, 2, 2)
    ghost_L = apply_bc_left_dynamic(u_tilde_cells, S_cells, c, S_star,v_bc_tilde, zeta, gamma, eps, kappa, omega_r, y , dt_y)

    ghost_R = apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c)

    return jnp.concatenate([ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)

def apply_bc_fixed(u_tilde_cells, phi, beta, Z, alpha, S_cells, c, S_star, zeta, gamma, eps, kappa, omega_r, y, dt_y,opening):

    " Fonction d'application des conditions aux limites, avec itération fixe à gauche pour la condition dynamique "

    # u_tilde_cells: (N, 2, 2)
    ghost_L = apply_bc_left_dynamic_fixed_implicit(
        u_tilde_cells, S_cells, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        y, dt_y, opening
    )

    ghost_R = apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c)

    return jnp.concatenate([ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)


# Uniquement pour test de convergence, pas de dynamique d'anche
def apply_bc_left_dynamic_infinite_pipe(u_cells, S_cells, c, S_star,v_bc_tilde,
                           zeta, gamma, eps, kappa, omega_r,y, dt_y):
    
    S_L = S_cells[0]
    f = S_star / S_L

    p_tilde_L = u_cells[0, 0, 0]
    v_tilde_L = u_cells[0, 1, 0]
    

    # Neumann exact : ghost symétrique en p, antisymétrique en v
    # ce qui impose v_interface = v_tilde_bc exactement
    w_minus = v_tilde_L - f * p_tilde_L
    w_plus  = 0.0

    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = 0.5 * (w_plus - w_minus) / f


    ghost_L = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])
    return ghost_L


def apply_bc_test(u_tilde_cells, phi, beta,v_bc_tilde, Z, alpha, S_cells, c, S_star, zeta, gamma, eps, kappa, omega_r, y, dt_y):
    # u_tilde_cells: (N, 2, 2)
    ghost_L = apply_bc_left_dynamic_infinite_pipe(u_tilde_cells, S_cells, c, S_star,v_bc_tilde, zeta, gamma, eps, kappa, omega_r, y , dt_y)

    ghost_R = apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c)

    return jnp.concatenate([ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)

