import jax.numpy as jnp
from dataclasses import dataclass
from src.utils.util_func import ReedOpening 
from src.utils.util_func import pressure_func as F_func
l=ReedOpening(a=1.0)  # fonction d'ouverture de l'anche, à calibrer
@dataclass(frozen=True)
class BC:
    type: str
    
    
def apply_bc_right_impedance(
    u_tilde_cells, phi, beta, Z, alpha,
    S_cells, S_star, c
):
    S_R = S_cells[-1]

    # état intérieur
    p_tilde_R = u_tilde_cells[-1, 0, 1]
    v_tilde_R = u_tilde_cells[-1, 1, 1]

    # coefficients
    f = S_R / S_star
    K = (beta / Z) * (S_star / S_R)**2
    C = (S_star / (c * S_R)) * jnp.sqrt(alpha)

    # invariant sortant
    w_plus = v_tilde_R + f * p_tilde_R

    # coefficient combiné
    ratio = K / f

    # invariant entrant
    w_minus = ((ratio - 1) / (ratio + 1)) * w_plus \
              - (2 * C / (ratio + 1)) * phi

    # reconstruction
    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = (w_plus - w_minus) / (2 * f)

    ghost_R = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])

    return ghost_R

def apply_bc_left_dynamic(u_cells, S_cells, c, S_star,v_bc_tilde,
                           zeta, gamma, eps, kappa, omega_r,y, dt_y):
    #S_L   = S_cells[0]
    #alpha = S_L / S_star        # coefficient d'impédance réduite
    #factor = alpha                       # à corriger si définition différente
#
    #p_tilde_L = u_cells[0, 0, 0]
    #v_tilde_L = u_cells[0, 1, 0]
#
    ## invariant sortant (intérieur)
    #w_minus = v_tilde_L - alpha * p_tilde_L
#
    ## point fixe sur w+
    #w_plus = w_minus  # initial guess
#
    #for _ in range(10):      # converge en 2-3 itérations
    #    p_interface = (w_plus - w_minus) / (2.0 * factor)
    #    arg_F       = gamma - (c * S_star / S_L) * p_interface * factor
    #    F_val       = F_func(arg_F)
    #    w_plus      = w_minus + 2 * alpha * (
    #                      zeta*l(y) * F_val
    #                    + eps * (kappa / omega_r) * alpha * dt_y
    #                  )
#
    ## reconstruction état fantôme
    #v_tilde_ext = 0.5 * (w_plus + w_minus)
    #p_tilde_ext = (w_plus - w_minus) / (2.0 * factor)
#
    #ghost_L = jnp.stack([
    #    jnp.array([p_tilde_ext, p_tilde_ext]),
    #    jnp.array([v_tilde_ext, v_tilde_ext])
    #])

    
    S_L = S_cells[0]
    f = S_L / S_star

    p_tilde_L = u_cells[0, 0, 0]
    v_tilde_L = u_cells[0, 1, 0]
    

    # Neumann exact : ghost symétrique en p, antisymétrique en v
    # ce qui impose v_interface = v_tilde_bc exactement
    w_minus = v_tilde_L - f * p_tilde_L
    w_plus  = 2.0 * v_bc_tilde - w_minus

    v_tilde_ext = 0.5 * (w_plus + w_minus)
    p_tilde_ext = 0.5 * (w_plus - w_minus) / f

    
    

    #S_L = S_cells[0]
#
    #p_tilde_L = u_cells[0, 0, 0]
    #v_tilde_L = u_cells[0, 1, 0]
#
    ## facteur section
    #factor = (S_L / (c * S_star))**2
#
    ## invariant sortant
    #w_minus = v_tilde_L - factor * p_tilde_L
#
    ## invariant entrant imposé par v_bc
    #v_tilde_ext = (S_star / (c * S_L)) * v_bc
#
    ## invariant entrant par méthode itérative (point fixe)
    #w_plus = w_minus.copy()  # initial guess
#
    #for _ in range(3):  # 2-3 itérations suffisent pour dt petit
    #    w_plus = w_minus + 2 * v_bc
#
    ## reconstruction
    #v_tilde_ext = 0.5 * (w_plus + w_minus)
    #p_tilde_ext = (w_plus - w_minus) / (2.0 * factor)

    ghost_L = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])
    return ghost_L



def apply_bc(u_tilde_cells, phi, beta,v_bc_tilde, Z, alpha, S_cells, c, S_star, zeta, gamma, eps, kappa, omega_r, y, dt_y):
    # u_tilde_cells: (N, 2, 2)
    ghost_L = apply_bc_left_dynamic(u_tilde_cells, S_cells, c, S_star,v_bc_tilde, zeta, gamma, eps, kappa, omega_r, y , dt_y)

    ghost_R = apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c)

    return jnp.concatenate([ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)



def apply_bc_left_dynamic_infinite_pipe(u_cells, S_cells, c, S_star,v_bc_tilde,
                           zeta, gamma, eps, kappa, omega_r,y, dt_y):
    
    S_L = S_cells[0]
    f = S_L / S_star

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

