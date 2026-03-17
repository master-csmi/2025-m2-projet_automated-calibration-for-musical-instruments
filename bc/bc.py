import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class BC:
    type: str
    
    
def apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c):
        
        SR=S_cells[-1]
        # values inside the domain at right boundary
        p_tilde_R = u_tilde_cells[-1,0,1]
        v_tilde_R = u_tilde_cells[-1,1,1]

        # reconstruct physical variables at right boundary
        pR = (c*S_star/SR) * p_tilde_R
        vR = (SR/(c*S_star)) * v_tilde_R

        # outgoing wave
        w_plus = pR + vR

        # reflection coefficient
        a = (1.0 - beta / Z) / (1.0 + beta / Z)
        b= 2.0 * jnp.sqrt(alpha) / (1.0 + beta / Z)

        # incoming wave from ODE
        w_minus = a * w_plus + b * phi

        # reconstruct p and v
        p_ext = 0.5 * (w_plus + w_minus)
        v_ext = 0.5 * (w_plus - w_minus)

        # conversion to tilde variables for ghost cell
        p_tilde_ext = (SR/(c*S_star)) * p_ext
        v_tilde_ext = (c*S_star/SR) * v_ext

        ghost_R = jnp.stack([
            jnp.array([p_tilde_ext, p_tilde_ext]),
            jnp.array([v_tilde_ext, v_tilde_ext])
        ])
        return ghost_R

def apply_bc_left_dynamic(u_cells, v_bc, S_cells, c, S_star):

    # section at first cell
    S_L = S_cells[0]

    # interior tilde variables at left boundary node
    p_tilde_L = u_cells[0,0,0]
    v_tilde_L = u_cells[0,1,0]

    # convert tilde -> physical
    pL = (c * S_star / S_L) * p_tilde_L
    vL = (S_L / (c * S_star)) * v_tilde_L

    # outgoing characteristic
    w_minus = pL - vL

    # impose velocity
    w_plus = pL + v_bc

    # reconstruct physical ghost state
    p_ext = 0.5 * (w_plus + w_minus)
    v_ext = 0.5 * (w_plus - w_minus)

    # convert back to tilde
    p_tilde_ext = (S_L / (c * S_star)) * p_ext
    v_tilde_ext = (c * S_star / S_L) * v_ext

    ghost_L = jnp.stack([
        jnp.array([p_tilde_ext, p_tilde_ext]),
        jnp.array([v_tilde_ext, v_tilde_ext])
    ])

    return ghost_L

def apply_bc(u_tilde_cells, phi, beta, Z, alpha, v_bc, S_cells, c, S_star):
    # u_tilde_cells: (N, 2, 2)
    ghost_L = apply_bc_left_dynamic(u_tilde_cells, v_bc, S_cells, c, S_star)

    ghost_R = apply_bc_right_impedance(u_tilde_cells, phi, beta, Z, alpha, S_cells, S_star, c)

    return jnp.concatenate([ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)


# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
def apply_bc_neumann(u_tilde_cells):
    ghost_L = u_tilde_cells[0]
    ghost_R = u_tilde_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_tilde_cells, ghost_R[None, ...]],
        axis=0
    )
