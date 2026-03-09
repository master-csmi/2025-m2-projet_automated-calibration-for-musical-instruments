import jax.numpy as jnp
from dataclasses import dataclass

@dataclass(frozen=True)
class BC:
    type: str
    left: tuple

def apply_bc_right_impedance(u_cells, phi, beta, Z, alpha):
        # values inside the domain at right boundary
        pR = u_cells[-1,0,1]
        vR = u_cells[-1,1,1]

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

        ghost_R = jnp.stack([
            jnp.array([p_ext, p_ext]),
            jnp.array([v_ext, v_ext])
        ])
        return ghost_R

def apply_bc(u_cells, bc_left, phi, beta, Z, alpha):
    # u_cells: (N, 2, 2)
    ghost_L = jnp.stack([
        jnp.array([bc_left[0], bc_left[0]]),
        jnp.array([bc_left[1], bc_left[1]])
    ])

    ghost_R = apply_bc_right_impedance(u_cells, phi, beta, Z, alpha)

    return jnp.concatenate([ghost_L[None, ...], u_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)


# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
def apply_bc_neumann(u_cells):
    ghost_L = u_cells[0]
    ghost_R = u_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_cells, ghost_R[None, ...]],
        axis=0
    )
