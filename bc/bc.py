import jax.numpy as jnp
from dataclasses import dataclass
from utils.util_function import phi_rhs, pressure_func, l


@dataclass(frozen=True)
class BC:
    type: str
    left: tuple
    right: tuple

#============================Right BC: impedance with ODE for phi===============================
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


def apply_bc_left_reed(
    u_cells,
    y, dy,
    gamma, epsilon,
    kappa, f_r, Qr, zeta
):
    p0 = u_cells[0, 0, 0]
    omega_r = 2 * jnp.pi * f_r

    # ===== RHS reed =====
    ddy = omega_r**2 * (
        epsilon*(gamma - p0)
        - (y - 1.0)
        - (1/(Qr*omega_r))*dy
    )

    dy_rhs = dy   

    v_plus = (
        zeta * l(y) * pressure_func(gamma - p0)
        + epsilon * kappa / omega_r * dy
    )

    ghost_L = jnp.stack([
        jnp.array([p0, p0]),
        jnp.array([v_plus, v_plus])
    ])

    return ghost_L, dy_rhs, ddy


#============================Apply BCs to extend u_cells with ghost cells===============================
def apply_bc(u_cells, phi, beta, Z, alpha, y, dy, gamma, epsilon, kappa, f_r, Qr, zeta):

    ghost_L, dy_rhs, ddy_rhs = apply_bc_left_reed(u_cells, y, dy, gamma, epsilon, kappa, f_r, Qr, zeta)

    ghost_R = apply_bc_right_impedance(u_cells, phi, beta, Z, alpha)

    return jnp.concatenate([ghost_L[None, ...], u_cells, ghost_R[None, ...]], axis=0), dy_rhs, ddy_rhs
    

# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
def apply_bc_neumann(u_cells):
    ghost_L = u_cells[0]
    ghost_R = u_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_cells, ghost_R[None, ...]],
        axis=0
    )



