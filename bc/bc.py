import jax.numpy as jnp
from dataclasses import dataclass
import jax

# Right Hand Side of the ODE for phi at right BC
def phi_rhs(pR, alpha, Z, T):
    return -jnp.sqrt(alpha)/ (Z * T) * pR

def F(delta_p):
    return jnp.sign(delta_p) * jnp.sqrt(jnp.abs(delta_p))

def l(y):
    return y  # simple linear opening


@dataclass(frozen=True)
class BC:
    """Boundary conditions"""
    type: str
    left: tuple    # (p, v) values at left boundary
    right: tuple   # (parameters for right boundary)

def apply_bc_right_impedance(u_cells, phi, beta, Z,T, alpha):
        # values inside the domain at right boundary
        pR = u_cells[-1,0,1]
        vR = u_cells[-1,1,1]

        # outgoing wave
        w_plus = pR + vR

        # reflection coefficient
        r = (1.0 - beta / (Z*T)) / (1.0 + beta / (Z*T))

        # incoming wave from ODE
        w_minus = r * w_plus + (2.0 * jnp.sqrt(alpha) / (1.0 + beta / (Z*T))) * phi

        # reconstruct p and v
        p_ext = 0.5 * (w_plus + w_minus)
        v_ext = 0.5 * (w_plus - w_minus)

        ghost_R = jnp.stack([
            jnp.array([p_ext, p_ext]),
            jnp.array([v_ext, v_ext])
        ])
        return ghost_R

def apply_bc_left_reed(u_cells, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt):
    # Pressure at left cell
    p0 = u_cells[0,0,0]

    # Integrate reed ODE (second-order ODE) explicitly
    ddy = omega_r**2 * (epsilon*(gamma - p0) - (y - 1) - (1/(Qr*omega_r))*dy)
    dy_new = dy + dt * ddy
    y_new = y + dt * dy_new

    # Compute left incoming velocity
    v_plus = zeta * l_func(y_new) * F_func(gamma - p0) + epsilon * kappa / omega_r * dy_new

    # Construct ghost state at left boundary
    ghost_L = jnp.stack([
        jnp.array([p0, p0]),
        jnp.array([v_plus, v_plus])
    ])
    
    return ghost_L, y_new, dy_new


def apply_bc(u_cells, bc_left, phi, beta, Z, T, alpha, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt):
    ghost_L, y_new, dy_new = apply_bc_left_reed(u_cells, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt)
    ghost_R = apply_bc_right_impedance(u_cells, phi, beta, Z, T, alpha)
    return jnp.concatenate([ghost_L[None, ...], u_cells, ghost_R[None, ...]], axis=0), y_new, dy_new



# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
def apply_bc_neumann(u_cells):
    ghost_L = u_cells[0]
    ghost_R = u_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_cells, ghost_R[None, ...]],
        axis=0
    )
