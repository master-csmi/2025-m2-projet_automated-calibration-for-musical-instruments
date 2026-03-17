import jax
import jax.numpy as jnp
from jax import lax

from dg_solver.rhs import dg_rhs_system
from utils.util_func import phi_rhs, reed_rhs, compute_v_bc_left


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     RK2 STEP (PDE + ODEs)
# ------------------------------------------------------------------------------------------------------------------------------

@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, z, gamma, eps, kappa, Q_r, omega_r, zeta,
    S_cells, S_star
):
    # pressures at boundaries
    pR = u_cells[-1, 0, 1]
    pL = u_cells[0, 0, 0]

    # -------------------
    # stage 1
    # -------------------
    k1_phi = phi_rhs(pR, alpha, Z)
    dy1, dz1 = reed_rhs(y, z, pL, eps, gamma, omega_r, Q_r)
    v_bc_1 = compute_v_bc_left(y, z, pL, zeta, gamma, eps, kappa, omega_r)

    k1_u = dg_rhs_system(
        u_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi, beta, Z, alpha,
        v_bc_1,
        S_cells, S_star
    )

    # -------------------
    # midpoint
    # -------------------
    u_mid = u_cells + 0.5 * dt * k1_u
    phi_mid = phi + 0.5 * dt * k1_phi
    y_mid = y + 0.5 * dt * dy1
    z_mid = z + 0.5 * dt * dz1

    pR_mid = u_mid[-1, 0, 1]
    pL_mid = u_mid[0, 0, 0]

    # -------------------
    # stage 2
    # -------------------
    k2_phi = phi_rhs(pR_mid, alpha, Z)
    dy2, dz2 = reed_rhs(y_mid, z_mid, pL_mid, eps, gamma, omega_r, Q_r)
    v_bc_2 = compute_v_bc_left(y_mid, z_mid, pL_mid, zeta, gamma, eps, kappa, omega_r)

    k2_u = dg_rhs_system(
        u_mid, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha,
        v_bc_2,
        S_cells, S_star
    )

    # -------------------
    # update
    # -------------------
    u_new = u_cells + dt * k2_u
    phi_new = phi + dt * k2_phi
    y_new = y + dt * dy2
    z_new = z + dt * dz2

    return u_new, phi_new, y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     EULER STEP
# ------------------------------------------------------------------------------------------------------------------------------

@jax.jit(static_argnames=("bc",))
def euler_step_system(
    u_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, z, gamma, eps, kappa, omega_r, zeta, Q_r,
    S_cells, S_star
):
    pR = u_cells[-1, 0, 1]
    pL = u_cells[0, 0, 0]

    # update phi
    k_phi = phi_rhs(pR, alpha, Z)
    phi_new = phi + dt * k_phi

    # update reed
    dy, dz = reed_rhs(y, z, pL, eps, gamma, omega_r, Q_r)
    y_new = y + dt * dy
    z_new = z + dt * dz

    # compute boundary velocity
    v_bc = compute_v_bc_left(y_new, z_new, pL, zeta, gamma, eps, kappa, omega_r)

    # PDE RHS
    k_u = dg_rhs_system(
        u_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_new, beta, Z, alpha,
        v_bc,
        S_cells, S_star
    )
    u_new = u_cells + dt * k_u

    return u_new, phi_new, y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     RK2 TIME INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------

def time_integrate_rk2(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, beta, Z, alpha,
    y0, z0,
    eps, kappa, gamma, omega_r, Q_r, zeta,
    S_cells, S_star,
    snapshot_steps
):
    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]
    y_snaps = jnp.zeros((nsnaps,))
    u_snaps = jnp.zeros((nsnaps,) + u0.shape)

    def step(carry, n):
        u, phi, y, z, y_snaps, u_snaps = carry
        u_next, phi_next, y_next, z_next = rk2_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z, gamma, eps, kappa, Q_r, omega_r, zeta,
            S_cells, S_star
        )
        mask = snapshot_steps == n  # shape (nsnaps,)
        y_snaps_next = y_snaps + mask* y_next  # broadcasting y_next
        u_snaps_next = u_snaps + mask[:, None, None, None] * u_next
        return (u_next, phi_next, y_next, z_next, y_snaps_next, u_snaps_next), None

    (u_final, phi_final, y_final, z_final, y_snaps, u_snaps), _ = lax.scan(
        step, (u0, phi0, y0, z0, y_snaps, u_snaps), jnp.arange(nsteps)
    )
    return u_final, phi_final, y_final, z_final, y_snaps, u_snaps

# ------------------------------------------------------------------------------------------------------------------------------
#                                                     EULER TIME INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------

def time_integrate_euler(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, beta, Z, alpha,
    y0, z0,
    eps, kappa, gamma, omega_r, Q_r, zeta,
    S_cells, S_star,
    snapshot_steps
):
    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]
    y_snaps = jnp.zeros((nsnaps,))
    u_snaps = jnp.zeros((nsnaps,) + u0.shape)

    def step(carry, n):
        u, phi, y, z, y_snaps, u_snaps = carry
        u_next, phi_next, y_next, z_next = euler_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z, gamma, eps, kappa, omega_r, zeta, Q_r,
            S_cells, S_star
        )
        mask = snapshot_steps == n
        y_snaps_next = y_snaps + mask * y_next
        u_snaps_next = u_snaps + mask[:, None, None, None] * u_next
        return (u_next, phi_next, y_next, z_next, y_snaps_next, u_snaps_next), None

    (u_final, phi_final, y_final, z_final, y_snaps, u_snaps), _ = lax.scan(
        step, (u0, phi0, y0, z0, y_snaps, u_snaps), jnp.arange(nsteps)
    )
    return u_final, phi_final, y_final, z_final, y_snaps, u_snaps