import jax
import jax.numpy as jnp
from jax import lax
from dg_solver.rhs import dg_rhs_system
from utils.util_func import phi_rhs, compute_v_bc_left

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------

# RK2 for phi ODE at right BC

@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha
):
    # --- stage 1 ---
    pL = u_cells[-1, 0, 1]          # p^n(L^-)
    k1_phi = phi_rhs(pL, alpha, Z)

    k1_u = dg_rhs_system(
        u_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi, beta, Z, alpha
    )

    # --- midpoint ---
    u_mid = u_cells + 0.5 * dt * k1_u
    phi_mid = phi + 0.5 * dt * k1_phi

    # --- stage 2 ---
    pL_mid = u_mid[-1, 0, 1]        # p^{n+1/2}(L^-)
    k2_phi = phi_rhs(pL_mid, alpha, Z)

    k2_u = dg_rhs_system(
        u_mid, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha
    )

    # --- update ---
    u_new = u_cells + dt * k2_u
    phi_new = phi + dt * k2_phi

    return u_new, phi_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                           Euler step
# ------------------------------------------------------------------------------------------------------------------------------

#Euler step for phi ODE at right BC
#\phi_n+1 = \phi_n + dt*f(phi_n) 
# where f is given by phi_rhs
@jax.jit
def euler_step_phi(u_cells, phi, dt, Z, alpha):
    pL = u_cells[-1, 0, 1]  # p^-
    k1 = phi_rhs(pL, alpha, Z)
    return phi + dt * k1 

# Euler step for system
@jax.jit(static_argnames=("bc",))
def euler_step_system(u_cells, x_nodes, c, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, Z, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, c, Mp_inv, Mv_inv, bc, phi_new, beta, Z, alpha)  # (N,2,2)
    return u_cells + dt * k1, phi_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations step
# ------------------------------------------------------------------------------------------------------------------------------
# First integrate 
# RK2 time integration
# RK2 time integration
def time_integrate_rk2(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, beta, Z, alpha,
    snapshot_steps
):

    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]

    # stockage snapshots
    u_snaps = jnp.zeros((nsnaps,) + u0.shape)

    def step(carry, n):

        u, phi, snaps = carry

        u_next, phi_next = rk2_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha
        )

        # vérifier si snapshot
        mask = snapshot_steps == n

        snaps = snaps + mask[:, None, None, None] * u_next

        return (u_next, phi_next, snaps), None


    (u_final, phi_final, snaps), _ = lax.scan(
        step,
        (u0, phi0, u_snaps),
        jnp.arange(nsteps)
    )

    return u_final, phi_final, snaps


# Euler time integration
def time_integrate_euler(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, beta, Z, alpha,
    snapshot_steps
):

    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]

    # stockage snapshots
    u_snaps = jnp.zeros((nsnaps,) + u0.shape)

    def step(carry, n):

        u, phi, snaps = carry

        u_next, phi_next = euler_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha
        )

        # vérifier si snapshot
        mask = snapshot_steps == n

        snaps = snaps + mask[:, None, None, None] * u_next

        return (u_next, phi_next, snaps), None

    (u_final, phi_final, snaps), _ = lax.scan(
        step,
        (u0, phi0, u_snaps),
        jnp.arange(nsteps)
    )

    return u_final, phi_final, snaps