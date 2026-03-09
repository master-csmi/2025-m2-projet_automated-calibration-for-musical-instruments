import jax
import jax.numpy as jnp
from jax import lax
from utils.util_func import phi_rhs, compute_v_bc_left
from dg_solver.rhs import dg_rhs_system


# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------

# RK2 for phi ODE at right BC

@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, z, gamma, eps, kappa, Q_r, omega_r, zeta,
    S_cells, S_star
):
    # --- stage 1 ---
    pL = u_cells[-1, 0, 1]  # p^n(L^-)
    p0 = u_cells[0, 0, 0]  # p^n(0^+)     
    k1_phi = phi_rhs(pL, alpha, Z)

    # Compute v_bc at left boundary for stage 1
    v_bc_stage_1 = compute_v_bc_left(y, z, p0, zeta, gamma, eps, kappa, omega_r) 

    k1_u = dg_rhs_system(
        u_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi, beta, Z, alpha,v_bc_stage_1, S_cells, S_star
    )

    k1_y = z
    k1_z = -(omega_r/Q_r) * z - (omega_r**2) * (y-1) - eps * omega_r**2 * (gamma - p0)


    # --- midpoint ---
    u_mid = u_cells + 0.5 * dt * k1_u
    phi_mid = phi + 0.5 * dt * k1_phi
    y_mid = y + 0.5 * dt * k1_y
    z_mid = z + 0.5 * dt * k1_z

    # --- stage 2 ---
    pL_mid = u_mid[-1, 0, 1]        # p^{n+1/2}(L^-)
    p0_mid = u_mid[0, 0, 0]        # p^{n+1/2}(0^+)

    v_bc_mid = compute_v_bc_left(y_mid, z_mid, p0_mid, zeta, gamma, eps, kappa, omega_r)
    k2_phi = phi_rhs(pL_mid, alpha, Z)

    k2_u = dg_rhs_system(
        u_mid, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha, v_bc_mid, S_cells, S_star
    )

    k2_y = z_mid
    k2_z = -(omega_r/Q_r) * z_mid - (omega_r**2) * (y_mid-1) - eps * omega_r**2 * (gamma - p0_mid)

    # --- update ---
    u_new = u_cells + dt * k2_u
    phi_new = phi + dt * k2_phi
    y_new = y + dt * k2_y
    z_new = z + dt * k2_z

    return u_new, phi_new, y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                           Euler step
# ------------------------------------------------------------------------------------------------------------------------------

#Euler step for phi ODE at right BC
#\phi_n+1 = \phi_n + dt*f(phi_n) 
# where f is given by phi_rhs


# Euler step for system
@jax.jit(static_argnames=("bc",))
def euler_step_system(u_cells, x_nodes, c, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha, y, z, gamma, eps, kappa, omega_r, zeta, Q_r, S_cells, S_star):
    # First phi
    pL = u_cells[-1, 0, 1]  # p^-
    k1_phi = phi_rhs(pL, alpha, Z)
    

    # Then compute v_bc at left boundary for Euler step
    p0 = u_cells[0, 0, 0]  # p(0^+)
    
    k1_y = z
    k1_z = -(omega_r/Q_r) * z - (omega_r**2) * (y-1) - eps * omega_r**2 * (gamma - p0)

    v_bc = compute_v_bc_left(y, z, p0, zeta, gamma, eps, kappa, omega_r)
    k1 = dg_rhs_system(u_cells, x_nodes, c, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha, v_bc=v_bc, S_cells=S_cells, S_star=S_star)  # (N,2,2)
    y_new = y + dt * k1_y
    z_new = z + dt * k1_z
    phi_new = phi + dt * k1_phi
    # Then RHS
    
    return u_cells + dt * k1, phi_new, y_new, z_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations step
# ------------------------------------------------------------------------------------------------------------------------------
# First integrate 
# RK2 time integration
def time_integrate_rk2(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, beta, Z, alpha,
    y, z, gamma, eps, kappa, omega_r, zeta, Q_r,
    S_cells, S_star,
    snapshot_steps
):

    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]

    # stockage snapshots
    u_snaps = jnp.zeros((nsnaps,) + u0.shape)

    def step(carry, n):

        u, phi, y, z, snaps = carry

        u_next, phi_next, y_next, z_next = rk2_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z, gamma, eps, kappa,
            Q_r, omega_r, zeta,
            S_cells, S_star
        )

        # vérifier si snapshot
        mask = snapshot_steps == n

        snaps = snaps + mask[:, None, None, None] * u_next

        return (u_next, phi_next, y_next, z_next, snaps), None


    (u_final, phi_final, y_final, z_final, snaps), _ = lax.scan(
        step,
        (u0, phi0, y, z, u_snaps),
        jnp.arange(nsteps)
    )

    return u_final, phi_final, y_final, z_final, snaps

# Euler time integration
def time_integrate_euler(u0, x_nodes, c, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, alpha, y, z, gamma, eps, kappa, omega_r, zeta, Q_r, S_cells, S_star):
    def step_sys(carry, _):
        u, phi, y, z = carry
        u_next,phi_next,y_next,z_next = euler_step_system(u, x_nodes, c, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha,y=y,z=z,gamma=gamma,
                                            eps=eps,kappa=kappa,
                                            omega_r=omega_r,zeta=zeta,Q_r=Q_r, S_cells=S_cells, S_star=S_star)  # (N,2,2)
        return (u_next, phi_next,y_next,z_next), None
    
    (u_final, phi_final,y_final,z_final), _ = lax.scan(step_sys, (u0, phi0,y,z), None, length=nsteps)

    return u_final, phi_final, y_final, z_final