import jax
import jax.numpy as jnp
from jax import lax
from bc.bc import phi_rhs
from dg_solver.rhs import dg_rhs_system

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------

# RK2 for phi ODE at right BC
@jax.jit
def rk2_step_phi(u_cells, phi, dt, Z, T, alpha):
    pL = u_cells[-1, 0, 1]  # p^-
    k1 = phi_rhs(pL, alpha, Z, T)
    phi_mid = phi + 0.5 * dt * k1
    k2 = phi_rhs(pL, alpha, Z, T)
    return phi + dt * k2

# RK2 time step for system
@jax.jit(static_argnames=("bc",))
def rk2_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha):
    # First phi
    phi_new = rk2_step_phi(u_cells, phi, dt, Z, T, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, Z, T, alpha)
    u_mid = u_cells + 0.5 * dt * k1
    k2 = dg_rhs_system(u_mid, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, Z, T, alpha)
    return u_cells + dt * k2, phi_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           Euler step
# ------------------------------------------------------------------------------------------------------------------------------

#Euler step for phi ODE at right BC
@jax.jit
def euler_step_phi(u_cells, phi, dt, Z, T, alpha):
    pL = u_cells[-1, 0, 1]  # p^-
    k1 = phi_rhs(pL, alpha, Z, T)
    return phi + dt * k1

# Euler step for system
@jax.jit(static_argnames=("bc",))
def euler_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, Z, T, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, Z, T, alpha)  # (N,2,2)
    return u_cells + dt * k1, phi_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations stef
# ------------------------------------------------------------------------------------------------------------------------------
# Fist integrate 
# RK2 time integration
def time_integrate_rk2(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, T, alpha):
    def step(carry, _):
        u, phi = carry
        u_next, phi_next = rk2_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha)
        return (u_next, phi_next), None
    (u_final, phi_final), _ = lax.scan(step, (u0, phi0), None, length=nsteps)
    return u_final, phi_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, T, alpha):
    def step_sys(carry, _):
        u, phi = carry
        u_next,phi_next = euler_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha)
        return (u_next, phi_next), None
    
    (u_final, phi_final), _ = lax.scan(step_sys, (u0, phi0), None, length=nsteps)

    return u_final, phi_final