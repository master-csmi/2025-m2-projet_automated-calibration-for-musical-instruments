import jax
import jax.numpy as jnp
from jax import lax
from bc.bc import phi_rhs
from dg_solver.rhs import dg_rhs_system
from bc.bc import apply_bc_left_reed, apply_bc_right_impedance, apply_bc

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------



@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_cells, x_nodes, c, smax, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, dy,                
    gamma, epsilon, kappa,
    f_r, Qr, zeta
):
    # --- stage 1 ---
    pR = u_cells[-1, 0, 1]          # p^n(L^-)
    k1_phi = phi_rhs(pR, alpha, Z)

    u_ext, k1_y, k1_dy = apply_bc(
    u_cells,phi, beta, Z, alpha,
    y, dy, gamma, epsilon, kappa, 
    f_r, Qr, zeta
    )
    

    k1_u = dg_rhs_system(
        u_ext, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi, beta, Z, alpha,
        y, dy, gamma, epsilon, kappa, f_r, Qr, zeta

    )

    # --- midpoint ---
    u_mid = u_cells + 0.5 * dt * k1_u
    phi_mid = phi + 0.5 * dt * k1_phi
    y_mid = y + 0.5 * dt * k1_y
    dy_mid = dy + 0.5 * dt * k1_dy

    # --- stage 2 ---
    pR_mid = u_mid[-1, 0, 1]        # p^{n+1/2}(L^-)
    k2_phi = phi_rhs(pR_mid, alpha, Z)

    u_ext_mid, k2_y, k2_dy = apply_bc(
    u_mid, phi_mid, beta, Z, alpha,
    y_mid, dy_mid, gamma, epsilon,
    kappa, f_r, Qr, zeta
    )

    k2_u = dg_rhs_system(
        u_ext_mid, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha,
        y_mid, dy_mid, gamma, epsilon, kappa, f_r, Qr, zeta
    )

    # --- update ---
    u_new = u_cells + dt * k2_u
    phi_new = phi + dt * k2_phi
    y_new = y + dt * k2_y
    dy_new = dy + dt * k2_dy

    return u_new, phi_new, y_new, dy_new


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
def euler_step_system(u_cells, x_nodes, c, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha, y, dy, gamma, epsilon, kappa, f_r, Qr, zeta):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, Z, alpha)
    # Then RHS
    u_ext, y_new, dy_new = apply_bc(u_cells, phi_new, beta, Z, alpha, y, dy, gamma, epsilon, kappa, f_r, Qr, zeta)
    k1 = dg_rhs_system(u_ext, x_nodes, c, Mp_inv, Mv_inv, bc, phi_new, beta, Z, alpha, y_new, dy_new, gamma, epsilon, kappa, f_r, Qr, zeta)  # (N+2*ghost_cells_size,N+2*ghost_cells_size)
    return u_cells + dt * k1, phi_new, y_new, dy_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations step
# ------------------------------------------------------------------------------------------------------------------------------
# First integrate 
# RK2 time integration

def time_integrate_rk2(u0, x_nodes, c, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, alpha, y0, dy0, gamma, epsilon, kappa, f_r, Qr, zeta):
    def step(carry, _):
        u, phi, y, dy = carry
        u_next, phi_next, y_next, dy_next = rk2_step_system(u, x_nodes, c, smax, dt, Mp_inv,Mv_inv,bc,
                                           phi,beta,Z,alpha,y,dy,gamma,epsilon,kappa,f_r,Qr,zeta)
        return (u_next, phi_next,y_next,dy_next), None
    (u_final, phi_final,y_final,dy_final), _ = lax.scan(step, (u0, phi0, y0, dy0), None, length=nsteps)
    return u_final, phi_final, y_final, dy_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, c, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, alpha, y0, dy0, gamma, epsilon, kappa, f_r, Qr, zeta):
    def step_sys(carry, _):
        u, phi, y, dy = carry
        u_next,phi_next,y_next,dy_next = euler_step_system(u, x_nodes, c, smax, dt, Mp_inv,Mv_inv,bc,
                                            phi,beta,Z,alpha,y,dy,gamma,epsilon,kappa,f_r,Qr,zeta)
        return (u_next, phi_next,y_next,dy_next), None
    
    (u_final, phi_final,y_final,dy_final), _ = lax.scan(step_sys, (u0, phi0, y0, dy0), None, length=nsteps)

    return u_final, phi_final, y_final, dy_final