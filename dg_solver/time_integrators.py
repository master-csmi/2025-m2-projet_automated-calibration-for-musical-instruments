import jax
import jax.numpy as jnp
from jax import lax
from bc.bc import phi_rhs,l,F
from dg_solver.rhs import dg_rhs_system

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------

# RK2 for phi ODE at right BC

<<<<<<< HEAD
# RK2 for reed ODE at left BC
@jax.jit(static_argnames=("l_func","F_func"))
def rk2_step_y(y, dy, pL, dt, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    # k1
    k1y  = dy
    k1dy = (1/omega_r**2) * (-y - (1/(Qr*omega_r))*dy + epsilon*(gamma - pL))
    
    # mid step
    y_mid  = y + 0.5*dt*k1y
    dy_mid = dy + 0.5*dt*k1dy
    
    # k2
    k2y  = dy_mid
    k2dy = (1/omega_r**2) * (-y_mid - (1/(Qr*omega_r))*dy_mid + epsilon*(gamma - pL))
    
    # update
    y_new  = y + dt*k2y
    dy_new = dy + dt*k2dy
    
    # incoming velocity using global l() and F()
    v_plus = zeta * l_func(y_new) * F_func(gamma - pL) + epsilon * kappa / omega_r * dy_new
    
    return y_new, dy_new, v_plus
=======
@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_cells, x_nodes, S_cells, c, A, smax, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha
):
    # --- stage 1 ---
    pL = u_cells[-1, 0, 1]          # p^n(L^-)
    k1_phi = phi_rhs(pL, alpha, Z)

    k1_u = dg_rhs_system(
        u_cells, x_nodes, S_cells, c, A,
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
        u_mid, x_nodes, S_cells, c, A,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha
    )

    # --- update ---
    u_new = u_cells + dt * k2_u
    phi_new = phi + dt * k2_phi

    return u_new, phi_new

>>>>>>> 0bea72a3f55d5c06fb96d589d2a96e42c9fd6484

# RK2 time step for system

@jax.jit(static_argnames=("bc","l_func","F_func"))
def rk2_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv,
                    bc, phi, beta, Z, T, alpha,
                    y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    
    # RK2 for phi
    phi_new = rk2_step_phi(u_cells, phi, dt, Z, T, alpha)
    
    # RK2 for reed
    pL = u_cells[0, 0, 0]  # left cell pressure
    y_new, dy_new, v_plus = rk2_step_y(y, dy, pL, dt, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func)
    
    # RHS using ghost cell with v_plus
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv,
                       bc, phi_new, beta, Z, T, alpha, y_new, dy_new,
                       gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt)
    
    u_mid = u_cells + 0.5*dt*k1
    k2 = dg_rhs_system(u_mid, x_nodes, S_cells, c, A, Mp_inv, Mv_inv,
                       bc, phi_new, beta, Z, T, alpha, y_new, dy_new,
                       gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt)
    
    u_new = u_cells + dt*k2
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

#Euler step for reed ODE at left BC
@jax.jit(static_argnames=("l_func","F_func"))
def euler_step_y(y, dy, pL, dt, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    ddy = (1/omega_r**2) * (-1*y - (1/(Qr*omega_r))*dy + epsilon*(gamma - pL))
    dy_new = dy + dt * ddy
    y_new = y + dt * dy
    return y_new, dy_new

# Euler step for system
<<<<<<< HEAD
@jax.jit(static_argnames=("bc","l_func","F_func"))
def euler_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha,
                      y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, Z, T, alpha)
    # Then y and dy
    pL = u_cells[0, 0, 0]  # node at left boundary
    y_new, dy_new = euler_step_y(y, dy, pL, dt, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func)  
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, Z, T, alpha, y_new, dy_new,
                       gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt)  # (N,2,2)
    return u_cells + dt * k1, phi_new, y_new, dy_new
=======
@jax.jit(static_argnames=("bc",))
def euler_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, Z, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, Z, alpha)  # (N,2,2)
    return u_cells + dt * k1, phi_new

>>>>>>> 0bea72a3f55d5c06fb96d589d2a96e42c9fd6484
# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations step
# ------------------------------------------------------------------------------------------------------------------------------
# First integrate 
# RK2 time integration
<<<<<<< HEAD
def time_integrate_rk2(u0, x_nodes, S_cells, c, A, smax, dt, nsteps,
                       Mp_inv, Mv_inv, bc, phi0, beta, Z, T, alpha,
                       y0, dy0, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    
    def step(carry, _):
        u, phi, y, dy = carry
        u_next, phi_next, y_next, dy_next = rk2_step_system(
            u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv,
            bc, phi, beta, Z, T, alpha,
            y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func
        )
        return (u_next, phi_next, y_next, dy_next), None
    
    (u_final, phi_final, y_final, dy_final), _ = lax.scan(step, (u0, phi0, y0, dy0), None, length=nsteps)
    return u_final, phi_final, y_final, dy_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, T, alpha,
                         y0, dy0, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    def step_sys(carry, _):
        u, phi, y, dy = carry
        u_next, phi_next, y_next, dy_next = euler_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha,
                                                             y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func)
        return (u_next, phi_next, y_next, dy_next), None
=======
def time_integrate_rk2(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, alpha):
    def step(carry, _):
        u, phi = carry
        u_next, phi_next = rk2_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha)
        return (u_next, phi_next), None
    (u_final, phi_final), _ = lax.scan(step, (u0, phi0), None, length=nsteps)
    return u_final, phi_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, Z, alpha):
    def step_sys(carry, _):
        u, phi = carry
        u_next,phi_next = euler_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha)
        return (u_next, phi_next), None
>>>>>>> 0bea72a3f55d5c06fb96d589d2a96e42c9fd6484
    
    (u_final, phi_final, y_final, dy_final), _ = lax.scan(step_sys, (u0, phi0, y0, dy0), None, length=nsteps)

    return u_final, phi_final, y_final, dy_final