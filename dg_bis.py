# DG P1 for 1D linear wave system (p,v) with p0 = Gaussian, v0 = 0
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time 
import csv
import os
import pandas as pd
import numpy as np
import argparse
from jax import lax
from dataclasses import dataclass

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          mesh / basis with ghost cells
# ------------------------------------------------------------------------------------------------------------------------------
def create_uniform_nodes_with_ghosts(N_intervals, x_min=0.0, x_max=1.0):
    dx = (x_max - x_min) / N_intervals
    
    # physical nodes
    x_nodes = jnp.linspace(x_min, x_max, N_intervals + 1)
    
    # ghost nodes nodes
    left_ghost  = jnp.array([x_nodes[0] - dx])
    right_ghost = jnp.array([x_nodes[-1] + dx])
    ghost_cells = jnp.concatenate([left_ghost, right_ghost])
    
    
    return x_nodes, ghost_cells

def cell_edges_from_nodes(x_nodes):
    return x_nodes[:-1], x_nodes[1:]

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          basis functions P1
# ------------------------------------------------------------------------------------------------------------------------------
def phi_at(x, xL, xR):
    h = xR - xL
    xi = 2.0 * (x - xL) / h - 1.0
    phi0 = 0.5 * (1.0 - xi)
    phi1 = 0.5 * (1.0 + xi)
    return jnp.stack([phi0, phi1])

vphi_at = jax.vmap(phi_at, in_axes=(0, None, None))

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          Fucntion of the instrument section S(x)
# ------------------------------------------------------------------------------------------------------------------------------
def S_of_x(x):
    # Example: constant cross-section
    return jnp.ones_like(x) 

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          analytic local mass matrix for P1
# ------------------------------------------------------------------------------------------------------------------------------

def local_mass_inv_system(h, c=1.0):
    Mloc = (h / 6.0) * jnp.array([[2., 1.],
                                 [1., 2.]])
    Minv = jnp.linalg.inv((1.0 / c) * Mloc)
    return Minv, Minv


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          flux for system (linear)
# ------------------------------------------------------------------------------------------------------------------------------
def linear_system_flux(A):
    def Flux(U):
        # U shape (...,2)
        return U @ A.T 
    return Flux

def rusanov_flux(U_L, U_R, A,smax):
   
    F_L = (U_L[None, :] @ A.T)[0]
    F_R = (U_R[None, :] @ A.T)[0]

    return 0.5*(F_L + F_R) - 0.5*smax*(U_R - U_L)

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          local volume term for system
# ------------------------------------------------------------------------------------------------------------------------------
def local_volume_system(u_cell, xL, xR, A, nq=24):
    h = xR - xL
    xq = jnp.linspace(xL, xR, nq)
    w  = jnp.ones(nq) * (h/(nq-1))
    w  = w.at[0].set(h/(2*(nq-1)))
    w  = w.at[-1].set(h/(2*(nq-1)))

    # reconstruction U(x)
    phi_q = vphi_at(xq, xL, xR)       
    p_q = phi_q @ u_cell[0]           
    v_q = phi_q @ u_cell[1]           
    Uq = jnp.stack([p_q, v_q], axis=1)

    Fq = Uq @ A.T                    

    # exact derivative of basis functions
    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    # Integration of flux * dphi
    V0 = jnp.sum(w[:,None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:,None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)   


v_local_volume_system = jax.vmap(local_volume_system, in_axes=(0, 0, 0, None, None))

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          Boundary conditions 
# ------------------------------------------------------------------------------------------------------------------------------

# RHS for phi ODE at right BC
def phi_rhs(pR, alpha, ZT):
    return -jnp.sqrt(alpha)/ZT * pR

# Define BC dataclass for Jax compatibility
@dataclass(frozen=True)
class BC:
    type: str
    left: tuple
    right: tuple

# Boundary condition at right with impedance and ODE

def apply_bc_right_impedance(u_cells, phi, beta, ZT, alpha):
        # values inside the domain at right boundary
        pR = u_cells[-1,0,1]
        vR = u_cells[-1,1,1]

        # outgoing wave
        w_plus = pR + vR

        # reflection coefficient
        r = (1.0 - beta / ZT) / (1.0 + beta / ZT)

        # incoming wave from ODE
        w_minus = r * w_plus + (2.0 * jnp.sqrt(alpha) / (1.0 + beta / ZT)) * phi

        # reconstruct p and v
        p_ext = 0.5 * (w_plus + w_minus)
        v_ext = 0.5 * (w_plus - w_minus)

        ghost_R = jnp.stack([
            jnp.array([p_ext, p_ext]),
            jnp.array([v_ext, v_ext])
        ])
        return ghost_R

def apply_bc(u_cells, bc_left, phi, beta, ZT, alpha):
    # u_cells: (N, 2, 2)
    ghost_L = jnp.stack([
        jnp.array([bc_left[0], bc_left[0]]),
        jnp.array([bc_left[1], bc_left[1]])
    ])

    ghost_R = apply_bc_right_impedance(u_cells, phi, beta, ZT, alpha)

    return jnp.concatenate([ghost_L[None, ...], u_cells, ghost_R[None, ...]],axis=0) #shape (N+2, 2, 2)


# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
def apply_bc_neumann(u_cells):
    ghost_L = u_cells[0]
    ghost_R = u_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_cells, ghost_R[None, ...]],
        axis=0
    )


def surface_term_system(u_ext, j, A, smax):
    # j = element index in original u_cells
    jp = j + 1  # step to account for ghost cell at start

    UL_left  = u_ext[jp-1, :, 1]   # ghost or cell j-1
    UR_left  = u_ext[jp,   :, 0]   # cell j

    UL_right = u_ext[jp,   :, 1]   # cell j
    UR_right = u_ext[jp+1, :, 0]   # cell j+1 or ghost
    f_left  = rusanov_flux(UL_left,  UR_left,  A, smax)
    f_right = rusanov_flux(UL_right, UR_right, A, smax)

    S = jnp.zeros((2,2))
    S = S.at[:,0].set(-f_left)
    S = S.at[:,1].set( f_right)
    return S


v_surface_term_system = jax.vmap(surface_term_system, in_axes=(None, 0, None, None))

def dg_rhs_system(u_cells, x_nodes, A, smax, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]
    
    # add ghost cells according to BC
    if bc.type == "dirichlet":
        u_ext = apply_bc(u_cells, bc.left, phi, beta, ZT, alpha)
    elif bc.type == "neumann":
        u_ext = apply_bc_neumann(u_cells)

    N = u_cells.shape[0]
    

    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, j, A, smax)
    )(jnp.arange(N))

    V_all = jax.vmap(
        lambda Ue, xL, xR: local_volume_system(Ue, xL, xR, A, 24)
        )(u_cells, xLs, xRs)
    
    def element_rhs(e):
        Vi = V_all[e]  
        Si = S_all[e]
        
        rhs_p = Mp_inv[e] @ (Vi[0] - Si[0])
        rhs_v = Mv_inv[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_p, rhs_v], axis=0)  
    RHS = jax.vmap(element_rhs)(jnp.arange(N))
    return RHS


# ------------------------------------------------------------------------------------------------------------------------------
#                                                           RK2 step
# ------------------------------------------------------------------------------------------------------------------------------

# RK2 for phi ODE at right BC
@jax.jit
def rk2_step_phi(u_cells, phi, dt, ZT, alpha):
    pL = u_cells[-1, 0, 1]  # p^-
    k1 = phi_rhs(pL, alpha, ZT)
    phi_mid = phi + 0.5 * dt * k1
    k2 = phi_rhs(pL, alpha, ZT)
    return phi + dt * k2

# RK2 time step for system
@jax.jit(static_argnames=("bc",))
def rk2_step_system(u_cells, x_nodes, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    # First phi
    phi_new = rk2_step_phi(u_cells, phi, dt, ZT, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, A, smax, Mp_inv, Mv_inv, bc, phi_new, beta, ZT, alpha)
    u_mid = u_cells + 0.5 * dt * k1
    k2 = dg_rhs_system(u_mid, x_nodes, A, smax, Mp_inv, Mv_inv, bc, phi_new, beta, ZT, alpha)
    return u_cells + dt * k2, phi_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           Euler step
# ------------------------------------------------------------------------------------------------------------------------------

#Euler step for phi ODE at right BC
@jax.jit
def euler_step_phi(u_cells, phi, dt, ZT, alpha):
    pL = u_cells[-1, 0, 1]  # p^-
    k1 = phi_rhs(pL, alpha, ZT)
    return phi + dt * k1

# Euler step for system
@jax.jit(static_argnames=("bc",))
def euler_step_system(u_cells, x_nodes, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, ZT, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, A, smax, Mp_inv, Mv_inv, bc,phi_new, beta, ZT, alpha)  # (N,2,2)
    return u_cells + dt * k1, phi_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           reconstruction for plotting
# ------------------------------------------------------------------------------------------------------------------------------
def reconstruct_system(u_cells, x_nodes, x_plot):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    h = xRs[0]-xLs[0]
    Ncells = u_cells.shape[0]
    idx = jnp.clip(jnp.floor(x_plot / h).astype(int), 0, Ncells-1)
    def eval_point(x, j):
        xL = xLs[j]; xR = xRs[j]
        ph = phi_at(x, xL, xR)
        # compute p and v
        p = jnp.dot(ph, u_cells[j,0])
        v = jnp.dot(ph, u_cells[j,1])
        return jnp.stack([p, v])
    UV = jax.vmap(eval_point)(x_plot, idx)  # (len(x_plot),2)
    return UV[:,0], UV[:,1] # p_rec, v_rec

# ------------------------------------------------------------------------------------------------------------------------------
#                                                 analytic solution for initial p0 Gaussian and v0=0
#                                                 decomposition into w+ = p+v, w- = p-v
# ------------------------------------------------------------------------------------------------------------------------------

def exact_solution_characteristics(x, t, p0_fun, c, L, alpha, beta, ZT, dt=1e-4):

    # ----------------------------
    # Precompute coefficients
    # ----------------------------
    a = (1.0 - beta / ZT) / (1.0 + beta / ZT)
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / ZT)
    c1 = jnp.sqrt(alpha) / (2.0 * ZT)
    c2 = c1 * b

    # ----------------------------
    # Left-going wave
    # ----------------------------
    w_plus = p0_fun(x - c * t)

    # ----------------------------
    # Boundary dynamics (x = L)
    # ----------------------------
    Nt = int(jnp.ceil(t / dt))
    t_grid = jnp.linspace(0.0, t, Nt)
    wp_L = p0_fun(L - c * t_grid)

    phi = 0.0
    w_minus_L = []

    for wp in wp_L:
        wm = a * wp + b * phi
        w_minus_L.append(wm)
        phi = phi + dt * (-c1 * wp - c2 * phi)

    w_minus_L = jnp.array(w_minus_L)

    # ----------------------------
    # Reflected wave propagation
    # ----------------------------
    t_ref = t - (L - x) / c

    w_minus_ref = jnp.where( #linear interpolation
        t_ref > 0.0,
        jnp.interp(t_ref, t_grid, w_minus_L, left=0.0, right=0.0),
        0.0
    )

    # ----------------------------
    # Initial right-going wave
    # ----------------------------
    w_minus_init = p0_fun(x + c * t)

    # ----------------------------
    # Total right-going wave
    # ----------------------------
    w_minus = w_minus_init + w_minus_ref

    # ----------------------------
    # Reconstruction
    # ----------------------------
    p = 0.5 * (w_plus + w_minus)
    v = 0.5 * (w_plus - w_minus)

    return p, v

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           time integrations stef
# ------------------------------------------------------------------------------------------------------------------------------
# Fist integrate 
# RK2 time integration
def time_integrate_rk2(u0, x_nodes, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, ZT, alpha):
    def step(carry, _):
        u, phi = carry
        u_next, phi_next = rk2_step_system(u, x_nodes, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)
        return (u_next, phi_next), None
    (u_final, phi_final), _ = lax.scan(step, (u0, phi0), None, length=nsteps)
    return u_final, phi_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, ZT, alpha):

    def step_sys(carry, _):
        u, phi = carry
        u_next,phi_next = euler_step_system(u, x_nodes, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)
        return (u_next, phi_next), None
    
    (u_final, phi_final), _ = lax.scan(step_sys, (u0, phi0), None, length=nsteps)

    return u_final, phi_final

# Made by chat Gpt
def parse_args():
    parser = argparse.ArgumentParser(description="DG P1 1D linear wave")
    parser.add_argument("--method", type=str, default="rk2",
                        choices=["euler", "rk2"],
                        help="Time integration method")
    parser.add_argument("--N", type=int, default=200,
                        help="Number of cells")
    parser.add_argument("--CFL", type=float, default=0.05,
                        help="CFL number")
    parser.add_argument("--tfinal", type=float, default=0.2,
                        help="Final time")
    parser.add_argument("--L", type=float, default=1,
                        help="Length of the domain")
    return parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------------------
#                                           Initial compactly supported bump function
# ------------------------------------------------------------------------------------------------------------------------------
#def init_func(x, phi0=1.0):
#    """
#    Bump function:
#        φ(x) = φ0/4 * exp( 1 - 1/(1 - (4x-2)**2) )  if |x-1/2| < 1/4
#               0                                  otherwise
#    """
#    mask = jnp.abs(x - 0.5) < 0.25
#    val = (phi0 / 4.0) * jnp.exp(1.0 - 1.0 / (1.0 - (4.0*x - 2.0)**2))
#    return jnp.where(mask, val, 0.0)

def init_func(x, L, phi0=1.0):
    xi = 4.0 * (x - 0.5 * L) / L

    inside = xi**2 < 1.0

    # dénominateur sûr (jamais nul)
    denom = jnp.where(inside, 1.0 - xi**2, 1.0)

    bump = (phi0 / 4.0) * jnp.exp(1.0 - 1.0 / denom)

    return jnp.where(inside, bump, 0.0)


def main():
    args = parse_args()

    method = args.method
    N = args.N
    CFL = args.CFL
    T = args.tfinal
    L = args.L

    # ------------------------------------------------------------------------------------------------------------------------------
    #                                                           physical params
    # ------------------------------------------------------------------------------------------------------------------------------
    c = 1.0
    alpha = 0.1  
    beta = 0.2  
    ZT = 1.0      

    A = jnp.array([[0.0, 1.0],[1.0, 0.0]])
    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))

    print('smax', smax)

    bc = BC(
        type="dirichlet",
        left=(0.0, 0.0),
        right=(0.0, 0.0),
    )

    # ------------------------------------------------------------------------------------------------------------------------------
    #                                                           initial condition
    # ------------------------------------------------------------------------------------------------------------------------------
    def p0(x):
        return init_func(x, L, phi0=1.0)

    def v0(x):
        return 0.0

    # ------------------------------------------------------------------------------------------------------------------------------
    #                                                           Convergence study
    # ------------------------------------------------------------------------------------------------------------------------------
    Ns = [100, 200, 400, 800]

    res_dir = 'Results'
    os.makedirs(res_dir, exist_ok=True)
    csv_file = os.path.join(res_dir, 'convergence_results.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N_cells', 'duration_sec', 'L2_error_p', 'L2_error_v', 'Linf_error_p', 'Linf_error_v'])

    p_errors = []
    v_errors = []
    durations = []

    # loop over Ns
    for N in Ns:
        print(f"Running simulation with N={N} cells")

        # mesh
        x_nodes, ghost_cells = create_uniform_nodes_with_ghosts(N, x_min=0.0, x_max=L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs
        #Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system)(hs, xLs, xRs)
        Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system)(hs)
       

        # u_cells shape (N,2,2)
        u_cells = jnp.stack([
            jnp.stack([
                jnp.array([p0(xLs[i]), p0(xRs[i])]),
                jnp.array([v0(xLs[i]), v0(xRs[i])])
            ]) for i in range(N)
        ], axis=0)

        # φ initial
        phi0_val = 0.0

        # CFL and dt
        h = xRs[0] - xLs[0]
        dt = CFL * h / smax
        nsteps = int(jnp.ceil(T / dt))
        print('dt', dt, 'nsteps', nsteps)

        # time integration
        u = u_cells.copy()
        phi = phi0_val
        t_start = time.time()

        if method == "euler":
            u, phi = time_integrate_euler(u, x_nodes, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)
        else:
            u, phi = time_integrate_rk2(u, x_nodes, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)

        duration = time.time() - t_start
        print("Simulation duration:", duration)
        durations.append(duration)

        # reconstruction
        x_plot = jnp.linspace(0.0, L, 2000)
        p_rec, v_rec = reconstruct_system(u, x_nodes, x_plot)

        # analytic
        #p_ex, v_ex, wplus_ex, wminus_ex = analytic_solution_pv(x_plot, p0, c, T)
        p_ex, v_ex = exact_solution_characteristics(x_plot, T, p0, c, L,alpha, beta, ZT,dt=1e-4)

        # errors
        dx_plot = x_plot[1] - x_plot[0]
        L2_p = jnp.sqrt(jnp.sum((p_rec - p_ex)**2) * dx_plot)
        L2_v = jnp.sqrt(jnp.sum((v_rec - v_ex)**2) * dx_plot)
        Linf_p = jnp.max(jnp.abs(p_rec - p_ex))
        Linf_v = jnp.max(jnp.abs(v_rec - v_ex))
        print(f"L2 error p: {L2_p:.3e}, v: {L2_v:.3e}")
        print(f"Linf error p: {Linf_p:.3e}, v: {Linf_v:.3e}")

        p_errors.append(float(L2_p))
        v_errors.append(float(L2_v))

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([N, duration, float(L2_p), float(L2_v), float(Linf_p), float(Linf_v)])

        # plotting per N
        plt.figure(figsize=(10,6))
        plt.subplot(2,1,1)
        plt.plot(x_plot, p_ex, '-', label='p exact')
        plt.plot(x_plot, p_rec, '--', label='p DG')
        plt.legend(); plt.grid(True); plt.title("Pressure p")

        plt.subplot(2,1,2)
        plt.plot(x_plot, v_ex, '-', label='v exact')
        plt.plot(x_plot, v_rec, '--', label='v DG')
        plt.legend(); plt.grid(True); plt.title("Velocity v")

        plt.tight_layout()
        plt.savefig(f'{res_dir}/dg_solution_N{N}.png', dpi=150)
        plt.close()

    # convergence plot
    plt.figure(figsize=(6,5))
    plt.loglog(Ns, p_errors, 'o-', label='L2 error p')
    plt.loglog(Ns, v_errors, 's--', label='L2 error v')
    plt.xlabel('Number of cells N')
    plt.ylabel('L2 error')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(f'{res_dir}/dg_convergence.png', dpi=150)
    plt.close()

    print("All done.")
if __name__ == "__main__":
    main()