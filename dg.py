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

import matplotlib.cm as cm



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
def S_of_x(x, type_S="const", **kwargs):
    """
    Cross-sectional area of the instrument at position x.

    Parameters
    ----------
    x : array_like
        Positions along the instrument.
    type_S : str
        Type of the section profile: "const", "exp", "cone", "bump".
    kwargs : dict
        Additional parameters for each type:
        - const: S0
        - exp: S0, k
        - cone: S0, k
        - bump: S0, A, x_c, sigma

    Returns
    -------
    Sx : array_like
        Cross-sectional area at positions x.
    """
    x = jnp.array(x)  # ensure JAX array

    if type_S == "const":
        S0 = kwargs.get("S0", 1.0)
        return jnp.ones_like(x) * S0

    elif type_S == "exp":
        S0 = kwargs.get("S0", 1.0)
        k  = kwargs.get("k", 0.8)
        return S0 * jnp.exp(k * x)

    elif type_S == "cone":
        S0 = kwargs.get("S0", 0.5)
        k  = kwargs.get("k", 1.0)
        return S0 + k * x

    elif type_S == "bump":
        S0    = kwargs.get("S0", 1.0)
        A     = kwargs.get("A", 0.5)
        x_c   = kwargs.get("x_c", 0.5)
        sigma = kwargs.get("sigma", 0.1)
        return S0 * (1 + A * jnp.exp(-(x - x_c)**2 / sigma**2))

    else:
        raise ValueError(f"Unknown type '{type_S}' for S(x)")
# ------------------------------------------------------------------------------------------------------------------------------
#                                                          analytic local mass matrix for P1 (S(x) variable)
# ------------------------------------------------------------------------------------------------------------------------------

def local_mass_inv_system(h, S_cell, c=1.0, S_star=1.0):
    """
    Correct DG P1 mass inverse for the weighted system
    """
    M_ref = (h / 6.0) * jnp.array([[2., 1.],
                                  [1., 2.]])

    Mp = (S_cell / (c * S_star)) * M_ref
    Mv = (S_star / (c * S_cell)) * M_ref

    return jnp.linalg.inv(Mp), jnp.linalg.inv(Mv)




# ------------------------------------------------------------------------------------------------------------------------------
#                                                          flux for system (linear)
# ------------------------------------------------------------------------------------------------------------------------------
def linear_system_flux(A):
    def Flux(U):
        # U shape (...,2)
        return U @ A.T 
    return Flux

def rusanov_flux(U_L, U_R, c=1.0):
    """
    Correct Rusanov flux for
    F(U) = (v, p)
    """
    smax = c
    F_L = jnp.array([U_L[1], U_L[0]])
    F_R = jnp.array([U_R[1], U_R[0]])

    return 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)

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


def surface_term_system(u_ext, j, c=1.0):
    """
    Compute DG surface term for cell j with S(x) variable.

    u_ext   : array (N+2,2,2) with ghost cells
    S_ext   : array (N+2,) S at nodes/interfaces
    j       : cell index in original u_cells
    """
    jp = j + 1  # offset due to ghost cell

    # Left interface
    UL_left  = u_ext[jp-1, :, 1]  # right node of left cell
    UR_left  = u_ext[jp,   :, 0]  # left node of current cell
   
    

    # Right interface
    UL_right = u_ext[jp,   :, 1]  # right node of current cell
    UR_right = u_ext[jp+1, :, 0]  # left node of right cell
    

    f_left  = rusanov_flux(UL_left, UR_left, c)
    f_right = rusanov_flux(UL_right, UR_right, c)
    # assemble surface term (2x2)
    S_term = jnp.zeros((2,2))
    S_term = S_term.at[:,0].set(-f_left)
    S_term = S_term.at[:,1].set( f_right)
    return S_term


v_surface_term_system = jax.vmap(surface_term_system, in_axes=(None, 0, None))

def dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]

    # add ghost cells according to BC
    if bc.type == "dirichlet":
        u_ext = apply_bc(u_cells, bc.left, phi, beta, ZT, alpha)
    elif bc.type == "neumann":
        u_ext = apply_bc_neumann(u_cells)

    # S at nodes/interfaces with ghost cells
    S_nodes_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])  

    # Surface term (fluxes)
    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, j, c=c)
    )(jnp.arange(N))

    # Volume term
    V_all = jax.vmap(
        lambda Ue, xL, xR: local_volume_system(Ue, xL, xR, A, 24)
    )(u_cells, xLs, xRs)

    # assemble RHS cell by cell
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
def rk2_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    # First phi
    phi_new = rk2_step_phi(u_cells, phi, dt, ZT, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, ZT, alpha)
    u_mid = u_cells + 0.5 * dt * k1
    k2 = dg_rhs_system(u_mid, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, ZT, alpha)
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
def euler_step_system(u_cells, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha):
    # First phi
    phi_new = euler_step_phi(u_cells, phi, dt, ZT, alpha)
    # Then RHS
    k1 = dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi_new, beta, ZT, alpha)  # (N,2,2)
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
def time_integrate_rk2(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, ZT, alpha):
    def step(carry, _):
        u, phi = carry
        u_next, phi_next = rk2_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)
        return (u_next, phi_next), None
    (u_final, phi_final), _ = lax.scan(step, (u0, phi0), None, length=nsteps)
    return u_final, phi_final

# Euler time integration
def time_integrate_euler(u0, x_nodes, S_cells, c, A, smax, dt, nsteps, Mp_inv, Mv_inv, bc, phi0, beta, ZT, alpha):
    def step_sys(carry, _):
        u, phi = carry
        u_next,phi_next = euler_step_system(u, x_nodes, S_cells, c, A, smax, dt, Mp_inv, Mv_inv, bc, phi, beta, ZT, alpha)
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
    parser.add_argument("--type_S", type=str, default="const",
                        help="Type of the section profile: 'const', 'exp', 'cone', 'bump'") 
    return parser.parse_args()


# ------------------------------------------------------------------------------------------------------------------------------
#                                           Initial compactly supported bump function
# ------------------------------------------------------------------------------------------------------------------------------


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
    type_S = args.type_S
    CFL = args.CFL
    L = args.L

    # -------------------------
    # Parameters
    # -------------------------
    Ts = [0.0001, 0.2, 0.5, 0.8]
    Ns = [100, 200, 400, 800]
    T_conv = 0.5

    c = 1.0
    alpha = 0.1
    beta = 0.2
    ZT = 1.0

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])
    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print("smax =", smax)

    bc = BC(type="dirichlet", left=(0.0, 0.0), right=(0.0, 0.0))

    def p0(x): return init_func(x, L, phi0=1.0)
    def v0(x): return 0.0

    output_dir = "Report_and_Presentation/Images"
    os.makedirs(output_dir, exist_ok=True)

    # ======================================================================
    # 1) SOLUTION FIGURES (for all N and T)
    # ======================================================================
    print("\n=== Computing solution plots ===")

    for N in Ns:
        print(f"\nSolutions for N = {N}")

        plt.figure(figsize=(16, 10))
        plt.subplot(2, 1, 1)
        plt.title(f"Pressure p (N={N})")
        plt.subplot(2, 1, 2)
        plt.title(f"Velocity v (N={N})")

        for T in Ts:
            print(f"  T = {T}")

            # ---- mesh
            x_nodes, ghost_cells = create_uniform_nodes_with_ghosts(
                N, 0.0, L
            )
            xLs, xRs = cell_edges_from_nodes(x_nodes)
            hs = xRs - xLs

            # ---- cross-section
            S_nodes = S_of_x(x_nodes, type_S)
            S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

            # ---- inverse mass matrices
            Mp_inv, Mv_inv = jax.vmap(
                local_mass_inv_system, in_axes=(0, 0, None, None)
            )(hs, S_cells, c, 1.0)

            # ---- initial condition
            u0 = jnp.stack([
                jnp.stack([
                    jnp.array([p0(xLs[i]), p0(xRs[i])]),
                    jnp.array([v0(xLs[i]), v0(xRs[i])])
                ])
                for i in range(N)
            ], axis=0)

            # ---- time step
            h = xRs[0] - xLs[0]
            dt = CFL * h / c
            nsteps = int(jnp.ceil(T / dt))

            # ---- time integration
            if method == "euler":
                u, phi = time_integrate_euler(
                    u0, x_nodes, S_cells, c, A, smax,
                    dt, nsteps, Mp_inv, Mv_inv,
                    bc, 0.0, beta, ZT, alpha
                )
            else:
                u, phi = time_integrate_rk2(
                    u0, x_nodes, S_cells, c, A, smax,
                    dt, nsteps, Mp_inv, Mv_inv,
                    bc, 0.0, beta, ZT, alpha
                )

            # ---- reconstruction
            x_plot = jnp.linspace(0.0, L, 2000)
            p_num, v_num = reconstruct_system(u, x_nodes, x_plot)

            # ---- exact solution (constant S only)
            if type_S == "const":
                p_ex, v_ex = exact_solution_characteristics(
                    x_plot, T, p0, c, L,
                    alpha, beta, ZT, dt=1e-4
                )
            else:
                p_ex = v_ex = None

            # ---- plots
            plt.subplot(2, 1, 1)
            if p_ex is not None:
                plt.plot(x_plot, p_ex, "-", alpha=0.7)
            plt.plot(x_plot, p_num, "--", label=f"T={T}")

            plt.subplot(2, 1, 2)
            if v_ex is not None:
                plt.plot(x_plot, v_ex, "-", alpha=0.7)
            plt.plot(x_plot, v_num, "--", label=f"T={T}")

        plt.subplot(2, 1, 1)
        plt.legend(); plt.grid(True)
        plt.subplot(2, 1, 2)
        plt.legend(); plt.grid(True)
        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"dg_solution_{type_S}_N{N}.png"),
            dpi=150
        )
        plt.close()

    # ======================================================================
    # 2) CONVERGENCE STUDY (T fixed)
    # ======================================================================
    if type_S != "const":
        print("\nConvergence skipped (no exact solution).")
        return

    print("\n=== Convergence study ===")

    p_errors = []
    v_errors = []

    for N in Ns:
        print(f"\nConvergence run N = {N}")

        # ---- mesh
        x_nodes, ghost_cells = create_uniform_nodes_with_ghosts(N, 0.0, L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs

        S_nodes = S_of_x(x_nodes, type_S)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

        Mp_inv, Mv_inv = jax.vmap(
            local_mass_inv_system, in_axes=(0, 0, None, None)
        )(hs, S_cells, c, 1.0)

        u0 = jnp.stack([
            jnp.stack([
                jnp.array([p0(xLs[i]), p0(xRs[i])]),
                jnp.array([v0(xLs[i]), v0(xRs[i])])
            ])
            for i in range(N)
        ], axis=0)

        h = xRs[0] - xLs[0]
        dt = CFL * h / c
        nsteps = int(jnp.ceil(T_conv / dt))

        # ---- time integration
        if method == "euler":
            u, phi = time_integrate_euler(
                u0, x_nodes, S_cells, c, A, smax,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, 0.0, beta, ZT, alpha
            )
        else:
            u, phi = time_integrate_rk2(
                u0, x_nodes, S_cells, c, A, smax,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, 0.0, beta, ZT, alpha
            )

        x_plot = jnp.linspace(0.0, L, 2000)
        p_num, v_num = reconstruct_system(u, x_nodes, x_plot)

        p_ex, v_ex = exact_solution_characteristics(
            x_plot, T_conv, p0, c, L,
            alpha, beta, ZT, dt=1e-4
        )

        dx = x_plot[1] - x_plot[0]
        p_errors.append(jnp.sqrt(jnp.sum((p_num - p_ex)**2) * dx))
        v_errors.append(jnp.sqrt(jnp.sum((v_num - v_ex)**2) * dx))

    # ---- plot convergence
    p_error_slope=jnp.polyfit(jnp.log10(jnp.array(Ns)), jnp.log10(jnp.array(p_errors)), 1)[0]
    v_error_slope=jnp.polyfit(jnp.log10(jnp.array(Ns)), jnp.log10(jnp.array(v_errors)), 1)[0]
    plt.figure(figsize=(6, 5))
    plt.loglog(Ns, p_errors, "o-", label=f"L2 error p (slope={p_error_slope:.2f})")
    plt.loglog(Ns, v_errors, "s--", label=f"L2 error v (slope={v_error_slope:.2f})")
    plt.xlabel("Number of cells N")
    plt.ylabel("L2 error")
    plt.title("DG P1 convergence at T = 0.5")
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"dg_convergence_T0p5_{method}.png"),
        dpi=150
    )
    plt.close()

    print("\nAll computations completed.")


if __name__ == "__main__":
    main()
