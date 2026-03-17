# ======================================================================================
# Main script for DG P1 simulation of the 1D linear wave equation with variable section
# ======================================================================================

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json

from parse_args import parse_args

# DG solver
from dg_solver.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from dg_solver.mass_matrix import local_mass_inv_system
from dg_solver.time_integrators import (
    time_integrate_euler,
    time_integrate_rk2
)
from dg_solver.reconstruction import reconstruct_system

# Boundary conditions
from bc.bc import BC

# Utilities
from utils.S_profiles import S_of_x
from utils.init_func import init_func, init_func_const

# Exact solution (only valid for constant section)
from physics.exact_solution import exact_solution_characteristics

jax.config.update("jax_enable_x64", True)


def main():

    #------------------------------------------------------------------------------
    # Read parameters from json file
    #------------------------------------------------------------------------------
    with open("utils/parameters.json", "r") as f:
        params = json.load(f)

        solver_params = params["solver_params"]
        left_bc_parameters = params["left_bc_params"]
        right_bc_parameters = params["right_bc_params"]
        initial_conditions_reed = params["init_cond_reed"]

        # Extract solver parameters
        c = solver_params["c"]
        S_star = solver_params["S_star"]
        T_max = solver_params["T_max"]
        phi0 = solver_params["phi0"]

        # Extract parameters for left BC
        gamma = left_bc_parameters["gamma"]
        epsilon = left_bc_parameters["epsilon"]
        kappa = left_bc_parameters["kappa"]
        f_r = left_bc_parameters["f_r"]
        Qr = left_bc_parameters["Qr"]
        zeta = left_bc_parameters["zeta"]

        # Extract initial conditions for reed
        y0 = initial_conditions_reed["y0"]
        z0 = initial_conditions_reed["y_dot0"]

        # Extract parameters for right BC
        beta = right_bc_parameters["beta"]
        Z = right_bc_parameters["Z"] #Z_T
        alpha = right_bc_parameters["alpha"]

    # ------------------------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------------------------
    args = parse_args()
    method = args.method
    type_S = args.type_S
    CFL = args.CFL
    L = args.L

    # ------------------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------------------
    N = 800       # Mesh refinements
    
    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])

    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print("smax =", smax)

    # Dirichlet BC at left + impedance BC at right
    bc = BC(type="dirichlet")

    # Initial conditions
    #def p0(x): return init_func(x, L, phi0=1.0)
    def p0(x): return init_func_const(x, L)
    def v0(x): return 0.0

    

    # Output directory
    output_dir = "Report_and_Presentation/Images"
    os.makedirs(output_dir, exist_ok=True)

    # Execution performance
    res_dir = "Results"
    os.makedirs(res_dir, exist_ok=True)


    # ==============================================================================
    # 1) SOLUTION FIGURES
    # ==============================================================================
    print("\n=== Computing solution plots ===")
   
    plt.figure(figsize=(16, 10))


    # ----------------------------------------------------------------------
    # Mesh
    # ----------------------------------------------------------------------
    x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    # ----------------------------------------------------------------------
    # Cross-section
    # ----------------------------------------------------------------------
    S_nodes = S_of_x(x_nodes, type_S)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    # ----------------------------------------------------------------------
    # Inverse mass matrices
    # ----------------------------------------------------------------------
    Mp_inv, Mv_inv = jax.vmap(
        local_mass_inv_system,
        in_axes=(0)
    )(hs)

    # ----------------------------------------------------------------------
    # Initial DG coefficients
    # ----------------------------------------------------------------------

    u0 = jnp.stack([
        jnp.stack([

            # ---- tilde p ----
            jnp.array([
                S_cells[i]/(c*S_star) * p0(xLs[i]),
                S_cells[i]/(c*S_star) * p0(xRs[i])
            ]),

            # ---- tilde v ----
            jnp.array([
                c*S_star/S_cells[i] * v0(xLs[i]),
                c*S_star/S_cells[i] * v0(xRs[i])
            ])

        ])
        for i in range(N)
    ], axis=0)

    # ----------------------------------------------------------------------
    # Time step
    # ----------------------------------------------------------------------
    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max/ dt))

    # To store snapshots for plot
    snapshot_steps = jnp.array([int(i * nsteps / 500) for i in range(1, 501)])
    print(f"  Snapshot steps: {snapshot_steps}")
    snapshots = {}

    # ----------------------------------------------------------------------
    # Time integration
    # ----------------------------------------------------------------------


    if method == "euler":
        u_tilde, phi, y, y_dot, y_snaps, u_tilde_snaps = time_integrate_euler(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0, beta, Z, alpha,
            y0, z0,
            epsilon, kappa, gamma, f_r, Qr, zeta,
            S_cells=S_cells, S_star=S_star,
            snapshot_steps=snapshot_steps

        )
    else:
        u_tilde, phi, y, y_dot, y_snaps, u_tilde_snaps = time_integrate_rk2(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0, beta, Z, alpha,
            y0, z0,
            epsilon, kappa, gamma, f_r, Qr, zeta,
            S_cells=S_cells, S_star=S_star,
            snapshot_steps=snapshot_steps
        )

    print("Before solve, u min/max:", u_tilde_snaps.min(), u_tilde_snaps.max())
    

    # ----------------------------------------------------------------------
    # Reconstruction
    # ----------------------------------------------------------------------
    x_plot = jnp.linspace(0.0, L, 2000)
    p_bell=[]
    for i, T in enumerate(snapshot_steps * dt):

        u_T = u_tilde_snaps[i]
        print(f"At T={T}, u min/max:", u_T.min(), u_T.max())
        p_num, v_num = reconstruct_system(
            u_T, x_nodes, x_plot, type_S, c, S_star
        )
        p_bell.append(p_num[-1])  # pressure at the bell (x=L)
    

    plt.plot(snapshot_steps * dt, p_bell, label="Pressure at bell (x=L)")
    plt.xlabel("Time")
    plt.ylabel("Pressure")
    plt.title(f"Pressure at bell over time (method={method}, type_S={type_S})")
    plt.legend()
            

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pressure_at_bell_{method}_{type_S}.png"),
        dpi=150
    )
    plt.close()

    print("y_snaps shape:", y_snaps.shape)
    # plot y 
    plt.figure(figsize=(8, 5))
    plt.plot(snapshot_steps * dt, y_snaps, label="Reed displacement y(t)")
    plt.xlabel("Time")
    plt.ylabel("Reed displacement y")
    plt.title(f"Reed displacement over time (method={method})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"reed_displacement_{method}.png"),
        dpi=150
    )
    plt.close()


   


if __name__ == "__main__":
    main()