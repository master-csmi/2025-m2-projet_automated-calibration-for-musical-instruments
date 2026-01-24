# ======================================================================================
# Main script for DG P1 simulation of the 1D linear wave equation with variable section
# ======================================================================================

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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
from utils.init_func import init_func

# Exact solution (only valid for constant section)
from physics.exact_solution import exact_solution_characteristics

jax.config.update("jax_enable_x64", True)


def main():

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
    Ts = [0.0001, 0.2, 0.5, 0.8]      # Times for solution plots
    Ns = [100, 200, 400, 800]         # Mesh refinements
    T_conv = 0.5                      # Final time for convergence study

    c = 1.0
    alpha = 0.1
    beta = 0.2
    Z = 1.0
    T_star = 1.0 

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])

    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print("smax =", smax)

    # Dirichlet BC at left + impedance BC at right
    bc = BC(type="dirichlet", left=(0.0, 0.0), right=(0.0, 0.0))

    # Initial conditions
    def p0(x): return init_func(x, L, phi0=1.0)
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

    for N in Ns:
        print(f"\nSolutions for N = {N}")

        

        plt.figure(figsize=(16, 10))

        plt.subplot(2, 1, 1)
        plt.title(f"Pressure $p$ (N={N})")

        plt.subplot(2, 1, 2)
        plt.title(f"Velocity $v$ (N={N})")

        for T in Ts:
            print(f"  T = {T}")

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
                in_axes=(0, 0, None, None)
            )(hs, S_cells, c, 1.0)

            # ----------------------------------------------------------------------
            # Initial DG coefficients
            # ----------------------------------------------------------------------
            u0 = jnp.stack([
                jnp.stack([
                    jnp.array([p0(xLs[i]), p0(xRs[i])]),
                    jnp.array([v0(xLs[i]), v0(xRs[i])])
                ])
                for i in range(N)
            ], axis=0)

            # ----------------------------------------------------------------------
            # Time step
            # ----------------------------------------------------------------------
            h = xRs[0] - xLs[0]
            dt = CFL * h / c
            nsteps = int(jnp.ceil(T / dt))

            # ----------------------------------------------------------------------
            # Time integration
            # ----------------------------------------------------------------------


            if method == "euler":
                u, phi = time_integrate_euler(
                    u0, x_nodes, S_cells, c, A, smax,
                    dt, nsteps, Mp_inv, Mv_inv,
                    bc, 0.0, beta, Z, T_star, alpha
                )
            else:
                u, phi = time_integrate_rk2(
                    u0, x_nodes, S_cells, c, A, smax,
                    dt, nsteps, Mp_inv, Mv_inv,
                    bc, 0.0, beta, Z, T_star, alpha
                )
            

            # ----------------------------------------------------------------------
            # Reconstruction
            # ----------------------------------------------------------------------
            x_plot = jnp.linspace(0.0, L, 2000)
            p_num, v_num = reconstruct_system(u, x_nodes, x_plot)

            # ----------------------------------------------------------------------
            # Exact solution (only for constant section)
            # ----------------------------------------------------------------------
            if type_S == "const":
                p_ex, v_ex = exact_solution_characteristics(
                    x_plot, T, p0, c, L,
                    alpha, beta, Z, T_star, dt=1e-4
                )
            else:
                p_ex = v_ex = None

            # ----------------------------------------------------------------------
            # Plots
            # ----------------------------------------------------------------------
            plt.subplot(2, 1, 1)
            if p_ex is not None:
                plt.plot(x_plot, p_ex, "-", alpha=0.6)
            plt.plot(x_plot, p_num, "--", label=f"T={T}")

            plt.subplot(2, 1, 2)
            if v_ex is not None:
                plt.plot(x_plot, v_ex, "-", alpha=0.6)
            plt.plot(x_plot, v_num, "--", label=f"T={T}")

        plt.subplot(2, 1, 1)
        plt.legend()
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(
            os.path.join(output_dir, f"dg_solution_{type_S}_N{N}.png"),
            dpi=150
        )
        plt.close()


    # ==============================================================================
    # 2) CONVERGENCE STUDY
    # ==============================================================================
    if type_S != "const":
        print("\nConvergence study skipped (no exact solution available).")
        return

    print("\n=== Convergence study ===")

    p_errors = []
    v_errors = []

    for N in Ns:
        print(f"\nConvergence run N = {N}")

        x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs

        S_nodes = S_of_x(x_nodes, type_S)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

        Mp_inv, Mv_inv = jax.vmap(
            local_mass_inv_system,
            in_axes=(0, 0, None, None)
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

        if method == "euler":
            u, phi = time_integrate_euler(
                u0, x_nodes, S_cells, c, A, smax,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, 0.0, beta, Z, T_star, alpha
            )
        else:
            u, phi = time_integrate_rk2(
                u0, x_nodes, S_cells, c, A, smax,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, 0.0, beta, Z, T_star, alpha
            )

        x_plot = jnp.linspace(0.0, L, 2000)
        p_num, v_num = reconstruct_system(u, x_nodes, x_plot)

        p_ex, v_ex = exact_solution_characteristics(
            x_plot, T_conv, p0, c, L,
            alpha, beta, Z, T_star, dt=1e-4
        )

        dx = x_plot[1] - x_plot[0]
        p_errors.append(jnp.sqrt(jnp.sum((p_num - p_ex)**2) * dx))
        v_errors.append(jnp.sqrt(jnp.sum((v_num - v_ex)**2) * dx))

    # --------------------------------------------------------------------------
    # Convergence plot
    # --------------------------------------------------------------------------
    p_slope = jnp.polyfit(jnp.log10(jnp.array(Ns)),
                          jnp.log10(jnp.array(p_errors)), 1)[0]
    v_slope = jnp.polyfit(jnp.log10(jnp.array(Ns)),
                          jnp.log10(jnp.array(v_errors)), 1)[0]

    plt.figure(figsize=(6, 5))
    plt.loglog(Ns, p_errors, "o-", label=f"L2 error p (slope={p_slope:.2f})")
    plt.loglog(Ns, v_errors, "s--", label=f"L2 error v (slope={v_slope:.2f})")
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
