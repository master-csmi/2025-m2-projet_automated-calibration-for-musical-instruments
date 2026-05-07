# ======================================================================================
# Main script for DG P1 simulation of the 1D linear wave equation with variable section
# ======================================================================================

import os
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json

from src.utils.parse_args import parse_args

from physics.mouth_pressure import pressure_at_mouth_alexis

# DG solver
from src.numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from src.numerics.dg.mass_matrix import local_mass_inv_system

# Time integration 
from src.numerics.time_integrators.euler import (
    time_integrate_euler
)
from src.numerics.time_integrators.rk2 import (
    time_integrate_rk2
)

from src.utils.reconstruction import reconstruct_system

# Boundary conditions
from src.physics.bc import BC

# Utilities
from utils.util_func import precompute_S_quad

from src.physics.init_func import init_func, init_func_const


# Exact solution (only valid for constant section)
from src.utils.exact_solution import exact_solution_characteristics

from utils.build_physical_data import build_physical_data

jax.config.update("jax_enable_x64", True)




def main():

    #------------------------------------------------------------------------------
    # Read simulation parameters from json file
    #------------------------------------------------------------------------------
    with open("../experiments/convergence/config/simu.json", "r") as f:
        params = json.load(f)

        solver_params = params["solver_params"]

        # Extract solver parameters
        T_max = solver_params["T_max"]
        CFL = solver_params["cfl"]
        Nx = solver_params["Nx"]
        N_snapshot_time = solver_params["N_snapshot"]
    #------------------------------------------------------------------------------
    # Read physical parameters from json file
    #------------------------------------------------------------------------------
    with open("../experiments/convergence/config/param.json", "r") as f:
        params = json.load(f)
        physical_params = params["physics"]
        initial_conditions_reed = params["init_cond_reed"]
        instrument_geometry = params["instrument_geometry"]

        # Extract solver parameters
        c = physical_params["c"]

        
        phi0 = physical_params["phi0"]

        L_tube = instrument_geometry["tube"]["L_tube"]

        # Extract initial conditions for reed
        y0 = initial_conditions_reed["y0"]
        z0 = initial_conditions_reed["y_dot0"]

    # ------------------------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------------------------
    args = parse_args()
    method = args.method
    type_S = args.type_S
    study = args.Th_study

    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell

    S_star = jnp.pi * (data.R_tube**2)  # section de référence pour les variables tilde
    print("Section de la reed", S_star)

    # ------------------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------------------
    Ts = [0.0001,0.1,0.2,0.5,0.8]      # Times for solution plots
    
    Ns = [100, 200, 400, 800]         # Mesh refinements
    T_convs = [0.5,0.8]
    N_convs = [100, 200, 400, 800,1000,1500,2000]    # Mesh refinements for convergence study
    

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])

    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print("smax =", smax)

    # Dirichlet BC at left + impedance BC at right
    type = "right_free"
    assert type in ["full", "right_free"], "Invalid BC type : for test use 'right_free', for full simulation use 'full'"
    bc = BC(type="right_free")

    # Initial conditions
    def p0(x): return init_func(x, L)
    #def p0(x): return 0.0
    def v0(x): return 0.0

    

    # Output directory
    output_dir = f"../experiments/convergence/results/{method}"
    os.makedirs(output_dir, exist_ok=True)

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


        # ----------------------------------------------------------------------
        # Mesh
        # ----------------------------------------------------------------------
        x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs

        # ----------------------------------------------------------------------
        # Cross-section
        # ----------------------------------------------------------------------
        S_nodes = data.section(x_nodes)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

        S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)  # (N, nq) sections pré-calculées pour quadrature


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
                    S_star/(c*S_cells[i]) * v0(xLs[i]),
                    S_star/(c*S_cells[i]) * v0(xRs[i])
                ])

            ])
            for i in range(N)
        ], axis=0)

        print("u0 min/max:", u0.min(), u0.max())
        print("u0 shape:", u0.shape)

        print("S_cells min/max:", S_cells.min(), S_cells.max())
        print("u0 min/max:", u0.min(), u0.max())
        print("y0:", y0, "z0:", z0)
        print("phi0:", phi0)
        print("CFL", CFL)

        # ----------------------------------------------------------------------
        # Time step
        # ----------------------------------------------------------------------
        h = xRs[0] - xLs[0]
        dt = CFL * h / c
        nsteps = int(jnp.ceil(T_max/ dt))
        print("nombre de steps", nsteps)

        # snapshot for plotting
        n_snaps = jnp.array([int(jnp.ceil(T / dt)) -1  for T in Ts])


        # Grille temporelle du solveur
        t_solver = jnp.arange(nsteps) * dt

        gamma_t = pressure_at_mouth_alexis(
        gamma_final = data.gamma_final,   # valeur plateau depuis JSON
        t_attack    = data.t_attack,  
        t    = t_solver
        )


        # ----------------------------------------------------------------------
        # Time integration
        # ----------------------------------------------------------------------


        if method == "euler":
            u_tilde, phi, y, y_dot,u_tilde_snaps, phi_snaps, y_snaps,z_snaps,= time_integrate_euler(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0,
                y0, z0,
                data,
                S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                snapshot_steps=n_snaps,
                gamma_target=gamma_t

            )
        else:
            u_tilde, phi, y, y_dot,u_tilde_snaps, phi_snaps, y_snaps,z_snaps= time_integrate_rk2(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0, 
                y0, z0,
                data,
                S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                snapshot_steps=n_snaps,
                gamma_target=gamma_t
            )

        print("Before solve, u min/max:", u_tilde_snaps.min(), u_tilde_snaps.max())
        print("u_tilde_snap.shape", u_tilde_snaps.shape)
        

        # ----------------------------------------------------------------------
        # Reconstruction
        # ----------------------------------------------------------------------
        x_plot = jnp.linspace(0.0, L, 2000)

        @jax.jit
        def reconstruct_all_snaps(u):
            return jax.vmap(
                lambda u_T: reconstruct_system(u_T, x_nodes, x_plot, data.section, c, S_star)
            )(u)

        if study == "with":
            # ----------------------------------------------------------------------
            # Exact solution (only for constant section)
            # ----------------------------------------------------------------------
            for idx, T in enumerate(Ts):
                u_T = u_tilde_snaps[idx, :, :, :]
                print(f"At T={T}, u min/max:", u_T.min(), u_T.max())
                p_num, v_num = reconstruct_system(u_T, x_nodes, x_plot, data.section, c, S_star)

                # ----------------------------------------------------------------------
                # Exact solution (only for constant section)
                # ----------------------------------------------------------------------
                if type_S == "const":
                    p_ex, v_ex = exact_solution_characteristics(
                        x_plot, T, p0, c, L,
                        data.alpha, data.beta, data.Zt,
                        dt=1e-4,
                        method=method
                    )
                else:
                    p_ex = v_ex = None

                # ----------------------------------------------------------------------
                # Plots
                # ----------------------------------------------------------------------
                # ----------------------------------------------------------------------
                # Plots
                # ----------------------------------------------------------------------
                plt.subplot(2, 1, 1)
                if p_ex is not None:
                    plt.plot(x_plot, p_ex, "-", alpha=0.6, label=f"Exact at T={T}")
                plt.plot(x_plot, p_num, "--", label=f"T={T}")

                plt.subplot(2, 1, 2)
                if v_ex is not None:
                    plt.plot(x_plot, v_ex, "-", alpha=0.6, label=f"Exact at T={T}")
                plt.plot(x_plot, v_num, "--", label=f"T={T}")

            plt.subplot(2, 1, 1)
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"dg_solution_{type_S}_N{N}_method_{method}.png"),
                dpi=150
            )
            plt.close()


    # ==============================================================================
    # 2) CONVERGENCE STUDY
    # ==============================================================================
    if study == "with":
        if type_S != "const":
            print("\nConvergence study skipped (no exact solution available).")
            return

        print("\n=== Convergence study ===")

        T_max_conv = max(T_convs)

        for T_conv in T_convs:

            print(f"\n--- Final time T = {T_conv} ---")

            p_errors = []
            v_errors = []

            for N in N_convs:

                print(f"\nConvergence run N = {N}")

                # ----------------------------------------------------------------------
                # Mesh
                # ----------------------------------------------------------------------
                x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
                xLs, xRs = cell_edges_from_nodes(x_nodes)
                hs = xRs - xLs

                # ----------------------------------------------------------------------
                # Cross-section
                # ----------------------------------------------------------------------
                S_nodes = data.section(x_nodes)
                S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

                S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)  # (N, nq) sections pré-calculées pour quadrature

                # ----------------------------------------------------------------------
                # Mass matrices
                # ----------------------------------------------------------------------
                Mp_inv, Mv_inv = jax.vmap(
                    local_mass_inv_system,
                    in_axes=(0)
                )(hs)

                # ----------------------------------------------------------------------
                # Initial condition
                # ----------------------------------------------------------------------
                u0 = jnp.stack([
                    jnp.stack([
                        jnp.array([
                            S_cells[i]/(c*S_star) * p0(xLs[i]),
                            S_cells[i]/(c*S_star) * p0(xRs[i])
                        ]),
                        jnp.array([
                            S_star/(c*S_cells[i]) * v0(xLs[i]),
                            S_star/(c*S_cells[i]) * v0(xRs[i])
                        ])
                    ])
                    for i in range(N)
                ], axis=0)

                # ----------------------------------------------------------------------
                # Time step
                # ----------------------------------------------------------------------
                h = xRs[0] - xLs[0]
                dt = CFL * h / c
                nsteps = int(jnp.ceil(T_max_conv / dt))

                n_snaps = jnp.array([int(jnp.ceil(T / dt))-1 for T in T_convs])
                N_snapshot_time = len(Ts)

                # Grille temporelle du solveur
                t_solver = jnp.arange(nsteps) * dt

                gamma_t = pressure_at_mouth_alexis(
                gamma_final = data.gamma_final,   # valeur plateau depuis JSON
                t_attack    = data.t_attack,  
                t    = t_solver
                )
                

        

                # ----------------------------------------------------------------------
                # Time integration
                # ----------------------------------------------------------------------
                if method == "euler":
                    u_tilde, phi, y, y_dot,u_tilde_snaps, phi_snaps, y_snaps,z_snaps= time_integrate_euler(
                        u0, x_nodes, c,
                        dt, nsteps, Mp_inv, Mv_inv,
                        bc, phi0,
                        y0, z0,
                        data,
                        S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                        snapshot_steps=n_snaps,
                        gamma_target=gamma_t

                    )
                else:
                    u_tilde, phi, y, y_dot,u_tilde_snaps, phi_snaps, y_snaps,z_snaps= time_integrate_rk2(
                        u0, x_nodes, c,
                        dt, nsteps, Mp_inv, Mv_inv,
                        bc, phi0, 
                        y0, z0,
                        data,
                        S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                        snapshot_steps=n_snaps,
                        gamma_target=gamma_t
                    )

                # ----------------------------------------------------------------------
                # Reconstruction grid
                # ----------------------------------------------------------------------
                x_plot = jnp.linspace(0.0, L, 2000)

                i = T_convs.index(T_conv)

                p_num, v_num = reconstruct_all_snaps(
                    u_tilde_snaps
                )

                p_ex, v_ex = exact_solution_characteristics(
                    x_plot, T_conv, p0, c, L,
                    data.alpha, data.beta, data.Zt,
                    dt=1e-4,
                    method=method
                )

                dx = x_plot[1] - x_plot[0]

                p_errors.append(jnp.sqrt(jnp.sum((p_num[i] - p_ex)**2) * dx))
                v_errors.append(jnp.sqrt(jnp.sum((v_num[i] - v_ex)**2) * dx))
                print(f"  L2 error p: {p_errors[-1]:.4e}, L2 error v: {v_errors[-1]:.4e}")

            # --------------------------------------------------------------------------
            # Convergence plot
            # --------------------------------------------------------------------------
            p_slope = jnp.polyfit(jnp.log10(jnp.array(N_convs)),
                                jnp.log10(jnp.array(p_errors)), 1)[0]
            v_slope = jnp.polyfit(jnp.log10(jnp.array(N_convs)),
                                jnp.log10(jnp.array(v_errors)), 1)[0]

            plt.figure(figsize=(6, 5))
            plt.loglog(N_convs, p_errors, "o-", label=f"L2 error p (slope={p_slope:.2f})")
            plt.loglog(N_convs, v_errors, "s--", label=f"L2 error v (slope={v_slope:.2f})")
            plt.xlabel("Number of cells N")
            plt.ylabel("L2 error")
            plt.title(f"DG P1 convergence at T = {T_conv}")
            plt.grid(True, which="both", ls="--")
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, f"dg_convergence_T{T_conv}_{method}.png"),
                dpi=150
            )
            plt.close()

        print("\n Theoretical Study : Done")
        

    print("\nAll computations completed.")


if __name__ == "__main__":
    main()