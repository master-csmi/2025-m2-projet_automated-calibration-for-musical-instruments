# ===========================================================================================================
# Performance analysis for DG P1 1D linear wave equation with variable cross-section
# Measures wall time vs N and estimates computational complexity
# ===========================================================================================================

import os
import time
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.utils.parse_args import parse_args

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

# Boundary conditions
from src.physics.bc import BC

# Utilities
from src.physics.S_profiles import S_of_x
from src.physics.init_func import init_func

jax.config.update("jax_enable_x64", True)


# -----------------------------------------------------------------------------------------------------------
# Utility: compute log–log slope
# -----------------------------------------------------------------------------------------------------------
def compute_loglog_slope(N, T):
    """
    Estimate alpha such that T ~ N^alpha using least squares in log–log scale.
    """
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)

    mask = (N > 0) & (T > 0)
    N = N[mask]
    T = T[mask]

    logN = np.log(N)
    logT = np.log(T)

    alpha, _ = np.polyfit(logN, logT, 1)
    return alpha


# -----------------------------------------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------------------------------------
def main():

    args = parse_args()

    # ------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------
    types = ["const", "exp", "cone"]
    methods = ["euler", "rk2"]
    Ns = [100, 200, 300, 400, 500, 600, 700, 800, 1000, 1500, 2000]

    L = 2.0
    Tfinal = 0.5
    CFL = args.CFL
    c = 1.0

    alpha_bc = 0.1
    beta = 0.2
    Z = 1.0
    

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])
    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))

    bc = BC(type="dirichlet", left=(0.0, 0.0), right=(0.0, 0.0))

    def p0(x): return init_func(x, L, phi0=1.0)
    def v0(x): return 0.0

    output_dir = "Report_and_Presentation/Images"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Loop over methods
    # ------------------------------------------------------------
    for method in methods:

        exec_times = {t: [] for t in types}

        print(f"\n=== Performance study ({method}) ===")

        for type_S in types:
            print(f"\nCross-section: {type_S}")

            for N in Ns:
                print(f"  N = {N}")

                # ------------------------------------------------
                # Mesh
                # ------------------------------------------------
                x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
                xLs, xRs = cell_edges_from_nodes(x_nodes)
                hs = xRs - xLs

                # ------------------------------------------------
                # Cross-section
                # ------------------------------------------------
                S_nodes = S_of_x(x_nodes, type_S)
                S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

                # ------------------------------------------------
                # Inverse mass matrices
                # ------------------------------------------------
                Mp_inv, Mv_inv = jax.vmap(
                    local_mass_inv_system,
                    in_axes=(0)
                )(hs)

                # ------------------------------------------------
                # Initial condition
                # ------------------------------------------------
                S_star = 1.0   # même valeur que dans le solver

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

                # ------------------------------------------------
                # Time step
                # ------------------------------------------------
                h = xRs[0] - xLs[0]
                dt = CFL * h / c
                nsteps = int(np.ceil(Tfinal / dt))

                # ------------------------------------------------
                # Time integration + timing
                # ------------------------------------------------
                t_start = time.perf_counter()

                if method == "euler":
                    u, phi = time_integrate_euler(
                        u0, x_nodes, c, smax,
                        dt, nsteps, Mp_inv, Mv_inv,
                        bc, 0.0, beta, Z, alpha_bc
                    )
                else:
                    u, phi = time_integrate_rk2(
                        u0, x_nodes, c, smax,
                        dt, nsteps, Mp_inv, Mv_inv,
                        bc, 0.0, beta, Z, alpha_bc
                    )

                jax.tree_util.tree_map(lambda x: x.block_until_ready(), (u, phi))
                t_end = time.perf_counter()

                exec_times[type_S].append(t_end - t_start)

        res_dir = "Results"
        os.makedirs(res_dir, exist_ok=True)

        csv_file = os.path.join(res_dir, f"performance_{method}.csv")
        with open(csv_file, "w") as f:
            f.write(f"{method} method\n")
            for type_S in types:
                f.write(f"cross-section: {type_S}\n")
                f.write("N,wall_time,nsteps\n")
                Tvals = exec_times[type_S]
                for i, N in enumerate(Ns[:len(Tvals)]):
                    # recompute nsteps for CSV accuracy
                    dt = CFL * (L / N) / c
                    nsteps = int(np.ceil(Tfinal / dt))
                    f.write(f"{N},{Tvals[i]:.12f},{nsteps}\n")

        # ------------------------------------------------------------
        # Plot performance
        # ------------------------------------------------------------
        plt.figure(figsize=(6.5, 4.5))

        markers = ["o", "s", "^"]
        colors = ["tab:blue", "tab:orange", "tab:green"]

        for i, type_S in enumerate(types):
            Tvals = exec_times[type_S]
            Nvals = np.array(Ns[:len(Tvals)])

            slope = compute_loglog_slope(Nvals[1:], Tvals[1:])  # skip first point

            plt.plot(
                Nvals,
                Tvals,
                marker=markers[i],
                color=colors[i],
                label=f"{type_S} (slope = {slope:.2f})"
            )
        # reference N^2 curve 
        Ns_arr = np.array(Ns)
        if 1000 in Ns_arr:
            idx1000 = int(np.where(Ns_arr == 1000)[0][0])
            t_refs = [exec_times[t][idx1000] for t in types if len(exec_times[t]) > idx1000]
            if t_refs:
                Tref = np.mean(t_refs)
                N_line = np.linspace(Ns_arr.min(), Ns_arr.max(), 400)
                T_line = Tref * (N_line / 1000.0) ** 2
                plt.plot(N_line, T_line, linestyle="--", color="black", label=r"$N^2$")

        plt.xlabel("Number of cells $N$")
        plt.ylabel("Wall time (s)")
        plt.title(f"Performance scaling ({method.upper()})")
        plt.xscale("log")
        plt.yscale("log")
        plt.grid(True, which="both")
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            os.path.join(output_dir, f"performance_{method}.png"),
            dpi=150
        )
        plt.close()

    print("\nPerformance study completed.")


if __name__ == "__main__":
    main()
