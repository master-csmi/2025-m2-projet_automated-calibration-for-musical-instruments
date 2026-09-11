# ======================================================================================
# Main script for DG P1 simulation of the 1D linear wave equation with constant section
# ======================================================================================

import os
import json

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from src.utils.parse_args import parse_args

from src.physics.mouth_pressure import pressure_at_mouth_alexis
from src.physics.bc import BC
from src.physics.init_func import init_func

from src.numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from src.numerics.dg.mass_matrix import local_mass_inv_system

from src.numerics.time_integrators.euler import time_integrate_euler
from src.numerics.time_integrators.rk2 import time_integrate_rk2

from src.utils.reconstruction import reconstruct_system
from src.utils.exact_solution import exact_solution_characteristics
from src.utils.util_func import precompute_S_quad
from src.utils.build_physical_data import build_physical_data


jax.config.update("jax_enable_x64", True)


PLOT_EXACT_DT = 1e-4
CONVERGENCE_EXACT_DT = 1e-4
N_PLOT_POINTS = 2000


def build_initial_state(xLs, xRs, S_cells, c, S_star, L):
    def p0(x):
        return init_func(x, L)

    p_edges = jnp.stack((p0(xLs), p0(xRs)), axis=1)
    p_tilde = (S_cells[:, None] / (c * S_star)) * p_edges
    v_tilde = jnp.zeros_like(p_tilde)
    u0 = jnp.stack((p_tilde, v_tilde), axis=1)

    return u0, p0


def run_one_simulation(
    params,
    type_S,
    method,
    N,
    T_max,
    CFL,
    snapshot_times,
):
    data = build_physical_data(params, type_S)

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    L = data.L_tube + data.L_bell
    S_star = jnp.pi * data.R_tube**2

    bc = BC(type="right_free")

    x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    S_ext = jnp.concatenate([
        S_cells[:1],
        S_cells,
        S_cells[-1:],
    ])

    S_quad = precompute_S_quad(
        data.section,
        xLs,
        xRs,
        nq=2,
    )

    Mp_inv_cell, Mv_inv_cell = local_mass_inv_system(hs[0])
    Mp_inv = jnp.broadcast_to(Mp_inv_cell, (N,) + Mp_inv_cell.shape)
    Mv_inv = jnp.broadcast_to(Mv_inv_cell, (N,) + Mv_inv_cell.shape)

    u0, p0 = build_initial_state(
        xLs,
        xRs,
        S_cells,
        c,
        S_star,
        L,
    )

    h = xRs[0] - xLs[0]
    dt_cfl = CFL * h / c
    nsteps = int(jnp.ceil(T_max / dt_cfl))
    dt = T_max / nsteps

    snapshot_steps = jnp.array([
        max(0, int(jnp.ceil(T / dt)) - 1)
        for T in snapshot_times
    ], dtype=jnp.int32)

    snapshot_steps = jnp.clip(snapshot_steps, 0, nsteps - 1)

    t_solver = jnp.arange(nsteps) * dt

    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_solver,
    )

    if method == "euler":
        u_final, phi, y, z, u_snaps, phi_snaps, y_snaps, z_snaps = time_integrate_euler(
            u0,
            x_nodes,
            c,
            dt,
            nsteps,
            Mp_inv,
            Mv_inv,
            bc,
            phi0,
            y0,
            z0,
            data,
            S_cells=S_cells,
            S_star=S_star,
            S_quad=S_quad,
            S_ext=S_ext,
            snapshot_steps=snapshot_steps,
            gamma_target=gamma_t,
        )

    elif method == "rk2":
        u_final, phi, y, z, u_snaps, phi_snaps, y_snaps, z_snaps = time_integrate_rk2(
            u0,
            x_nodes,
            c,
            dt,
            nsteps,
            Mp_inv,
            Mv_inv,
            bc,
            phi0,
            y0,
            z0,
            data,
            S_cells=S_cells,
            S_star=S_star,
            S_quad=S_quad,
            S_ext=S_ext,
            snapshot_steps=snapshot_steps,
            gamma_target=gamma_t,
        )

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "data": data,
        "c": c,
        "L": L,
        "S_star": S_star,
        "x_nodes": x_nodes,
        "dt": dt,
        "nsteps": nsteps,
        "snapshot_steps": snapshot_steps,
        "u_snaps": u_snaps,
        "p0": p0,
    }


def plot_solution_figures(
    params,
    type_S,
    method,
    Ns,
    Ts,
    CFL,
    output_dir,
):
    print("\n=== Computing solution plots ===")

    T_max = max(Ts)

    for N in Ns:
        print(f"\nSolutions for N = {N}")

        sol = run_one_simulation(
            params=params,
            type_S=type_S,
            method=method,
            N=N,
            T_max=T_max,
            CFL=CFL,
            snapshot_times=Ts,
        )

        data = sol["data"]
        c = sol["c"]
        L = sol["L"]
        S_star = sol["S_star"]
        x_nodes = sol["x_nodes"]
        dt = sol["dt"]
        snapshot_steps = sol["snapshot_steps"]
        u_snaps = sol["u_snaps"]
        p0 = sol["p0"]

        print("dt =", dt)
        print("nsteps =", sol["nsteps"])
        print("u_snaps shape =", u_snaps.shape)
        print("u_snaps min/max =", u_snaps.min(), u_snaps.max())

        x_plot = jnp.linspace(0.0, L, N_PLOT_POINTS)

        plt.figure(figsize=(16, 10))

        plt.subplot(2, 1, 1)
        plt.title(f"Pressure p, N={N}, method={method}")

        plt.subplot(2, 1, 2)
        plt.title(f"Velocity v, N={N}, method={method}")

        for i, T_requested in enumerate(Ts):
            u_T = u_snaps[i]
            t_snap = float((snapshot_steps[i] + 1) * dt)

            print(
                f"T demandé={T_requested:.6e}, "
                f"t_snap={t_snap:.6e}, "
                f"diff={t_snap - T_requested:.3e}, "
                f"u min/max={u_T.min():.4e}/{u_T.max():.4e}"
            )

            p_num, v_num = reconstruct_system(
                u_T,
                x_nodes,
                x_plot,
                data.section,
                c,
                S_star,
            )

            if type_S == "const":
                p_ex, v_ex = exact_solution_characteristics(
                    x_plot,
                    t_snap,
                    p0,
                    c,
                    L,
                    data.alpha,
                    data.beta,
                    data.Zt,
                    dt=PLOT_EXACT_DT,
                    method=method,
                )
            else:
                p_ex = None
                v_ex = None

            plt.subplot(2, 1, 1)
            if p_ex is not None:
                plt.plot(x_plot, p_ex, "-", alpha=0.5, label=f"exact t={t_snap:.3e}")
            plt.plot(x_plot, p_num, "--", label=f"num t={t_snap:.3e}")

            plt.subplot(2, 1, 2)
            if v_ex is not None:
                plt.plot(x_plot, v_ex, "-", alpha=0.5, label=f"exact t={t_snap:.3e}")
            plt.plot(x_plot, v_num, "--", label=f"num t={t_snap:.3e}")

        plt.subplot(2, 1, 1)
        plt.grid(True)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        fig_path = os.path.join(
            output_dir,
            f"dg_solution_{type_S}_N{N}_{method}.png",
        )

        plt.savefig(fig_path, dpi=150)
        plt.close()

        print("Figure sauvegardée :", fig_path)


def convergence_study(
    params,
    type_S,
    method,
    N_convs,
    T_convs,
    CFL,
    output_dir,
):
    if type_S != "const":
        print("Convergence skipped: exact solution only available for const.")
        return

    print("\n=== Convergence study ===")

    T_max_conv = max(T_convs)
    p_errors_by_time = [[] for _ in T_convs]
    v_errors_by_time = [[] for _ in T_convs]

    for N in N_convs:
        print(f"\nConvergence run N = {N}")

        sol = run_one_simulation(
            params=params,
            type_S=type_S,
            method=method,
            N=N,
            T_max=T_max_conv,
            CFL=CFL,
            snapshot_times=T_convs,
        )

        data = sol["data"]
        c = sol["c"]
        L = sol["L"]
        S_star = sol["S_star"]
        x_nodes = sol["x_nodes"]
        dt = sol["dt"]
        snapshot_steps = sol["snapshot_steps"]
        u_snaps = sol["u_snaps"]
        p0 = sol["p0"]

        x_plot = jnp.linspace(0.0, L, N_PLOT_POINTS)
        dx = x_plot[1] - x_plot[0]

        for i, T_conv in enumerate(T_convs):
            t_snap = float((snapshot_steps[i] + 1) * dt)
            u_T = u_snaps[i]

            p_num, v_num = reconstruct_system(
                u_T,
                x_nodes,
                x_plot,
                data.section,
                c,
                S_star,
            )

            p_ex, v_ex = exact_solution_characteristics(
                x_plot,
                t_snap,
                p0,
                c,
                L,
                data.alpha,
                data.beta,
                data.Zt,
                dt=CONVERGENCE_EXACT_DT,
                method=method,
            )

            err_p = jnp.sqrt(jnp.sum((p_num - p_ex) ** 2) * dx)
            err_v = jnp.sqrt(jnp.sum((v_num - v_ex) ** 2) * dx)

            p_errors_by_time[i].append(float(err_p))
            v_errors_by_time[i].append(float(err_v))

            print(
                f"N={N:5d} | "
                f"T={T_conv:.6e} | "
                f"t_snap={t_snap:.8e} | "
                f"L2 p={err_p:.4e} | "
                f"L2 v={err_v:.4e}"
            )

    for i, T_conv in enumerate(T_convs):
        print(f"\n--- Final time T = {T_conv} ---")

        p_errors = p_errors_by_time[i]
        v_errors = v_errors_by_time[i]

        N_arr = jnp.array(N_convs, dtype=jnp.float64)
        p_arr = jnp.array(p_errors, dtype=jnp.float64)
        v_arr = jnp.array(v_errors, dtype=jnp.float64)

        p_order = -jnp.polyfit(jnp.log10(N_arr), jnp.log10(p_arr), 1)[0]
        v_order = -jnp.polyfit(jnp.log10(N_arr), jnp.log10(v_arr), 1)[0]

        print(f"\nOrder p = {p_order:.4f}")
        print(f"Order v = {v_order:.4f}")

        plt.figure(figsize=(7, 5))

        plt.loglog(
            N_convs,
            p_errors,
            "o-",
            label=f"L2 p, order={p_order:.2f}",
        )

        plt.loglog(
            N_convs,
            v_errors,
            "s--",
            label=f"L2 v, order={v_order:.2f}",
        )

        # références visuelles ordre 1 et 2
        ref1 = p_errors[0] * (N_arr / N_arr[0]) ** (-1)
        ref2 = p_errors[0] * (N_arr / N_arr[0]) ** (-2)

        plt.loglog(N_convs, ref1, ":", label="ordre 1")
        plt.loglog(N_convs, ref2, ":", label="ordre 2")

        plt.xlabel("Number of cells N")
        plt.ylabel("L2 error")
        plt.title(f"DG P1 convergence at T={T_conv}, method={method}")
        plt.grid(True, which="both", ls="--")
        plt.legend()
        plt.tight_layout()

        fig_path = os.path.join(
            output_dir,
            f"dg_convergence_T{T_conv}_{method}.png",
        )

        plt.savefig(fig_path, dpi=150)
        plt.close()

        print("Figure sauvegardée :", fig_path)


def main():
    with open("../experiments/open_instrument/config/simu.json", "r") as f:
        simu_params = json.load(f)

    with open("../experiments/open_instrument/config/param.json", "r") as f:
        params = json.load(f)

    args = parse_args()

    method = args.method
    type_S = args.type_S
    study = args.Th_study

    CFL = simu_params["solver_params"]["cfl"]

    if method == "euler":
        CFL = 0.02

    if method == "rk2":
        CFL = 0.2

    data = build_physical_data(params, type_S)

    S_star = jnp.pi * data.R_tube**2
    print("Section de la reed", S_star)

    A = jnp.array([[0.0, 1.0], [1.0, 0.0]])
    smax = params["physics"]["c"] * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print("smax =", smax)

    output_dir = f"../experiments/open_instrument/results/{method}"
    os.makedirs(output_dir, exist_ok=True)

    Ts = [1e-4, 0.1, 0.2, 0.5, 0.8]
    Ns = [100, 200, 400, 800]

    T_convs = [0.05, 0.08]
    N_convs = [1600, 3200, 6400, 12800]

    plot_solution_figures(
        params=params,
        type_S=type_S,
        method=method,
        Ns=Ns,
        Ts=Ts,
        CFL=CFL,
        output_dir=output_dir,
    )

    if study == "with":
        convergence_study(
            params=params,
            type_S=type_S,
            method=method,
            N_convs=N_convs,
            T_convs=T_convs,
            CFL=CFL,
            output_dir=output_dir,
        )

    print("\nAll computations completed.")


if __name__ == "__main__":
    main()
