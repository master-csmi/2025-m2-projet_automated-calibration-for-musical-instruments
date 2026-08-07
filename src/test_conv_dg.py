import csv
import json
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from utils.parse_args import parse_args
from utils.convergence import compute_metrics
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system
from numerics.time_integrators.rk2 import time_integrate_rk2
from utils.reconstruction import reconstruct_system
from utils.util_func import precompute_S_quad
from utils.build_physical_data import build_physical_data
from physics.bc import BC
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis


jax.config.update("jax_enable_x64", True)


def plot_DG_convergence(results, output_dir, type_S):
    h = np.array([r["h"] for r in results], dtype=float)
    err = np.array([r["rel_l2_shift"] for r in results], dtype=float)

    plt.figure(figsize=(7, 5))
    plt.loglog(h, err, "o-", linewidth=2, label="Erreur DG")

    ref1 = err[0] * (h / h[0])**1
    ref2 = err[0] * (h / h[0])**2

    plt.loglog(h, ref1, "--", label="ordre 1")
    plt.loglog(h, ref2, "--", label="ordre 2")

    plt.gca().invert_xaxis()
    plt.xlabel(r"$h \sim 1/N_x$")
    plt.ylabel("Erreur relative L2 recalée")
    plt.title(f"Convergence DG vs DG ({type_S})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"dg_convergence_{type_S}.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Figure sauvegardée :", fig_path)


def run_DG_solution(
    params,
    T_max,
    type_S,
    Nx,
    CFL,
    N_snapshot,
):
    data = build_physical_data(params, type_S)

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    L = data.L_tube + data.L_bell
    S_star = data.section(0.0)

    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)

    S_ext = S_nodes
    S_bc = S_cells.at[0].set(S_nodes[0]).at[-1].set(S_nodes[-1])

    Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=0)(hs)

    def p0(x):
        return init_func_const(x, L)

    def v0(x):
        return 0.0

    u0 = jnp.stack([
        jnp.stack([
            jnp.array([
                S_nodes[i] / (c * S_star) * p0(xLs[i]),
                S_nodes[i + 1] / (c * S_star) * p0(xRs[i]),
            ]),
            jnp.array([
                S_star / (c * S_nodes[i]) * v0(xLs[i]),
                S_star / (c * S_nodes[i + 1]) * v0(xRs[i]),
            ]),
        ])
        for i in range(Nx)
    ], axis=0)

    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max / dt))

    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot)
    ).astype(jnp.int32)

    t_solver = jnp.arange(nsteps) * dt

    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_solver,
    )

    bc = BC(type="full")

    (
        u_tilde,
        phi,
        y,
        y_dot,
        u_tilde_snaps,
        phi_snaps,
        y_snaps,
        z_snaps,
    ) = time_integrate_rk2(
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
        S_cells=S_bc,
        S_star=S_star,
        S_quad=S_quad,
        S_ext=S_ext,
        snapshot_steps=n_snaps,
        gamma_target=gamma_t,
    )

    jax.block_until_ready(u_tilde_snaps)

    x_plot = jnp.linspace(0.0, L, 1000)

    @jax.jit
    def reconstruct_all_snaps(u_tilde_snaps):
        return jax.vmap(
            lambda u_T: reconstruct_system(
                u_T,
                x_nodes,
                x_plot,
                data.section,
                c,
                S_star,
            )
        )(u_tilde_snaps)

    p_all, v_all = reconstruct_all_snaps(u_tilde_snaps)
    jax.block_until_ready(p_all)

    p_bell = p_all[:, -1]
    t_dg = jnp.asarray((n_snaps + 1) * dt)

    Pclosed = 5e3
    p_bell_scaled = Pclosed * p_bell

    return {
        "Nx": Nx,
        "h": float(1.0 / Nx),
        "t": t_dg,
        "p": p_bell_scaled,
        "dt": float(dt),
        "nsteps": int(nsteps),
    }


def run_DG_convergence(
    params,
    T_max,
    type_S,
    N_list,
    output_dir,
    CFL,
    N_snapshot,
):
    os.makedirs(output_dir, exist_ok=True)

    solutions = []

    for Nx in N_list:
        print("\n" + "=" * 80)
        print(f"DG : type_S={type_S}, Nx={Nx}")
        print("=" * 80)

        sol = run_DG_solution(
            params=params,
            T_max=T_max,
            type_S=type_S,
            Nx=Nx,
            CFL=CFL,
            N_snapshot=N_snapshot,
        )

        solutions.append(sol)

    results = []

    for k in range(len(solutions) - 1):
        coarse = solutions[k]
        fine = solutions[k + 1]

        t_coarse = coarse["t"]
        p_coarse = coarse["p"]

        common = (t_coarse >= fine["t"][0]) & (t_coarse <= fine["t"][-1])
        t_coarse = t_coarse[common]
        p_coarse = p_coarse[common]

        p_fine_interp = jnp.interp(
            t_coarse,
            fine["t"],
            fine["p"],
        )

        metrics = compute_metrics(
            p_coarse,
            p_fine_interp,
            t_coarse,
        )

        results.append({
            "Nx": coarse["Nx"],
            "Nx_ref": fine["Nx"],
            "h": coarse["h"],
            "rel_l2": float(metrics["rel_l2"]),
            "rel_l2_shift": float(metrics["rel_l2_shift"]),
            "corr_shifted": float(metrics["corr_shifted"]),
            "lag_index": int(metrics.get("lag_index", 0)),
            "shift_time": float(metrics.get("shift_time", 0.0)),
            "dt": coarse["dt"],
            "nsteps": coarse["nsteps"],
            "order_shift": None,
        })

    for i in range(1, len(results)):
        e_prev = results[i - 1]["rel_l2_shift"]
        e_curr = results[i]["rel_l2_shift"]

        h_prev = results[i - 1]["h"]
        h_curr = results[i]["h"]

        if e_prev > 0 and e_curr > 0:
            results[i]["order_shift"] = float(
                jnp.log(e_prev / e_curr) / jnp.log(h_prev / h_curr)
            )

    csv_path = os.path.join(
        output_dir,
        f"dg_convergence_{type_S}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nCSV sauvegardé :", csv_path)

    for r in results:
        print(
            f"Nx={r['Nx']:5d} -> {r['Nx_ref']:5d} | "
            f"err_shift={r['rel_l2_shift']:.4e} | "
            f"order={r['order_shift']}"
        )

    plot_DG_convergence(results, output_dir, type_S)


def main():
    args = parse_args()
    output_dir = "../experiments/convergence/conv_dg"

    with open("../experiments/convergence/config/simu.json", "r") as f:
        simu_params = json.load(f)

    T_max = simu_params["solver_params"]["T_max"]

    with open("../experiments/convergence/config/param.json", "r") as f:
        params = json.load(f)

    for type_S in ["const", "exp"]:
        run_DG_convergence(
            params=params,
            T_max=T_max,
            type_S=type_S,
            N_list=[400, 800,1600, 3200],
            output_dir=output_dir,
            CFL=simu_params["solver_params"]["cfl"],
            N_snapshot=simu_params["solver_params"]["N_snapshot"],
        )


if __name__ == "__main__":
    main()
