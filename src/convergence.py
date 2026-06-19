# ======================================================================================
# Pressure at bell test script
# ======================================================================================

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.parse_args import parse_args

from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system

from numerics.time_integrators.euler import time_integrate_euler
from numerics.time_integrators.rk2 import time_integrate_rk2

from utils.reconstruction import reconstruct_system
from utils.util_func import precompute_S_quad, project_L2, compute_v_bc_left, best_time_shift
from utils.param_func import set_param
from utils.res_openwind import run_openwind_reference
from utils.build_physical_data import build_physical_data
from utils.convergence import compute_metrics

from physics.bc import BC
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis

from inverse.spectral_loss import spectrogram_db


jax.config.update("jax_enable_x64", True)


GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

def run_case(Nx, CFL, N_snapshot_time, method, type_S):
    start_time = time.time()

    # --------------------------------------------------------------------------
    # Lecture config physique
    # --------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/param.json", "r") as f:
        params = json.load(f)

    with open("../experiments/pressure_at_bell/config/simu.json", "r") as f:
        simu_params = json.load(f)

    solver_params = simu_params["solver_params"]
    T_max = solver_params["T_max"]

    physical_params = params["physics"]
    initial_conditions_reed = params["init_cond_reed"]

    c = physical_params["c"]
    phi0 = physical_params["phi0"]

    y0 = initial_conditions_reed["y0"]
    z0 = initial_conditions_reed["y_dot0"]

    output_dir = "../experiments/pressure_at_bell/results"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Données physiques DG
    # --------------------------------------------------------------------------
    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell

    S_star = data.section(0.0)
    Zt = S_star / data.section(L)

    # --------------------------------------------------------------------------
    # OpenWind référence
    # --------------------------------------------------------------------------
    (
        t_ow,
        p_ow_left,
        p_ow_right,
        y_ow,
        gamma_ow,
        ow_rad_params,
    ) = run_openwind_reference(
        param_json=params,
        T_max=T_max,
        type_S=type_S,
    )

    # --------------------------------------------------------------------------
    # Injection des paramètres OpenWind dans DG
    # --------------------------------------------------------------------------
    data = set_param(data, "Zt", jnp.array(Zt), GEO_KEYS)

    if ow_rad_params["alpha"] is not None:
        data = set_param(
            data,
            "alpha",
            jnp.array(ow_rad_params["alpha"]),
            GEO_KEYS,
        )

    if ow_rad_params["beta"] is not None:
        data = set_param(
            data,
            "beta",
            jnp.array(ow_rad_params["beta"]),
            GEO_KEYS,
        )

    # --------------------------------------------------------------------------
    # Conditions initiales
    # --------------------------------------------------------------------------
    bc = BC(type="full")

    def p0(x):
        return init_func_const(x, L)

    def v0(x):
        return 0.0

    # --------------------------------------------------------------------------
    # Maillage DG
    # --------------------------------------------------------------------------
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])
    S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)

    Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=0)(hs)


    u0 = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.array(
                        [
                            S_cells[i] / (c * S_star) * p0(xLs[i]),
                            S_cells[i] / (c * S_star) * p0(xRs[i]),
                        ]
                    ),
                    jnp.array(
                        [
                            S_star / (c * S_cells[i]) * v0(xLs[i]),
                            S_star / (c * S_cells[i]) * v0(xRs[i]),
                        ]
                    ),
                ]
            )
            for i in range(Nx)
        ],
        axis=0,
    )

    # --------------------------------------------------------------------------
    # Temps
    # --------------------------------------------------------------------------
    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max / dt))

    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot_time)
    ).astype(jnp.int32)

    t_solver = jnp.arange(nsteps) * dt

    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_solver,
    )

    # --------------------------------------------------------------------------
    # Intégration temporelle DG
    # --------------------------------------------------------------------------
    if method == "euler":
        (
            u_tilde,
            phi,
            y,
            y_dot,
            u_tilde_snaps,
            phi_snaps,
            y_snaps,
            z_snaps,
        ) = time_integrate_euler(
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
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
        )
    else:
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
            S_cells=S_cells,
            S_star=S_star,
            S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
        )

    # --------------------------------------------------------------------------
    # Reconstruction au pavillon et à l'entrée
    # --------------------------------------------------------------------------
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

    p_bell = p_all[:, -1]
    p_left_dg = p_all[:, 0]

    t_dg = jnp.asarray(n_snaps * dt)

    # --------------------------------------------------------------------------
    # Interpolation OpenWind sur les temps DG
    # --------------------------------------------------------------------------
    p_ow_left_interp = jnp.interp(t_dg, t_ow, p_ow_left)
    p_ow_right_interp = jnp.interp(t_dg, t_ow, p_ow_right)

    opening = 5e-4
    y_ow_interp = jnp.interp(t_dg, t_ow, y_ow) / opening
    gamma_ow_interp = jnp.interp(t_dg, t_ow, gamma_ow)

    # --------------------------------------------------------------------------
    # Mise à l'échelle pression DG
    # --------------------------------------------------------------------------
    Pclosed = 5e3

    p_dg_scaled = Pclosed * p_bell
    p_left_dg_scaled = Pclosed * p_left_dg

    # --------------------------------------------------------------------------
    # Métriques principales sur pression au pavillon
    # --------------------------------------------------------------------------
    metrics = compute_metrics(
        p_dg_scaled,
        p_ow_right_interp,
        t_dg,
    )

    # --------------------------------------------------------------------------
    # Métriques supplémentaires
    # --------------------------------------------------------------------------
    gamma_dg_snaps = gamma_t[n_snaps]

    delta_p_dg = gamma_dg_snaps - p_left_dg
    delta_p_ow = gamma_ow_interp - p_ow_left_interp / Pclosed

    rel_err_delta_p = jnp.linalg.norm(delta_p_dg - delta_p_ow) / (
        jnp.linalg.norm(delta_p_ow) + 1e-12
    )

    rel_err_y = jnp.linalg.norm(y_snaps - y_ow_interp) / (
        jnp.linalg.norm(y_ow_interp) + 1e-12
    )

    corr = (
        jnp.vdot(
            p_dg_scaled - jnp.mean(p_dg_scaled),
            p_ow_right_interp - jnp.mean(p_ow_right_interp),
        )
        / (
            jnp.linalg.norm(p_dg_scaled - jnp.mean(p_dg_scaled))
            * jnp.linalg.norm(p_ow_right_interp - jnp.mean(p_ow_right_interp))
            + 1e-12
        )
    )

    scale_opt = jnp.vdot(p_bell, p_ow_right_interp) / (
        jnp.vdot(p_bell, p_bell) + 1e-12
    )

    # --------------------------------------------------------------------------
    # Fréquence dominante DG
    # --------------------------------------------------------------------------
    dt_snap = float(t_dg[1] - t_dg[0])

    p_signal = p_bell - jnp.mean(p_bell)
    freqs = jnp.fft.fftfreq(len(p_signal), d=dt_snap)
    spectrum = jnp.abs(jnp.fft.fft(p_signal))
    pos_mask = freqs > 0

    f_play = float(freqs[pos_mask][jnp.argmax(spectrum[pos_mask])])

    i_start = int(jnp.searchsorted(t_dg, data.t_attack * 3))

    p_steady = p_bell[i_start:] - jnp.mean(p_bell[i_start:])
    freqs_s = jnp.fft.fftfreq(len(p_steady), d=dt_snap)
    spectrum_s = jnp.abs(jnp.fft.fft(p_steady))
    pos_mask_s = freqs_s > 0

    f_play_steady = float(
        freqs_s[pos_mask_s][jnp.argmax(spectrum_s[pos_mask_s])]
    )

    amp_steady = float(0.5 * (jnp.max(p_steady) - jnp.min(p_steady)))
    y_mean_steady = float(jnp.mean(y_snaps[i_start:]))

    elapsed_time = time.time() - start_time

    # --------------------------------------------------------------------------
    # Stockage des résultats
    # --------------------------------------------------------------------------
    metrics.update(
        {
            "Nx": int(Nx),
            "CFL": float(CFL),
            "dt": float(dt),
            "nsteps": int(nsteps),
            "T_max": float(T_max),
            "N_snapshot": int(N_snapshot_time),
            "elapsed": float(elapsed_time),
            "rel_err_delta_p": float(rel_err_delta_p),
            "rel_err_y": float(rel_err_y),
            "corr_raw": float(corr),
            "scale_opt": float(scale_opt),
            "scale_opt_over_Pclosed": float(scale_opt / Pclosed),
            "max_abs_p_dg": float(jnp.max(jnp.abs(p_bell))),
            "max_abs_p_dg_scaled": float(jnp.max(jnp.abs(p_dg_scaled))),
            "max_abs_p_ow_right": float(jnp.max(jnp.abs(p_ow_right_interp))),
            "max_abs_p_ow_left": float(jnp.max(jnp.abs(p_ow_left_interp))),
            "f_play": float(f_play),
            "f_play_steady": float(f_play_steady),
            "amp_steady": float(amp_steady),
            "y_mean_steady": float(y_mean_steady),
        }
    )

    print("\n=== Résultat cas convergence ===")
    print(f"Nx={Nx}, CFL={CFL}, dt={float(dt):.3e}, nsteps={nsteps}")
    print(f"rel_l2        = {metrics['rel_l2']:.4e}")
    print(f"rel_l2_shift  = {metrics['rel_l2_shift']:.4e}")
    print(f"rel_linf      = {metrics['rel_linf']:.4e}")
    print(f"corr_shifted  = {metrics['corr_shifted']:.4f}")
    print(f"shift_time    = {metrics['shift_time']:.4e} s")
    print(f"elapsed       = {metrics['elapsed']:.2f} s")

    return metrics



def main():
    args = parse_args()
    method = args.method
    type_S = args.type_S

    output_dir = "../experiments/pressure_at_bell/results"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Lecture config solveur
    # --------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/simu.json", "r") as f:
        simu_params = json.load(f)

    solver_params = simu_params["solver_params"]

    Nx_ref = solver_params["Nx"]
    CFL_ref = solver_params["cfl"]
    N_snapshot_time = solver_params["N_snapshot"]

    # --------------------------------------------------------------------------
    # Cas de convergence
    # --------------------------------------------------------------------------
    cases = [
        {"Nx": 50, "CFL": CFL_ref},
        {"Nx": 100, "CFL": CFL_ref},
        {"Nx": 200, "CFL": CFL_ref},
        {"Nx": 400, "CFL": CFL_ref},
        {"Nx": 800, "CFL": CFL_ref},
        {"Nx": 1600, "CFL": CFL_ref},
    ]

    # Si tu veux centrer autour de ta config actuelle :
    # cases = [
    #     {"Nx": max(25, Nx_ref // 4), "CFL": CFL_ref},
    #     {"Nx": max(50, Nx_ref // 2), "CFL": CFL_ref},
    #     {"Nx": Nx_ref, "CFL": CFL_ref},
    #     {"Nx": 2 * Nx_ref, "CFL": CFL_ref},
    # ]

    results = []

    for case in cases:
        print("\n" + "=" * 80)
        print(
            f"Étude de convergence : "
            f"Nx={case['Nx']}, CFL={case['CFL']}, "
            f"method={method}, type_S={type_S}"
        )
        print("=" * 80)

        metrics = run_case(
            Nx=case["Nx"],
            CFL=case["CFL"],
            N_snapshot_time=N_snapshot_time,
            method=method,
            type_S=type_S,
        )

        results.append(metrics)

    # --------------------------------------------------------------------------
    # Estimation des ordres de convergence
    # --------------------------------------------------------------------------
    for i in range(len(results)):
        results[i]["order_rel_l2"] = None
        results[i]["order_rel_l2_shift"] = None
        results[i]["order_rel_linf"] = None

    for i in range(1, len(results)):
        e_prev = results[i - 1]["rel_l2"]
        e_curr = results[i]["rel_l2"]

        e_prev_shift = results[i - 1]["rel_l2_shift"]
        e_curr_shift = results[i]["rel_l2_shift"]

        e_prev_inf = results[i - 1]["rel_linf"]
        e_curr_inf = results[i]["rel_linf"]

        if e_curr > 0 and e_prev > 0:
            results[i]["order_rel_l2"] = float(jnp.log(e_prev / e_curr) / jnp.log(2.0))

        if e_curr_shift > 0 and e_prev_shift > 0:
            results[i]["order_rel_l2_shift"] = float(
                jnp.log(e_prev_shift / e_curr_shift) / jnp.log(2.0)
            )

        if e_curr_inf > 0 and e_prev_inf > 0:
            results[i]["order_rel_linf"] = float(
                jnp.log(e_prev_inf / e_curr_inf) / jnp.log(2.0)
            )

    # --------------------------------------------------------------------------
    # Sauvegarde CSV
    # --------------------------------------------------------------------------
    import csv

    csv_path = os.path.join(
        output_dir,
        f"convergence_{method}_{type_S}.csv",
    )

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("\n" + "=" * 80)
    print("Résultats de convergence")
    print("=" * 80)

    for r in results:
        print(
            f"Nx={r['Nx']:4d} | "
            f"dt={r['dt']:.3e} | "
            f"nsteps={r['nsteps']:7d} | "
            f"rel_l2={r['rel_l2']:.4e} | "
            f"rel_l2_shift={r['rel_l2_shift']:.4e} | "
            f"order_shift={r['order_rel_l2_shift']} | "
            f"elapsed={r['elapsed']:.2f}s"
        )

    print("\nCSV sauvegardé :", csv_path)

    # --------------------------------------------------------------------------
    # Figure convergence
    # --------------------------------------------------------------------------
    Nx_values = jnp.array([r["Nx"] for r in results])
    h_values = 1.0 / Nx_values

    err_l2 = jnp.array([r["rel_l2"] for r in results])
    err_l2_shift = jnp.array([r["rel_l2_shift"] for r in results])
    err_linf = jnp.array([r["rel_linf"] for r in results])

    plt.figure(figsize=(8, 5))
    plt.loglog(h_values, err_l2, "o-", label="Erreur L2 brute")
    plt.loglog(h_values, err_l2_shift, "o-", label="Erreur L2 recalée")
    plt.loglog(h_values, err_linf, "o-", label="Erreur Linf")

    plt.gca().invert_xaxis()
    plt.xlabel("h ~ 1/Nx")
    plt.ylabel("Erreur relative")
    plt.title(f"Convergence DG vs OpenWind — {method}, {type_S}")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        f"convergence_{method}_{type_S}.png",
    )

    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("Figure sauvegardée :", fig_path)


if __name__ == "__main__":
    main()