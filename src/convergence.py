# ======================================================================================
# Convergence DG vs OpenWind — comparaison const / exp
# ======================================================================================

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import time
import copy
import csv
import hashlib

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.parse_args import parse_args

from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system

from numerics.time_integrators.rk2 import time_integrate_rk2

from utils.reconstruction import reconstruct_system
from utils.util_func import precompute_S_quad
from utils.res_openwind import run_openwind_reference
from utils.build_physical_data import build_physical_data
from utils.convergence import compute_metrics

from physics.bc import BC
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis


jax.config.update("jax_enable_x64", True)


OPENWIND_REF_ORDER = 4
OPENWIND_REF_THETA = 0.5
OPENWIND_REF_L_ELE = {
    "const": 1e-3,
    "exp": 1e-3,
}
OPENWIND_REF_N_POINTS = {
    "const": 2,
    "exp": 401,
}
DG_NX_LIST = [200,400,800,1600]
MAX_SNAPSHOTS = 2000
CACHE_OPENWIND_REFERENCE = True
METHOD_CFL = {
    "rk2": None,
}
EXPECTED_ORDER = {
    "rk2": 2,
}


def openwind_cache_path(output_dir, params, T_max, type_S, n_points, l_ele, order, theta):
    cache_dir = os.path.join(output_dir, "openwind_cache")
    os.makedirs(cache_dir, exist_ok=True)

    key_data = {
        "instrument_geometry": params["instrument_geometry"],
        "left_bc_params": params["left_bc_params"],
        "T_max": float(T_max),
        "type_S": type_S,
        "openwind_geometry_mode": "native_linear_exponential_v1",
        "n_points": int(n_points),
        "l_ele": float(l_ele),
        "order": int(order),
        "theta": float(theta),
    }
    key = hashlib.sha1(json.dumps(key_data, sort_keys=True).encode()).hexdigest()[:12]
    filename = f"ow_ref_{type_S}_np{n_points}_le{l_ele:.1e}_{key}.npz"
    return os.path.join(cache_dir, filename)


def load_openwind_reference(cache_path):
    if not os.path.exists(cache_path):
        return None

    data = np.load(cache_path)
    rad_params = json.loads(str(data["rad_params_json"]))
    return (
        data["t_ow"],
        data["p_ow_left"],
        data["p_ow_right"],
        data["y_ow"],
        data["gamma_ow"],
        rad_params,
    )


def save_openwind_reference(cache_path, openwind_reference):
    t_ow, p_ow_left, p_ow_right, y_ow, gamma_ow, rad_params = openwind_reference
    np.savez(
        cache_path,
        t_ow=np.asarray(t_ow),
        p_ow_left=np.asarray(p_ow_left),
        p_ow_right=np.asarray(p_ow_right),
        y_ow=np.asarray(y_ow),
        gamma_ow=np.asarray(gamma_ow),
        rad_params_json=json.dumps(rad_params),
    )


# ======================================================================================
# Un cas de convergence pour une géométrie donnée
# ======================================================================================

def run_case(
    Nx,
    CFL,
    N_snapshot_time,
    method,
    type_S,
    params,
    T_max,
    openwind_reference,
):
    start_time = time.time()

    physical_params = params["physics"]
    initial_conditions_reed = params["init_cond_reed"]

    c = physical_params["c"]
    phi0 = physical_params["phi0"]

    y0 = initial_conditions_reed["y0"]
    z0 = initial_conditions_reed["y_dot0"]

    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell
    S_star = data.section(0.0)

    (
        t_ow,
        p_ow_left,
        p_ow_right,
        y_ow,
        gamma_ow,
        _,
    ) = openwind_reference

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

    S_quad = precompute_S_quad(
        data.section,
        xLs,
        xRs,
        nq=2,
    )

    S_ext = S_nodes
    S_bc = S_cells.at[0].set(S_nodes[0]).at[-1].set(S_nodes[-1])

    Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=0)(hs)

    u0 = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.array(
                        [
                            S_nodes[i] / (c * S_star) * p0(xLs[i]),
                            S_nodes[i + 1] / (c * S_star) * p0(xRs[i]),
                        ]
                    ),
                    jnp.array(
                        [
                            S_star / (c * S_nodes[i]) * v0(xLs[i]),
                            S_star / (c * S_nodes[i + 1]) * v0(xRs[i]),
                        ]
                    ),
                ]
            )
            for i in range(Nx)
        ],
        axis=0,
    )

    # --------------------------------------------------------------------------
    # Temps DG
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
    if method != "rk2":
        raise ValueError(f"Méthode inconnue : {method}")

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

    # --------------------------------------------------------------------------
    # Reconstruction
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
    jax.block_until_ready(p_all)

    p_bell = p_all[:, -1]
    p_left_dg = p_all[:, 0]

    # Les intégrateurs stockent u_next quand n == snapshot_step.
    # Le snapshot d'indice n correspond donc au temps physique (n + 1) * dt.
    t_dg = jnp.asarray((n_snaps + 1) * dt)

    common = (t_dg >= t_ow[0]) & (t_dg <= t_ow[-1])
    t_dg = t_dg[common]
    p_bell = p_bell[common]
    p_left_dg = p_left_dg[common]
    y_snaps = y_snaps[common]

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

    # --------------------------------------------------------------------------
    # Métriques principales
    # --------------------------------------------------------------------------
    metrics = compute_metrics(
        p_dg_scaled,
        p_ow_right_interp,
        t_dg,
    )

    gamma_dg_snaps = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_dg,
    )

    delta_p_dg = gamma_dg_snaps - p_left_dg
    delta_p_ow = gamma_ow_interp - p_ow_left_interp / Pclosed

    rel_err_delta_p = jnp.linalg.norm(delta_p_dg - delta_p_ow) / (
        jnp.linalg.norm(delta_p_ow) + 1e-12
    )

    rel_err_y = jnp.linalg.norm(y_snaps - y_ow_interp) / (
        jnp.linalg.norm(y_ow_interp) + 1e-12
    )

    corr_raw = (
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
    # Fréquences dominantes
    # --------------------------------------------------------------------------
    dt_snap = float(t_dg[1] - t_dg[0])

    p_signal = p_bell - jnp.mean(p_bell)
    freqs = jnp.fft.fftfreq(len(p_signal), d=dt_snap)
    spectrum = jnp.abs(jnp.fft.fft(p_signal))
    pos_mask = freqs > 0

    f_play = float(freqs[pos_mask][jnp.argmax(spectrum[pos_mask])])

    i_start = int(jnp.searchsorted(t_dg, data.t_attack * 3))

    # Sécurité : on garde au moins une partie du signal
    if i_start >= len(p_bell) - 2:
        print("Attention : régime établi trop court, utilisation de la seconde moitié du signal.")
        i_start = len(p_bell) // 2

    p_steady = p_bell[i_start:] - jnp.mean(p_bell[i_start:])

    # valeurs par défaut
    f_play_steady = 0.0
    amp_steady = 0.0
    y_mean_steady = float(jnp.mean(y_snaps))

    if len(p_steady) < 2:
        print("Signal trop court pour calculer une fréquence dominante.")
        f_play_steady = 0.0
    else:
        freqs_s = jnp.fft.fftfreq(len(p_steady), d=dt_snap)
        spectrum_s = jnp.abs(jnp.fft.fft(p_steady))
        pos_mask_s = freqs_s > 0

        if jnp.sum(pos_mask_s) == 0:
            f_play_steady = 0.0
        else:
            f_play_steady = float(
                freqs_s[pos_mask_s][jnp.argmax(spectrum_s[pos_mask_s])]
            )

            amp_steady = float(
                0.5 * (jnp.max(p_steady) - jnp.min(p_steady))
            )

            y_mean_steady = float(jnp.mean(y_snaps[i_start:]))
    

    elapsed_time = time.time() - start_time

    metrics.update(
        {
            "type_S": type_S,
            "Nx": int(Nx),
            "CFL": float(CFL),
            "dt": float(dt),
            "nsteps": int(nsteps),
            "T_max": float(T_max),
            "N_snapshot": int(N_snapshot_time),
            "elapsed": float(elapsed_time),
            "rel_err_delta_p": float(rel_err_delta_p),
            "rel_err_y": float(rel_err_y),
            "corr_raw": float(corr_raw),
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
    print(f"type_S={type_S}, Nx={Nx}, CFL={CFL}, dt={float(dt):.3e}, nsteps={nsteps}")
    print(f"rel_l2_shift = {metrics['rel_l2_shift']:.4e}")
    print(f"corr_shifted = {metrics['corr_shifted']:.4f}")
    print(f"elapsed      = {metrics['elapsed']:.2f} s")

    return metrics


# ======================================================================================
# Ajout des ordres de convergence
# ======================================================================================

def add_orders(results):
    for r in results:
        r["order_rel_l2"] = None
        r["order_rel_l2_shift"] = None
        r["order_rel_linf"] = None

    for i in range(1, len(results)):
        e_prev = results[i - 1]["rel_l2"]
        e_curr = results[i]["rel_l2"]

        e_prev_shift = results[i - 1]["rel_l2_shift"]
        e_curr_shift = results[i]["rel_l2_shift"]

        e_prev_inf = results[i - 1]["rel_linf"]
        e_curr_inf = results[i]["rel_linf"]

        if e_curr > 0 and e_prev > 0:
            results[i]["order_rel_l2"] = float(
                jnp.log(e_prev / e_curr) / jnp.log(2.0)
            )

        if e_curr_shift > 0 and e_prev_shift > 0:
            results[i]["order_rel_l2_shift"] = float(
                jnp.log(e_prev_shift / e_curr_shift) / jnp.log(2.0)
            )

        if e_curr_inf > 0 and e_prev_inf > 0:
            results[i]["order_rel_linf"] = float(
                jnp.log(e_prev_inf / e_curr_inf) / jnp.log(2.0)
            )

    return results


# ======================================================================================
# Sauvegarde CSV
# ======================================================================================

def save_results_csv(results, output_dir, method, type_S):
    csv_path = os.path.join(
        output_dir,
        f"convergence_{method}_{type_S}.csv",
    )

    fieldnames = list(results[0].keys())

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print("CSV sauvegardé :", csv_path)


# ======================================================================================
# Figure RK2, erreur recalée normalisée pour comparer la pente
# ======================================================================================

def plot_shift_comparison(all_results, output_dir):
    plt.figure(figsize=(8, 5))

    styles = {
        "rk2": "s-",
    }
    colors = {
        "rk2": "tab:orange",
    }

    labels = {
        "rk2": "RK2",
    }

    markers = {
        "const": "s-",
        "exp": "o-",
    }

    for method, results_by_type in all_results.items():
        for type_S, results in results_by_type.items():

            Nx_values = np.array([r["Nx"] for r in results], dtype=float)
            h_values = 1.0 / Nx_values

            err_shift_raw = np.array(
                [r["rel_l2_shift"] for r in results],
                dtype=float,
            )

            valid_mask = np.isfinite(err_shift_raw) & (err_shift_raw > 0.0)
            if not np.any(valid_mask):
                print(f"Aucune erreur valide pour {method}/{type_S}, courbe ignorée.")
                continue

            h_values = h_values[valid_mask]
            err_shift = err_shift_raw[valid_mask]

            valid_orders = [
                r["order_rel_l2_shift"]
                for r in results
                if r["order_rel_l2_shift"] is not None
                and np.isfinite(r["order_rel_l2_shift"])
            ]

            order_label = " (pas de convergence)"
            if valid_orders:
                mean_order = float(np.mean(valid_orders))
                if mean_order > 0.0:
                    order_label = f" (ordre moyen={mean_order:.2f})"
                else:
                    order_label = f" (divergence, pente={mean_order:.2f})"

            plt.loglog(
                h_values,
                err_shift,
                markers.get(type_S, styles[method]),
                linewidth=2,
                label=f"{labels[method]} {type_S}{order_label}",
            )

            if len(h_values) > 0:
                expected_order = EXPECTED_ORDER[method]
                ref = err_shift[0] * (h_values / h_values[0]) ** expected_order

                plt.loglog(
                    h_values,
                    ref,
                    "--",
                    linewidth=1.0,
                    alpha=0.5,
                    label=f"reference {type_S} ordre {expected_order}",
                )

    plt.gca().invert_xaxis()
    plt.xlabel(r"$h \sim 1/N_x$")
    plt.ylabel("Erreur L2 relative après recalage")
    plt.title("Ordre de convergence RK2 DG vs OpenWind")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        "convergence_compare_shift_rk2.png",
    )

    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("Figure comparaison shift sauvegardée :", fig_path)


# ======================================================================================
# Main
# ======================================================================================

def main():
    args = parse_args()
    methods = ["rk2"]

    output_dir = "../experiments/convergence/results"
    os.makedirs(output_dir, exist_ok=True)

    with open("../experiments/convergence/config/simu.json", "r") as f:
        simu_params = json.load(f)

    solver_params = simu_params["solver_params"]

    CFL_ref = solver_params["cfl"]
    N_snapshot_time = min(solver_params["N_snapshot"], MAX_SNAPSHOTS)
    T_max = solver_params["T_max"]

    with open("../experiments/convergence/config/param.json", "r") as f:
        params = json.load(f)

    print("\n=== Vérification paramètres anche ===")
    print("left_bc_params =", params["left_bc_params"])

    cases = [{"Nx": Nx, "CFL": CFL_ref} for Nx in DG_NX_LIST]

    all_results = {method: {} for method in methods}

    for type_S in ["const", "exp"]:
        print("\n" + "=" * 80)
        print(f"Calcul complet pour type_S = {type_S}")
        print("=" * 80)

        print("\nCalcul de la référence OpenWind...")

        reference_data_for_mesh = build_physical_data(params, type_S)
        L_ref = reference_data_for_mesh.L_tube + reference_data_for_mesh.L_bell
        max_dg_nx = max(case["Nx"] for case in cases)
        h_min_dg = L_ref / max_dg_nx
        l_el = OPENWIND_REF_L_ELE[type_S]
        n_points = OPENWIND_REF_N_POINTS[type_S]
        if n_points is None:
            n_points = 2 * max_dg_nx + 1

        print("\n=== Référence OpenWind fine ===")
        print("L_ref       =", L_ref)
        print("max DG Nx   =", max_dg_nx)
        print("h_min DG    =", h_min_dg)
        print("N snapshots =", N_snapshot_time)
        print("OpenWind l_ele =", l_el)
        print("OpenWind estimated elements =", int(np.ceil(float(L_ref) / l_el)))
        print("OpenWind/DG h ratio =", float(l_el / h_min_dg))
        print("OpenWind order =", OPENWIND_REF_ORDER)
        print("OpenWind theta =", OPENWIND_REF_THETA)
        print("OpenWind geometry points =", n_points)
        print("OpenWind geometry h ratio =", float((L_ref / (n_points - 1)) / h_min_dg))

        cache_path = openwind_cache_path(
            output_dir,
            params,
            T_max,
            type_S,
            n_points,
            l_el,
            OPENWIND_REF_ORDER,
            OPENWIND_REF_THETA,
        )

        openwind_reference = None
        if CACHE_OPENWIND_REFERENCE:
            openwind_reference = load_openwind_reference(cache_path)
            if openwind_reference is not None:
                print("Référence OpenWind chargée depuis le cache :", cache_path)

        if openwind_reference is None:
            openwind_reference = run_openwind_reference(
                param_json=params,
                T_max=T_max,
                type_S=type_S,
                n_points=n_points,
                l_ele=l_el,
                order=OPENWIND_REF_ORDER,
                theta=OPENWIND_REF_THETA,
            )
            if CACHE_OPENWIND_REFERENCE:
                save_openwind_reference(cache_path, openwind_reference)
                print("Référence OpenWind mise en cache :", cache_path)

        ow_rad_params = openwind_reference[-1]

        params_dg = copy.deepcopy(params)

        reference_data = build_physical_data(params_dg, type_S)
        L = reference_data.L_tube + reference_data.L_bell

        S_star = reference_data.section(0.0)
        Zt = S_star / reference_data.section(L)

        params_dg["right_bc_params"]["Zt"] = float(Zt)

        if ow_rad_params["alpha"] is not None:
            params_dg["right_bc_params"]["alpha"] = float(ow_rad_params["alpha"])

        if ow_rad_params["beta"] is not None:
            params_dg["right_bc_params"]["beta"] = float(ow_rad_params["beta"])

        print("\n=== Paramètres radiation injectés dans DG ===")
        print("type_S =", type_S)
        print("Zt    =", params_dg["right_bc_params"]["Zt"])
        print("alpha =", params_dg["right_bc_params"]["alpha"])
        print("beta  =", params_dg["right_bc_params"]["beta"])

        for method in methods:
            results = []

            for case in cases:
                cfl_method = (
                    case["CFL"]
                    if METHOD_CFL[method] is None
                    else METHOD_CFL[method]
                )

                print("\n" + "-" * 80)
                print(
                    f"Étude de convergence : "
                    f"type_S={type_S}, Nx={case['Nx']}, CFL={cfl_method}, method={method}"
                )
                print("-" * 80)

                metrics = run_case(
                    Nx=case["Nx"],
                    CFL=cfl_method,
                    N_snapshot_time=N_snapshot_time,
                    method=method,
                    type_S=type_S,
                    params=params_dg,
                    T_max=T_max,
                    openwind_reference=openwind_reference,
                )

                results.append(metrics)

            results = add_orders(results)
            all_results[method][type_S] = results

            save_results_csv(
                results=results,
                output_dir=output_dir,
                method=method,
                type_S=type_S,
            )

    print("\n" + "=" * 80)
    print("Résumé convergence shift")
    print("=" * 80)

    for method, results_by_type in all_results.items():
        for type_S, results in results_by_type.items():
            print(f"\n--- {method} / {type_S} ---")
            for r in results:
                print(
                    f"Nx={r['Nx']:4d} | "
                    f"rel_l2_shift={r['rel_l2_shift']:.4e} | "
                    f"order_shift={r['order_rel_l2_shift']} | "
                    f"elapsed={r['elapsed']:.2f}s"
                )

    plot_shift_comparison(
        all_results=all_results,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
