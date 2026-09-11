import csv
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.parse_args import parse_args
from utils.res_openwind import run_openwind_reference

def best_time_shift_np(x, y, dt):
    x0 = x - np.mean(x)
    y0 = y - np.mean(y)

    corr_full = np.correlate(x0, y0, mode="full")
    lag_index = int(np.argmax(corr_full) - (len(y0) - 1))

    corr = float(
        np.max(corr_full)
        / (np.linalg.norm(x0) * np.linalg.norm(y0) + 1e-12)
    )

    return lag_index, lag_index * dt, corr


def compute_metrics_np(p_coarse, p_fine, t_coarse):
    dt_snap = float(t_coarse[1] - t_coarse[0])

    rel_l2 = np.linalg.norm(p_coarse - p_fine) / (
        np.linalg.norm(p_fine) + 1e-12
    )

    lag_index, shift_time, corr = best_time_shift_np(
        p_coarse,
        p_fine,
        dt_snap,
    )

    if lag_index > 0:
        p_coarse_s = p_coarse[lag_index:]
        p_fine_s = p_fine[:-lag_index]
    elif lag_index < 0:
        k = -lag_index
        p_coarse_s = p_coarse[:-k]
        p_fine_s = p_fine[k:]
    else:
        p_coarse_s = p_coarse
        p_fine_s = p_fine

    rel_l2_shift = np.linalg.norm(p_coarse_s - p_fine_s) / (
        np.linalg.norm(p_fine_s) + 1e-12
    )

    return {
        "rel_l2": rel_l2,
        "rel_l2_shift": rel_l2_shift,
        "corr_shifted": corr,
        "lag_index": lag_index,
        "shift_time": shift_time,
    }


def plot_openwind_convergence(results, output_dir, type_S):
    h = np.array([r["h"] for r in results], dtype=float)

    err = np.array(
        [r["rel_l2_shift"] for r in results],
        dtype=float,
    )

    plt.figure(figsize=(7, 5))

    plt.loglog(h, err, "o-", linewidth=2, label="Erreur OpenWind")

    ref1 = err[0] * (h / h[0])**1
    ref2 = err[0] * (h / h[0])**2

    plt.loglog(h, ref1, "--", label="ordre 1")
    plt.loglog(h, ref2, "--", label="ordre 2")

    plt.gca().invert_xaxis()
    plt.xlabel(r"$h_{\mathrm{eff}} = L/N_{\mathrm{ele}}$")
    plt.ylabel("Erreur relative L2 recalée")
    plt.title(f"Convergence OpenWind ({type_S})")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        f"openwind_convergence_{type_S}.png",
    )

    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Figure sauvegardée :", fig_path)


def run_openwind_convergence(
    params,
    T_max,
    type_S,
    l_ele_list,
    output_dir,
    n_points=200,
    order=4,
    theta=0.5,
):
    os.makedirs(output_dir, exist_ok=True)

    solutions = []

    for l_ele in l_ele_list:
        print("\n" + "=" * 80)
        print(f"OpenWind : type_S={type_S}, l_ele={l_ele}, order={order}")
        print("=" * 80)

        t, p_left, p_right, y, gamma, rad_params = run_openwind_reference(
            param_json=params,
            T_max=T_max,
            type_S=type_S,
            n_points=n_points,
            l_ele=l_ele,
            order=order,
            theta=theta,
        )

        solutions.append({
            "l_ele": float(l_ele),
            "n_elements": int(rad_params["n_elements_total"]),
            "h_eff": float(rad_params["h_eff"]),
            "dt": float(rad_params["dt"]),
            "t": np.asarray(t),
            "p": np.asarray(p_right),
        })

    results = []

    for k in range(len(solutions) - 1):
        coarse = solutions[k]
        fine = solutions[k + 1]

        t_coarse = coarse["t"]
        p_coarse = coarse["p"]

        common = (t_coarse >= fine["t"][0]) & (t_coarse <= fine["t"][-1])
        t_coarse = t_coarse[common]
        p_coarse = p_coarse[common]

        p_fine_interp = np.interp(
            t_coarse,
            fine["t"],
            fine["p"],
        )

        metrics = compute_metrics_np(
            p_coarse,
            p_fine_interp,
            t_coarse,
        )

        results.append({
            "l_ele": coarse["l_ele"],
            "l_ele_ref": fine["l_ele"],
            "n_elements": coarse["n_elements"],
            "n_elements_ref": fine["n_elements"],
            "h": coarse["h_eff"],
            "rel_l2": float(metrics["rel_l2"]),
            "rel_l2_shift": float(metrics["rel_l2_shift"]),
            "corr_shifted": float(metrics["corr_shifted"]),
            "lag_index": int(metrics.get("lag_index", 0)),
            "shift_time": float(metrics.get("shift_time", 0.0)),
            "dt": coarse["dt"],
            "order_shift": None,
        })

    for i in range(1, len(results)):
        e_prev = results[i - 1]["rel_l2_shift"]
        e_curr = results[i]["rel_l2_shift"]

        h_prev = results[i - 1]["h"]
        h_curr = results[i]["h"]

        if e_prev > 0 and e_curr > 0 and h_prev > 0 and h_curr > 0:
            results[i]["order_shift"] = float(
                np.log(e_prev / e_curr) / np.log(h_prev / h_curr)
            )

    csv_path = os.path.join(
        output_dir,
        f"openwind_convergence_{type_S}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nCSV sauvegardé :", csv_path)

    for r in results:
        print(
            f"l_ele={r['l_ele']:.4e} -> {r['l_ele_ref']:.4e} | "
            f"err_shift={r['rel_l2_shift']:.4e} | "
            f"order={r['order_shift']}"
        )

    plot_openwind_convergence(results, output_dir, type_S)


def main():
    args = parse_args()
    output_dir = "../experiments/convergence/conv_ow"

    with open("../experiments/convergence/config/simu.json", "r") as f:
        simu_params = json.load(f)

    #T_max = simu_params["solver_params"]["T_max"]
    T_max = 0.05

    with open("../experiments/convergence/config/param.json", "r") as f:
        params = json.load(f)

    for type_S in ["const", "exp"]:
        n_points = 2 if type_S == "const" else 200
        l_ele_list = (
            [0.0025, 0.00125, 0.000625, 0.0003125]
            if type_S == "const"
            else [0.003, 0.0015, 0.0007, 0.00033]
        )
        run_openwind_convergence(
            params=params,
            T_max=T_max,
            type_S=type_S,
            l_ele_list=l_ele_list,
            output_dir=output_dir,
            n_points=n_points,
            order=4,
            theta=0.5,
        )


if __name__ == "__main__":
    main()
