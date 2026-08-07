import copy
import csv
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import (
    cell_edges_from_nodes,
    create_uniform_nodes_with_ghosts,
)
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.parse_args import parse_args
from utils.res_openwind import run_openwind_reference
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)


P_CLOSED = 5e3
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

SCAN_PARAMS = (
    "kappa",
    "gamma_final",
    "zeta",
    "Qr",
)

PARAM_LABELS = {
    "kappa": r"$\kappa$",
    "gamma_final": r"$\gamma$",
    "zeta": r"$\zeta$",
    "Qr": r"$Q_r$",
}

PARAM_JSON_PATHS = {
    "alpha": ("right_bc_params", "alpha"),
    "beta": ("right_bc_params", "beta"),
    "Zt": ("right_bc_params", "Zt"),
    "kappa": ("left_bc_params", "kappa"),
    "gamma_final": (
        "left_bc_params",
        "mouth_pressure_params",
        "gamma_final",
    ),
    "zeta": ("left_bc_params", "zeta"),
    "Qr": ("left_bc_params", "Qr"),
}

SCAN_RANGES = {
    "kappa": (0.1, 2.0),
    "gamma_final": (0.1, 0.9),
    "zeta": (0.1, 0.9),
    "Qr": (20.0, 500.0),
}


def get_nested(mapping, path):
    value = mapping
    for key in path:
        value = value[key]
    return value


def set_nested(mapping, path, value):
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = float(value)


def value_from_params(params, name):
    return float(get_nested(params, PARAM_JSON_PATHS[name]))


def repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_project_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path

    parts = path.parts
    if (
        len(parts) >= 2
        and parts[0] == ".."
        and parts[1] == "experiments"
    ):
        return root.joinpath(*parts[1:])

    return root / path


def parse_stft_resolutions(value):
    resolutions = []

    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            n_fft, hop = item.split(":", maxsplit=1)
        elif "x" in item:
            n_fft, hop = item.split("x", maxsplit=1)
        else:
            raise ValueError(
                "Chaque résolution STFT doit être au format "
                f"n_fft:hop, reçu : {item}"
            )

        n_fft = int(n_fft)
        hop = int(hop)

        if n_fft <= 0 or hop <= 0:
            raise ValueError(
                f"Résolution STFT invalide : {item}"
            )

        resolutions.append((n_fft, hop))

    if not resolutions:
        raise ValueError(
            "La liste --scan_stft_resolutions est vide."
        )

    return tuple(resolutions)


def parse_float_list(value, name):
    values = []

    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    if not values:
        raise ValueError(
            f"La liste --{name} est vide."
        )

    return tuple(values)


def make_solver_data(
    T_max,
    CFL,
    Nx,
    N_snapshot,
    L_ref,
    c,
    bc,
    phi0,
    y0,
    z0,
):
    x_nodes, _ = create_uniform_nodes_with_ghosts(
        Nx,
        0.0,
        L_ref,
    )
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
    n_snaps = jnp.round(
        jnp.linspace(
            0,
            nsteps - 1,
            N_snapshot,
        )
    ).astype(jnp.int32)

    solve_kwargs = dict(
        dt=dt,
        nsteps=nsteps,
        bc=bc,
        phi0=phi0,
        y0=y0,
        z0=z0,
        t_solver=t_solver,
        n_snaps=n_snaps,
    )

    return dt, nsteps, solve_kwargs


def params_with_openwind_radiation(
    base_params,
    ow_params,
    type_S,
):
    params = copy.deepcopy(base_params)

    if ow_params.get("alpha") is not None:
        set_nested(
            params,
            PARAM_JSON_PATHS["alpha"],
            ow_params["alpha"],
        )

    if ow_params.get("beta") is not None:
        set_nested(
            params,
            PARAM_JSON_PATHS["beta"],
            ow_params["beta"],
        )

    data = build_physical_data(params, type_S)
    length = (
        data.section.L_tube
        + data.section.L_bell
    )

    set_nested(
        params,
        PARAM_JSON_PATHS["Zt"],
        float(
            data.section(0.0)
            / data.section(length)
        ),
    )

    return params


def make_openwind_pressure_target(
    params_true,
    type_S,
    T_max,
    snapshot_times,
    args,
):
    ow_l_ele = (
        args.ow_l_ele
        if args.ow_l_ele is not None
        else 5.0e-4
    )

    (
        t_ow,
        _p_left,
        p_right,
        _y_ow,
        _gamma_ow,
        ow_params,
    ) = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=ow_l_ele,
        order=args.ow_order,
    )

    snapshot_times_np = np.asarray(
        snapshot_times,
        dtype=float,
    )
    t_ow_np = np.asarray(
        t_ow,
        dtype=float,
    )

    target_p = np.interp(
        snapshot_times_np,
        t_ow_np,
        np.asarray(
            p_right,
            dtype=float,
        ),
    )

    target_p = jnp.asarray(
        target_p / P_CLOSED,
        dtype=jnp.float64,
    )

    return target_p, ow_params


def make_target_for_source(
    source,
    params,
    geometry,
    c,
    solve_kwargs,
    T_max,
    snapshot_times,
    args,
):
    if source == "openwind":
        target_p, ow_params = (
            make_openwind_pressure_target(
                params,
                args.type_S,
                T_max,
                snapshot_times,
                args,
            )
        )

        params_dg = params_with_openwind_radiation(
            params,
            ow_params,
            args.type_S,
        )

        print(
            f"OpenWind alpha={ow_params.get('alpha')}"
        )
        print(
            f"OpenWind beta={ow_params.get('beta')}"
        )

        return target_p, params_dg

    data_true = build_physical_data(
        params,
        args.type_S,
    )

    target_p = forward_snapshots(
        data_true,
        geometry,
        c,
        **solve_kwargs,
    )

    return target_p, copy.deepcopy(params)


def make_scan_loss(
    param_name,
    data_base,
    geometry,
    c,
    target_p,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    """
    Retourne uniquement la loss MSTS sur la pression au pavillon.
    """

    def loss(value):
        data = set_param(
            data_base,
            param_name,
            value,
            GEO_KEYS,
        )

        pred_p = forward_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )

        return loss_fn_signal(
            pred_p,
            target_p,
            time_weight=0.0,
            spec_weight=1.0,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    return jax.jit(loss)


def write_summary_csv(
    summary,
    output_dir,
):
    csv_path = os.path.join(
        output_dir,
        "scan_1D_MSTS_4params_summary.csv",
    )

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(summary[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary)

    return csv_path


def save_all_scans_npz(
    output_dir,
    source,
    t_max_values,
    scan_results,
):
    path = os.path.join(
        output_dir,
        f"scan_1D_MSTS_4params_{source}.npz",
    )

    arrays = {
        "t_max_values": np.asarray(
            t_max_values,
            dtype=float,
        ),
    }

    for param_name, result in scan_results.items():
        arrays[f"{param_name}_values"] = result["values"]
        arrays[f"{param_name}_losses"] = result["losses"]
        arrays[f"{param_name}_true_values"] = result["true_values"]
        arrays[f"{param_name}_min_values"] = result["min_values"]
        arrays[f"{param_name}_min_losses"] = result["min_losses"]

    np.savez(
        path,
        **arrays,
    )

    return path


def plot_all_scans_2x2(
    output_dir,
    source,
    t_max_values,
    scan_results,
):
    path = os.path.join(
        output_dir,
        f"scan_1D_MSTS_4params_{source}.png",
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.0, 8.5),
    )
    axes = axes.ravel()

    colors = plt.cm.viridis(
        np.linspace(
            0.0,
            1.0,
            len(t_max_values),
        )
    )

    for ax, param_name in zip(
        axes,
        SCAN_PARAMS,
    ):
        result = scan_results[param_name]
        values = result["values"]
        losses_all = result["losses"]
        true_values = result["true_values"]
        min_values = result["min_values"]
        min_losses = result["min_losses"]

        for t_idx, T_max in enumerate(
            t_max_values
        ):
            losses = losses_all[t_idx]
            true_value = true_values[t_idx]
            min_value = min_values[t_idx]
            min_loss = min_losses[t_idx]
            color = colors[t_idx]

            ax.plot(
                values,
                losses,
                linewidth=1.8,
                color=color,
                label=rf"$T={T_max:.3f}$ s",
            )

            ax.axvline(
                true_value,
                color=color,
                linestyle="--",
                linewidth=1.2,
                alpha=0.7,
            )

            ax.axvline(
                min_value,
                color=color,
                linestyle=":",
                linewidth=1.5,
                alpha=0.95,
            )

            ax.scatter(
                [min_value],
                [min_loss],
                color=color,
                s=35,
                zorder=4,
            )

        ax.set_xlabel(
            PARAM_LABELS[param_name]
        )
        ax.set_ylabel(
            r"$\mathcal{L}_{\mathrm{MSTS}}$"
        )
        ax.set_title(
            f"Scan de {PARAM_LABELS[param_name]}"
        )
        ax.set_yscale("log")
        ax.grid(
            True,
            alpha=0.3,
        )
        ax.legend(
            fontsize=8,
        )

    fig.suptitle(
        (
            "Scans 1D de la loss MSTS "
            f"— cible {source.upper()}"
        ),
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.96)
    )
    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

    return path


def main():
    args = parse_args()
    root = repo_root()

    output_dir = resolve_project_path(
        root,
        args.scan_output_dir,
    )
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    stft_resolutions = (
        parse_stft_resolutions(
            args.scan_stft_resolutions
        )
    )
    stft_allow_padding = (
        not args.scan_no_stft_padding
    )
    t_max_values = parse_float_list(
        args.scan_t_max_values,
        "scan_t_max_values",
    )

    with open(
        root
        / "experiments/gradient/config/simu.json",
        "r",
    ) as file:
        solver_params = json.load(file)[
            "solver_params"
        ]

    with open(
        root
        / "experiments/gradient/config/param.json",
        "r",
    ) as file:
        params = json.load(file)

    train_params = solver_params["train"]

    scan_nx = (
        args.scan_nx
        if args.scan_nx is not None
        else train_params["Nx"]
    )
    scan_n_snapshot = (
        args.scan_n_snapshot
        if args.scan_n_snapshot is not None
        else train_params["N_snapshot"]
    )

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"][
        "y_dot0"
    ]

    data_ref = build_physical_data(
        params,
        args.type_S,
    )
    L_ref = (
        data_ref.section.L_tube
        + data_ref.section.L_bell
    )

    geometry = build_solver_geometry(
        data_ref,
        scan_nx,
        c,
    )
    bc = BC(type="full")

    if args.scan_target_source == "both":
        target_sources = (
            "dg",
            "openwind",
        )
    else:
        target_sources = (
            args.scan_target_source,
        )

    print(
        "Loss utilisée : MSTS pression uniquement"
    )
    print(
        "Paramètres scannés : "
        + ", ".join(SCAN_PARAMS)
    )
    print(
        f"Nx={scan_nx}, "
        f"N_snapshot={scan_n_snapshot}"
    )
    print(
        f"Résolutions STFT : "
        f"{stft_resolutions}"
    )
    print(
        f"T_max : {t_max_values}"
    )

    summary = []

    for source in target_sources:
        print(
            "\n"
            + "#" * 80
        )
        print(
            f"ÉTUDE {source.upper()} / DG"
        )
        print(
            "#" * 80
        )

        source_cases = []

        for T_max in t_max_values:
            (
                dt,
                nsteps,
                solve_kwargs,
            ) = make_solver_data(
                T_max,
                train_params["cfl"],
                scan_nx,
                scan_n_snapshot,
                L_ref,
                c,
                bc,
                phi0,
                y0,
                z0,
            )

            snapshot_times = (
                solve_kwargs["n_snaps"] + 1
            ) * solve_kwargs["dt"]

            print(
                f"\nPréparation cible {source}/DG : "
                f"T_max={T_max:.4f}, "
                f"dt={dt:.6e}, "
                f"nsteps={nsteps}"
            )

            (
                target_p,
                params_dg,
            ) = make_target_for_source(
                source,
                params,
                geometry,
                c,
                solve_kwargs,
                T_max,
                snapshot_times,
                args,
            )

            source_cases.append(
                {
                    "T_max": T_max,
                    "solve_kwargs": solve_kwargs,
                    "target_p": target_p,
                    "params_dg": params_dg,
                }
            )

        scan_results = {}

        for param_name in SCAN_PARAMS:
            vmin, vmax = SCAN_RANGES[
                param_name
            ]

            values = jnp.linspace(
                vmin,
                vmax,
                args.scan_n,
            )
            values_np = np.asarray(
                values,
                dtype=float,
            )

            print(
                "\n"
                + "=" * 80
            )
            print(
                f"SCAN 1D MSTS [{source}/DG] : "
                f"{param_name}"
            )
            print(
                f"range=[{vmin:.6g}, {vmax:.6g}], "
                f"n={args.scan_n}"
            )
            print(
                "=" * 80
            )

            losses_all = []
            true_values = []
            min_values = []
            min_losses = []

            for case in source_cases:
                T_max = case["T_max"]
                true_value = value_from_params(
                    case["params_dg"],
                    param_name,
                )

                params_scan = copy.deepcopy(
                    case["params_dg"]
                )

                for name in params_scan["trainable"]:
                    params_scan["trainable"][name] = (
                        name == param_name
                    )

                data_base = build_physical_data(
                    params_scan,
                    args.type_S,
                )

                scan_loss = make_scan_loss(
                    param_name,
                    data_base,
                    geometry,
                    c,
                    case["target_p"],
                    case["solve_kwargs"],
                    stft_resolutions,
                    args.scan_stft_dynamic_db,
                    stft_allow_padding,
                )

                losses = []

                print(
                    f"\nT_max={T_max:.4f}, "
                    f"vrai={true_value:.6g}"
                )

                for idx, value in enumerate(
                    values
                ):
                    loss_device = scan_loss(value)
                    loss_device.block_until_ready()
                    loss_value = float(
                        jax.device_get(
                            loss_device
                        )
                    )

                    losses.append(
                        loss_value
                    )

                    if (
                        idx < 3
                        or idx
                        % args.scan_print_every
                        == 0
                        or idx
                        == args.scan_n - 1
                    ):
                        print(
                            f"{idx:>4}/"
                            f"{args.scan_n - 1:<4} | "
                            f"{param_name}="
                            f"{float(value):.6g} | "
                            f"L_MSTS="
                            f"{loss_value:.6e}"
                        )

                losses = np.asarray(
                    losses,
                    dtype=float,
                )

                idx_min = int(
                    np.argmin(losses)
                )
                value_min = float(
                    values_np[idx_min]
                )
                loss_min = float(
                    losses[idx_min]
                )
                rel_error_min = (
                    abs(
                        value_min
                        - true_value
                    )
                    / max(
                        abs(true_value),
                        1e-12,
                    )
                )

                losses_all.append(
                    losses
                )
                true_values.append(
                    true_value
                )
                min_values.append(
                    value_min
                )
                min_losses.append(
                    loss_min
                )

                summary.append(
                    {
                        "target_source": source,
                        "T_max": T_max,
                        "param": param_name,
                        "true": true_value,
                        "MSTS_min": value_min,
                        "MSTS_rel_error_min": (
                            rel_error_min
                        ),
                        "MSTS_loss_min": loss_min,
                        "range_min": vmin,
                        "range_max": vmax,
                        "n_scan": args.scan_n,
                    }
                )

                print(
                    f"Minimum : "
                    f"{param_name}={value_min:.6g}, "
                    f"L_MSTS={loss_min:.6e}, "
                    f"erreur relative="
                    f"{100.0 * rel_error_min:.3f}%"
                )

            scan_results[param_name] = {
                "values": values_np,
                "losses": np.asarray(
                    losses_all,
                    dtype=float,
                ),
                "true_values": np.asarray(
                    true_values,
                    dtype=float,
                ),
                "min_values": np.asarray(
                    min_values,
                    dtype=float,
                ),
                "min_losses": np.asarray(
                    min_losses,
                    dtype=float,
                ),
            }

        figure_path = plot_all_scans_2x2(
            output_dir,
            source,
            t_max_values,
            scan_results,
        )

        data_path = save_all_scans_npz(
            output_dir,
            source,
            t_max_values,
            scan_results,
        )

        print(
            "\nFigure 2x2 sauvegardée : "
            f"{figure_path}"
        )
        print(
            "Données sauvegardées : "
            f"{data_path}"
        )

    summary_csv = write_summary_csv(
        summary,
        output_dir,
    )

    print(
        "\n=== Résumé des scans ==="
    )

    for item in summary:
        print(
            f"{item['target_source']:>8}/DG | "
            f"T={item['T_max']:.4f} | "
            f"{item['param']:>12} | "
            f"vrai={item['true']:.6g} | "
            f"minimum={item['MSTS_min']:.6g} | "
            f"erreur="
            f"{100.0 * item['MSTS_rel_error_min']:.3f}%"
        )

    print(
        "CSV résumé : "
        f"{summary_csv}"
    )


if __name__ == "__main__":
    main()