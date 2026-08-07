import copy
import csv
import gc
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
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
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

PARAM_JSON_PATHS = {
    "epsilon": ("left_bc_params", "epsilon"),
    "beta": ("right_bc_params", "beta"),
    "alpha": ("right_bc_params", "alpha"),
    "zeta": ("left_bc_params", "zeta"),
    "kappa": ("left_bc_params", "kappa"),
    "Zt": ("right_bc_params", "Zt"),
    "fr": ("left_bc_params", "fr"),
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "t_attack": ("left_bc_params", "mouth_pressure_params", "t_attack"),
    "L_tube": ("instrument_geometry", "tube", "L_tube"),
    "R_tube": ("instrument_geometry", "tube", "R_tube"),
    "L_bell": ("instrument_geometry", "bell", "L_bell"),
    "k_bell": ("instrument_geometry", "bell", "k_bell"),
    "Qr": ("left_bc_params", "Qr"),
}

SCAN_RANGES = {
    "alpha": (6.0e4, 1.0e5),
    "beta": (0.1, 1.0),
    "Zt": (0.2, 3.0),
    "kappa": (0.1, 2.0),
    "fr": (50.0, 260.0),
    "Qr": (20.0, 250.0),
    "gamma_final": (0.1, 0.9),
    "zeta": (0.1, 0.9),
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
    if len(parts) >= 2 and parts[0] == ".." and parts[1] == "experiments":
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
                "Chaque resolution STFT doit etre au format n_fft:hop, "
                f"recu: {item}"
            )

        n_fft = int(n_fft)
        hop = int(hop)
        if n_fft <= 0 or hop <= 0:
            raise ValueError(f"Resolution STFT invalide: {item}")

        resolutions.append((n_fft, hop))

    if len(resolutions) == 0:
        raise ValueError("La liste --scan_stft_resolutions est vide.")

    return tuple(resolutions)


def parse_float_list(value, name):
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    if len(values) == 0:
        raise ValueError(f"La liste --{name} est vide.")

    return tuple(values)


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
    n_snaps = jnp.round(jnp.linspace(0, nsteps - 1, N_snapshot)).astype(jnp.int32)

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


def params_with_openwind_radiation(base_params, ow_params, type_S):
    params = copy.deepcopy(base_params)

    if ow_params.get("alpha") is not None:
        set_nested(params, PARAM_JSON_PATHS["alpha"], ow_params["alpha"])
    if ow_params.get("beta") is not None:
        set_nested(params, PARAM_JSON_PATHS["beta"], ow_params["beta"])

    data = build_physical_data(params, type_S)
    length = data.section.L_tube + data.section.L_bell
    set_nested(
        params,
        PARAM_JSON_PATHS["Zt"],
        float(data.section(0.0) / data.section(length)),
    )

    return params



def make_openwind_target(params_true, type_S, T_max, snapshot_times, args):
    """Construit uniquement la cible de pression OpenWind."""
    ow_l_ele = args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4

    t_ow, _, p_right, _, _, ow_params = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=ow_l_ele,
        order=args.ow_order,
    )

    target_p = np.interp(
        np.asarray(snapshot_times, dtype=float),
        np.asarray(t_ow, dtype=float),
        np.asarray(p_right, dtype=float),
    )

    return (
        jnp.asarray(target_p / P_CLOSED, dtype=jnp.float64),
        ow_params,
    )


def make_target_for_source(
    source,
    params,
    geometry,
    c,
    solve_kwargs,
    train_params,
    snapshot_times,
    args,
):
    if source == "openwind":
        target_p, ow_params = make_openwind_target(
            params,
            args.type_S,
            train_params["T_max"],
            snapshot_times,
            args,
        )
        params_dg = params_with_openwind_radiation(
            params,
            ow_params,
            args.type_S,
        )
        print(f"OpenWind alpha={ow_params.get('alpha')}")
        print(f"OpenWind beta={ow_params.get('beta')}")
        return target_p, params_dg

    data_true = build_physical_data(params, args.type_S)
    target_p = forward_snapshots(
        data_true,
        geometry,
        c,
        **solve_kwargs,
    )
    return target_p, copy.deepcopy(params)


def make_scan_loss_2d(
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
    Construit uniquement la loss MSTS de pression :

        L_MSTS(kappa, Qr)

    Tous les autres paramètres physiques restent fixés.
    """

    def loss_one(kappa_value, qr_value):
        data = set_param(
            data_base,
            "kappa",
            kappa_value,
            GEO_KEYS,
        )
        data = set_param(
            data,
            "Qr",
            qr_value,
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

    @jax.jit
    def loss_row(kappa_values, qr_value):
        """
        Calcule une ligne complète de la grille.

        lax.map conserve une exécution séquentielle sur les valeurs de kappa,
        contrairement à vmap qui pourrait lancer toutes les simulations
        simultanément et augmenter fortement la mémoire GPU.
        """
        return jax.lax.map(
            lambda kappa_value: loss_one(kappa_value, qr_value),
            kappa_values,
        )

    return loss_row


def safe_log10(values, floor=1e-12):
    return np.log10(np.maximum(values, floor))


def plot_loss_map(
    kappa_values,
    qr_values,
    losses,
    true_kappa,
    true_qr,
    min_kappa,
    min_qr,
    title,
    path,
    use_log=False,
):
    displayed = safe_log10(losses) if use_log else losses
    colorbar_label = (
        r"$\log_{10}(\mathcal{L}_{\mathrm{MSTS}})$"
        if use_log
        else r"$\mathcal{L}_{\mathrm{MSTS}}$"
    )

    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    contour = ax.contourf(
        kappa_values,
        qr_values,
        displayed,
        levels=40,
    )
    fig.colorbar(contour, ax=ax, label=colorbar_label)

    ax.axvline(
        true_kappa,
        linestyle="--",
        linewidth=1.5,
        label=r"Vrai $\kappa$",
    )
    ax.axhline(
        true_qr,
        linestyle="--",
        linewidth=1.5,
        label=r"Vrai $Q_r$",
    )
    ax.scatter(
        [true_kappa],
        [true_qr],
        marker="x",
        s=90,
        linewidths=2.2,
        label="Vrai couple",
    )
    ax.scatter(
        [min_kappa],
        [min_qr],
        marker="o",
        s=55,
        label="Minimum de la grille",
    )

    ax.set_xlabel(r"$\kappa$")
    ax.set_ylabel(r"$Q_r$")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)



def plot_2d_diagnostics(
    kappa_values,
    qr_values,
    losses,
    true_kappa,
    true_qr,
    min_kappa,
    min_qr,
    title,
    path,
):
    """
    Génère une figure de diagnostic contenant :

    1. la carte 2D de log10(L_MSTS),
    2. la coupe selon kappa au Qr du minimum,
    3. la coupe selon Qr au kappa du minimum.

    Les coupes rendent directement visible la forte sensibilité à kappa
    et la faible sensibilité éventuelle à Qr.
    """
    losses = np.asarray(losses, dtype=float)
    displayed = safe_log10(losses)

    i_qr_min = int(np.argmin(np.abs(qr_values - min_qr)))
    i_kappa_min = int(np.argmin(np.abs(kappa_values - min_kappa)))

    loss_kappa_cut = losses[i_qr_min, :]
    loss_qr_cut = losses[:, i_kappa_min]

    fig = plt.figure(figsize=(14.0, 9.0))
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.45, 1.0),
        hspace=0.32,
        wspace=0.28,
    )

    # ------------------------------------------------------------------
    # Carte 2D
    # ------------------------------------------------------------------
    ax_map = fig.add_subplot(grid[0, :])
    contour = ax_map.contourf(
        kappa_values,
        qr_values,
        displayed,
        levels=50,
    )
    contour_lines = ax_map.contour(
        kappa_values,
        qr_values,
        displayed,
        levels=12,
        linewidths=0.6,
    )
    ax_map.clabel(
        contour_lines,
        inline=True,
        fontsize=7,
        fmt="%.2f",
    )
    fig.colorbar(
        contour,
        ax=ax_map,
        label=r"$\log_{10}(\mathcal{L}_{\mathrm{MSTS}})$",
    )

    ax_map.axvline(
        true_kappa,
        linestyle="--",
        linewidth=1.5,
        label=rf"Vrai $\kappa={true_kappa:.4g}$",
    )
    ax_map.axhline(
        true_qr,
        linestyle="--",
        linewidth=1.5,
        label=rf"Vrai $Q_r={true_qr:.4g}$",
    )
    ax_map.scatter(
        [true_kappa],
        [true_qr],
        marker="x",
        s=100,
        linewidths=2.4,
        label="Vrai couple",
        zorder=5,
    )
    ax_map.scatter(
        [min_kappa],
        [min_qr],
        marker="o",
        s=70,
        label=(
            rf"Minimum : $\kappa={min_kappa:.4g}$, "
            rf"$Q_r={min_qr:.4g}$"
        ),
        zorder=5,
    )

    ax_map.set_xlabel(r"$\kappa$")
    ax_map.set_ylabel(r"$Q_r$")
    ax_map.set_title(title)
    ax_map.grid(True, alpha=0.2)
    ax_map.legend(loc="best")

    # ------------------------------------------------------------------
    # Coupe suivant kappa
    # ------------------------------------------------------------------
    ax_kappa = fig.add_subplot(grid[1, 0])
    ax_kappa.plot(
        kappa_values,
        loss_kappa_cut,
        marker="o",
        markersize=3,
        linewidth=1.4,
    )
    ax_kappa.axvline(
        true_kappa,
        linestyle="--",
        linewidth=1.3,
        label=rf"Vrai $\kappa={true_kappa:.4g}$",
    )
    ax_kappa.axvline(
        min_kappa,
        linestyle=":",
        linewidth=1.5,
        label=rf"Minimum $\kappa={min_kappa:.4g}$",
    )
    ax_kappa.set_xlabel(r"$\kappa$")
    ax_kappa.set_ylabel(r"$\mathcal{L}_{\mathrm{MSTS}}$")
    ax_kappa.set_title(
        rf"Coupe selon $\kappa$ à $Q_r={qr_values[i_qr_min]:.4g}$"
    )
    ax_kappa.set_yscale("log")
    ax_kappa.grid(True, alpha=0.25)
    ax_kappa.legend()

    # ------------------------------------------------------------------
    # Coupe suivant Qr
    # ------------------------------------------------------------------
    ax_qr = fig.add_subplot(grid[1, 1])
    ax_qr.plot(
        qr_values,
        loss_qr_cut,
        marker="o",
        markersize=3,
        linewidth=1.4,
    )
    ax_qr.axvline(
        true_qr,
        linestyle="--",
        linewidth=1.3,
        label=rf"Vrai $Q_r={true_qr:.4g}$",
    )
    ax_qr.axvline(
        min_qr,
        linestyle=":",
        linewidth=1.5,
        label=rf"Minimum $Q_r={min_qr:.4g}$",
    )
    ax_qr.set_xlabel(r"$Q_r$")
    ax_qr.set_ylabel(r"$\mathcal{L}_{\mathrm{MSTS}}$")
    ax_qr.set_title(
        rf"Coupe selon $Q_r$ à $\kappa={kappa_values[i_kappa_min]:.4g}$"
    )
    ax_qr.set_yscale("log")
    ax_qr.grid(True, alpha=0.25)
    ax_qr.legend()

    fig.tight_layout()
    fig.savefig(
        path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def write_summary_csv(rows, output_dir):
    path = os.path.join(
        output_dir,
        "scan_2D_kappa_Qr_MSTS_summary.csv",
    )
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def main():
    args = parse_args()
    root = repo_root()

    output_dir = resolve_project_path(
        root,
        args.scan_output_dir,
    )
    os.makedirs(output_dir, exist_ok=True)

    stft_resolutions = parse_stft_resolutions(
        args.scan_stft_resolutions
    )
    stft_allow_padding = not args.scan_no_stft_padding
    t_max_values = parse_float_list(
        args.scan_t_max_values,
        "scan_t_max_values",
    )

    n_kappa = int(args.scan_n)
    n_qr = int(args.scan_n)
    if n_kappa <= 1 or n_qr <= 1:
        raise ValueError("--scan_n doit être supérieur à 1.")

    # Le premier intervalle de --scan_range contrôle kappa.
    # Pour Qr, on utilise la plage physique définie dans SCAN_RANGES.
    if args.scan_range is None:
        kappa_min, kappa_max = SCAN_RANGES["kappa"]
    else:
        parts = [
            item.strip()
            for item in args.scan_range.split(",")
        ]
        if len(parts) != 2:
            raise ValueError(
                "--scan_range doit être au format min,max "
                "et contrôle ici la plage de kappa."
            )
        kappa_min, kappa_max = map(float, parts)

    # Option facultative : --scan_range_qr min,max, si elle existe
    # dans parse_args. Sinon, plage par défaut.
    scan_range_qr = getattr(args, "scan_range_qr", None)
    if scan_range_qr is None:
        qr_min, qr_max = SCAN_RANGES["Qr"]
    else:
        parts = [
            item.strip()
            for item in scan_range_qr.split(",")
        ]
        if len(parts) != 2:
            raise ValueError(
                "--scan_range_qr doit être au format min,max."
            )
        qr_min, qr_max = map(float, parts)

    with open(
        root / "experiments/gradient/config/simu.json",
        "r",
    ) as file:
        solver_params = json.load(file)["solver_params"]

    with open(
        root / "experiments/gradient/config/param.json",
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
    z0 = params["init_cond_reed"]["y_dot0"]

    data_ref = build_physical_data(
        params,
        args.type_S,
    )
    length = (
        data_ref.section.L_tube
        + data_ref.section.L_bell
    )
    geometry = build_solver_geometry(
        data_ref,
        scan_nx,
        c,
    )
    bc = BC(type="full")

    target_sources = (
        ["dg", "openwind"]
        if args.scan_target_source == "both"
        else [args.scan_target_source]
    )

    kappa_values = np.linspace(
        kappa_min,
        kappa_max,
        n_kappa,
    )
    qr_values = np.linspace(
        qr_min,
        qr_max,
        n_qr,
    )

    print("\n=== Scan 2D MSTS : (kappa, Qr) ===")
    print(
        f"kappa : [{kappa_min}, {kappa_max}], "
        f"n={n_kappa}"
    )
    print(
        f"Qr    : [{qr_min}, {qr_max}], "
        f"n={n_qr}"
    )
    print(
        f"Grille : {n_kappa * n_qr} simulations "
        "par durée et par source"
    )
    print(
        f"DG : Nx={scan_nx}, "
        f"N_snapshot={scan_n_snapshot}"
    )
    print(f"Résolutions MSTS : {stft_resolutions}")
    print(
        "OpenWind l_ele : "
        f"{args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4}"
    )

    summary_rows = []

    for source in target_sources:
        for T_max in t_max_values:
            dt, nsteps, solve_kwargs = make_solver_data(
                T_max,
                train_params["cfl"],
                scan_nx,
                scan_n_snapshot,
                length,
                c,
                bc,
                phi0,
                y0,
                z0,
            )
            snapshot_times = (
                solve_kwargs["n_snaps"] + 1
            ) * solve_kwargs["dt"]

            print("\n" + "#" * 80)
            print(
                f"Source={source} | T_max={T_max:.4f} s | "
                f"dt={dt:.6e} | nsteps={nsteps}"
            )
            print("#" * 80)

            target_p, params_dg = make_target_for_source(
                source,
                params,
                geometry,
                c,
                solve_kwargs,
                {"T_max": T_max},
                snapshot_times,
                args,
            )

            true_kappa = value_from_params(
                params_dg,
                "kappa",
            )
            true_qr = value_from_params(
                params_dg,
                "Qr",
            )

            params_scan = copy.deepcopy(params_dg)
            for name in params_scan["trainable"]:
                params_scan["trainable"][name] = (
                    name in ("kappa", "Qr")
                )

            data_base = build_physical_data(
                params_scan,
                args.type_S,
            )

            scan_loss = make_scan_loss_2d(
                data_base,
                geometry,
                c,
                target_p,
                solve_kwargs,
                stft_resolutions,
                args.scan_stft_dynamic_db,
                stft_allow_padding,
            )

            stem = (
                f"scan_2D_kappa_Qr_MSTS_"
                f"{source}_T{T_max:.3f}"
            ).replace(".", "p")

            checkpoint_path = os.path.join(
                output_dir,
                stem + "_checkpoint.npz",
            )

            losses_msts = np.full(
                (n_qr, n_kappa),
                np.nan,
                dtype=float,
            )
            completed_rows = np.zeros(n_qr, dtype=bool)

            # Reprise automatique après une interruption ou un plantage CUDA.
            if os.path.exists(checkpoint_path):
                with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
                    saved_kappa = np.asarray(checkpoint["kappa_values"])
                    saved_qr = np.asarray(checkpoint["qr_values"])

                    compatible = (
                        saved_kappa.shape == kappa_values.shape
                        and saved_qr.shape == qr_values.shape
                        and np.allclose(saved_kappa, kappa_values)
                        and np.allclose(saved_qr, qr_values)
                    )
                    if compatible:
                        losses_msts[:] = np.asarray(
                            checkpoint["losses_msts"],
                            dtype=float,
                        )
                        completed_rows[:] = np.asarray(
                            checkpoint["completed_rows"],
                            dtype=bool,
                        )
                        print(
                            f"Reprise du checkpoint : "
                            f"{int(np.sum(completed_rows))}/{n_qr} lignes terminées."
                        )
                    else:
                        print(
                            "Checkpoint ignoré : la grille sauvegardée "
                            "est incompatible avec la grille demandée."
                        )

            kappa_values_device = jnp.asarray(
                kappa_values,
                dtype=jnp.float64,
            )

            for i_qr, qr_value in enumerate(qr_values):
                if completed_rows[i_qr]:
                    print(
                        f"Ligne {i_qr + 1:>3}/{n_qr} déjà calculée "
                        f"(Qr={qr_value:.4f})"
                    )
                    continue

                qr_device = jnp.asarray(
                    qr_value,
                    dtype=jnp.float64,
                )

                # Un seul appel compilé par ligne. block_until_ready force
                # la fin du calcul avant le transfert vers la mémoire hôte.
                row_device = scan_loss(
                    kappa_values_device,
                    qr_device,
                )
                row_device.block_until_ready()
                row_host = np.asarray(
                    jax.device_get(row_device),
                    dtype=float,
                )

                if row_host.shape != (n_kappa,):
                    raise RuntimeError(
                        f"Forme de ligne inattendue : {row_host.shape}, "
                        f"attendu={(n_kappa,)}."
                    )
                if not np.all(np.isfinite(row_host)):
                    bad = np.flatnonzero(~np.isfinite(row_host))
                    raise FloatingPointError(
                        f"Loss non finie pour Qr={qr_value:.6g}, "
                        f"indices kappa={bad.tolist()}."
                    )

                losses_msts[i_qr, :] = row_host
                completed_rows[i_qr] = True

                row_min_index = int(np.argmin(row_host))
                print(
                    f"Ligne {i_qr + 1:>3}/{n_qr} | "
                    f"Qr={qr_value:9.4f} | "
                    f"min L_MSTS={row_host[row_min_index]:.6e} "
                    f"à kappa={kappa_values[row_min_index]:.6f}"
                )

                # Sauvegarde après chaque ligne : en cas de nouveau plantage,
                # le script reprend à la première ligne incomplète.
                np.savez(
                    checkpoint_path,
                    kappa_values=kappa_values,
                    qr_values=qr_values,
                    losses_msts=losses_msts,
                    completed_rows=completed_rows,
                    true_kappa=true_kappa,
                    true_Qr=true_qr,
                    T_max=T_max,
                )

                del row_device, row_host
                gc.collect()

            if not np.all(completed_rows):
                raise RuntimeError(
                    "Le scan est incomplet malgré la fin de la boucle."
                )

            flat_index = int(
                np.nanargmin(losses_msts)
            )
            i_qr_min, i_kappa_min = np.unravel_index(
                flat_index,
                losses_msts.shape,
            )

            min_kappa = float(
                kappa_values[i_kappa_min]
            )
            min_qr = float(
                qr_values[i_qr_min]
            )
            min_loss = float(
                losses_msts[i_qr_min, i_kappa_min]
            )

            rel_error_kappa = abs(
                min_kappa - true_kappa
            ) / max(abs(true_kappa), 1e-12)
            rel_error_qr = abs(
                min_qr - true_qr
            ) / max(abs(true_qr), 1e-12)

            npz_path = os.path.join(
                output_dir,
                stem + ".npz",
            )
            np.savez(
                npz_path,
                kappa_values=kappa_values,
                qr_values=qr_values,
                losses_msts=losses_msts,
                completed_rows=completed_rows,
                true_kappa=true_kappa,
                true_Qr=true_qr,
                min_kappa=min_kappa,
                min_Qr=min_qr,
                min_loss=min_loss,
                T_max=T_max,
            )

            # Le fichier final est complet : le checkpoint temporaire
            # n'est plus nécessaire.
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)

            for use_log in (False, True):
                suffix = "_log10" if use_log else ""
                figure_path = os.path.join(
                    output_dir,
                    stem + suffix + ".png",
                )
                plot_loss_map(
                    kappa_values,
                    qr_values,
                    losses_msts,
                    true_kappa,
                    true_qr,
                    min_kappa,
                    min_qr,
                    (
                        f"Loss MSTS pression, "
                        f"{source}/DG, "
                        rf"$T={T_max:.2f}$ s"
                    ),
                    figure_path,
                    use_log=use_log,
                )

            # Figure complète : carte 2D + deux coupes 1D.
            diagnostics_path = os.path.join(
                output_dir,
                stem + "_diagnostics_2D.png",
            )
            plot_2d_diagnostics(
                kappa_values,
                qr_values,
                losses_msts,
                true_kappa,
                true_qr,
                min_kappa,
                min_qr,
                (
                    f"Scan 2D MSTS pression, "
                    f"{source}/DG, "
                    rf"$T={T_max:.2f}$ s"
                ),
                diagnostics_path,
            )
            print("  figure 2D :", diagnostics_path)

            summary_rows.append(
                {
                    "target_source": source,
                    "T_max": T_max,
                    "true_kappa": true_kappa,
                    "true_Qr": true_qr,
                    "min_kappa": min_kappa,
                    "min_Qr": min_qr,
                    "min_loss": min_loss,
                    "rel_error_kappa": rel_error_kappa,
                    "rel_error_Qr": rel_error_qr,
                    "n_kappa": n_kappa,
                    "n_Qr": n_qr,
                    "kappa_min": kappa_min,
                    "kappa_max": kappa_max,
                    "Qr_min": qr_min,
                    "Qr_max": qr_max,
                }
            )

            print("\nMinimum de la grille :")
            print(
                f"  vrai    : kappa={true_kappa:.6g}, "
                f"Qr={true_qr:.6g}"
            )
            print(
                f"  minimum : kappa={min_kappa:.6g}, "
                f"Qr={min_qr:.6g}, "
                f"loss={min_loss:.6e}"
            )
            print(
                f"  erreurs : "
                f"kappa={100.0 * rel_error_kappa:.3f}% | "
                f"Qr={100.0 * rel_error_qr:.3f}%"
            )
            print(f"  données : {npz_path}")

    summary_path = write_summary_csv(
        summary_rows,
        output_dir,
    )
    print("\nCSV résumé :", summary_path)




# =============================================================================
# VERSION MULTI-SCANS : (gamma_final, Qr), (zeta, Qr), (kappa, Qr)
# =============================================================================

SCAN_PAIRS = (
    ("gamma_final", "Qr"),
    ("zeta", "Qr"),
    ("kappa", "Qr"),
)

PARAM_LABELS = {
    "gamma_final": r"$\gamma$",
    "zeta": r"$\zeta$",
    "kappa": r"$\kappa$",
    "Qr": r"$Q_r$",
}


def make_scan_loss_2d_generic(
    param_x,
    param_y,
    data_base,
    geometry,
    c,
    target_p,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    """Loss MSTS 2D générique pour un couple de paramètres."""

    def loss_one(x_value, y_value):
        data = set_param(data_base, param_x, x_value, GEO_KEYS)
        data = set_param(data, param_y, y_value, GEO_KEYS)

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

    @jax.jit
    def loss_row(x_values, y_value):
        return jax.lax.map(
            lambda x_value: loss_one(x_value, y_value),
            x_values,
        )

    return loss_row


def plot_three_scans(pair_results, source, T_max, output_path):
    """Une image horizontale avec les trois cartes 2D en log10."""
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(19.5, 6.0),
        constrained_layout=True,
    )

    for ax, result in zip(axes, pair_results):
        displayed = safe_log10(result["losses"])

        contour = ax.contourf(
            result["x_values"],
            result["y_values"],
            displayed,
            levels=50,
        )
        lines = ax.contour(
            result["x_values"],
            result["y_values"],
            displayed,
            levels=12,
            linewidths=0.55,
        )
        ax.clabel(lines, inline=True, fontsize=6, fmt="%.2f")

        fig.colorbar(
            contour,
            ax=ax,
            label=r"$\log_{10}(\mathcal{L}_{\mathrm{MSTS}})$",
        )

        ax.axvline(result["true_x"], linestyle="--", linewidth=1.3)
        ax.axhline(result["true_y"], linestyle="--", linewidth=1.3)
        ax.scatter(
            [result["true_x"]],
            [result["true_y"]],
            marker="x",
            s=95,
            linewidths=2.3,
            label="Vrai couple",
            zorder=5,
        )
        ax.scatter(
            [result["min_x"]],
            [result["min_y"]],
            marker="o",
            s=60,
            label="Minimum",
            zorder=5,
        )

        px = result["param_x"]
        py = result["param_y"]
        ax.set_xlabel(PARAM_LABELS[px])
        ax.set_ylabel(PARAM_LABELS[py])
        ax.set_title(f"Scan {PARAM_LABELS[px]}–{PARAM_LABELS[py]}")
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=8)

    fig.suptitle(
        f"Scans 2D de la loss MSTS — cible {source.upper()}, "
        rf"$T={T_max:.3f}$ s",
        fontsize=15,
    )
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_three_scan_summary(rows, output_dir):
    path = os.path.join(
        output_dir,
        "scan_2D_three_pairs_MSTS_summary.csv",
    )
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_one_pair_scan(
    *,
    param_x,
    param_y,
    source,
    T_max,
    params_dg,
    geometry,
    c,
    target_p,
    solve_kwargs,
    args,
    output_dir,
    n_x,
    n_y,
    stft_resolutions,
    stft_allow_padding,
):
    x_min, x_max = SCAN_RANGES[param_x]
    y_min, y_max = SCAN_RANGES[param_y]
    x_values = np.linspace(x_min, x_max, n_x)
    y_values = np.linspace(y_min, y_max, n_y)

    true_x = value_from_params(params_dg, param_x)
    true_y = value_from_params(params_dg, param_y)

    params_scan = copy.deepcopy(params_dg)
    for name in params_scan["trainable"]:
        params_scan["trainable"][name] = name in (param_x, param_y)

    data_base = build_physical_data(params_scan, args.type_S)
    scan_loss = make_scan_loss_2d_generic(
        param_x,
        param_y,
        data_base,
        geometry,
        c,
        target_p,
        solve_kwargs,
        stft_resolutions,
        args.scan_stft_dynamic_db,
        stft_allow_padding,
    )

    stem = (
        f"scan_2D_{param_x}_{param_y}_MSTS_{source}_T{T_max:.3f}"
    ).replace(".", "p")
    checkpoint_path = os.path.join(output_dir, stem + "_checkpoint.npz")

    losses = np.full((n_y, n_x), np.nan, dtype=float)
    completed_rows = np.zeros(n_y, dtype=bool)

    if os.path.exists(checkpoint_path):
        with np.load(checkpoint_path, allow_pickle=False) as checkpoint:
            saved_x = np.asarray(checkpoint["x_values"])
            saved_y = np.asarray(checkpoint["y_values"])
            compatible = (
                saved_x.shape == x_values.shape
                and saved_y.shape == y_values.shape
                and np.allclose(saved_x, x_values)
                and np.allclose(saved_y, y_values)
            )
            if compatible:
                losses[:] = np.asarray(checkpoint["losses"], dtype=float)
                completed_rows[:] = np.asarray(
                    checkpoint["completed_rows"], dtype=bool
                )
                print(
                    f"Reprise ({param_x},{param_y}) : "
                    f"{int(np.sum(completed_rows))}/{n_y} lignes."
                )

    x_device = jnp.asarray(x_values, dtype=jnp.float64)

    print("\n" + "=" * 80)
    print(f"SCAN ({param_x}, {param_y})")
    print(f"x=[{x_min}, {x_max}], y=[{y_min}, {y_max}], grille={n_x}x{n_y}")
    print("=" * 80)

    for i_y, y_value in enumerate(y_values):
        if completed_rows[i_y]:
            continue

        y_device = jnp.asarray(y_value, dtype=jnp.float64)
        row_device = scan_loss(x_device, y_device)
        row_device.block_until_ready()
        row_host = np.asarray(jax.device_get(row_device), dtype=float)

        if row_host.shape != (n_x,):
            raise RuntimeError(
                f"Forme inattendue : {row_host.shape}, attendu={(n_x,)}"
            )
        if not np.all(np.isfinite(row_host)):
            bad = np.flatnonzero(~np.isfinite(row_host))
            raise FloatingPointError(
                f"Loss non finie pour {param_y}={y_value:.6g}, "
                f"indices={bad.tolist()}."
            )

        losses[i_y, :] = row_host
        completed_rows[i_y] = True

        i_row_min = int(np.argmin(row_host))
        print(
            f"Ligne {i_y + 1:>3}/{n_y} | "
            f"{param_y}={y_value:10.5f} | "
            f"min={row_host[i_row_min]:.6e} "
            f"à {param_x}={x_values[i_row_min]:.6f}"
        )

        np.savez(
            checkpoint_path,
            x_values=x_values,
            y_values=y_values,
            losses=losses,
            completed_rows=completed_rows,
            param_x=param_x,
            param_y=param_y,
            true_x=true_x,
            true_y=true_y,
            T_max=T_max,
        )

        del row_device, row_host
        gc.collect()

    if not np.all(completed_rows):
        raise RuntimeError(f"Le scan ({param_x},{param_y}) est incomplet.")

    flat_index = int(np.nanargmin(losses))
    i_y_min, i_x_min = np.unravel_index(flat_index, losses.shape)
    min_x = float(x_values[i_x_min])
    min_y = float(y_values[i_y_min])
    min_loss = float(losses[i_y_min, i_x_min])

    rel_error_x = abs(min_x - true_x) / max(abs(true_x), 1e-12)
    rel_error_y = abs(min_y - true_y) / max(abs(true_y), 1e-12)

    npz_path = os.path.join(output_dir, stem + ".npz")
    np.savez(
        npz_path,
        x_values=x_values,
        y_values=y_values,
        losses_msts=losses,
        completed_rows=completed_rows,
        param_x=param_x,
        param_y=param_y,
        true_x=true_x,
        true_y=true_y,
        min_x=min_x,
        min_y=min_y,
        min_loss=min_loss,
        T_max=T_max,
    )

    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

    print(
        f"Minimum ({param_x},{param_y}) = "
        f"({min_x:.6g}, {min_y:.6g}), loss={min_loss:.6e}"
    )

    result = {
        "param_x": param_x,
        "param_y": param_y,
        "x_values": x_values,
        "y_values": y_values,
        "losses": losses,
        "true_x": true_x,
        "true_y": true_y,
        "min_x": min_x,
        "min_y": min_y,
        "min_loss": min_loss,
    }
    summary = {
        "target_source": source,
        "T_max": T_max,
        "param_x": param_x,
        "param_y": param_y,
        "true_x": true_x,
        "true_y": true_y,
        "min_x": min_x,
        "min_y": min_y,
        "min_loss": min_loss,
        "rel_error_x": rel_error_x,
        "rel_error_y": rel_error_y,
        "n_x": n_x,
        "n_y": n_y,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
    }
    return result, summary


def main_three_scans():
    args = parse_args()
    root = repo_root()
    output_dir = resolve_project_path(root, args.scan_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    stft_resolutions = parse_stft_resolutions(args.scan_stft_resolutions)
    stft_allow_padding = not args.scan_no_stft_padding
    t_max_values = parse_float_list(
        args.scan_t_max_values,
        "scan_t_max_values",
    )

    n_x = int(args.scan_n)
    n_y = int(args.scan_n)
    if n_x <= 1 or n_y <= 1:
        raise ValueError("--scan_n doit être supérieur à 1.")

    with open(root / "experiments/gradient/config/simu.json", "r") as file:
        solver_params = json.load(file)["solver_params"]
    with open(root / "experiments/gradient/config/param.json", "r") as file:
        params = json.load(file)

    train_params = solver_params["train"]
    scan_nx = args.scan_nx if args.scan_nx is not None else train_params["Nx"]
    scan_n_snapshot = (
        args.scan_n_snapshot
        if args.scan_n_snapshot is not None
        else train_params["N_snapshot"]
    )

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    data_ref = build_physical_data(params, args.type_S)
    length = data_ref.section.L_tube + data_ref.section.L_bell
    geometry = build_solver_geometry(data_ref, scan_nx, c)
    bc = BC(type="full")

    target_sources = (
        ["dg", "openwind"]
        if args.scan_target_source == "both"
        else [args.scan_target_source]
    )

    print("\n=== Trois scans 2D MSTS successifs ===")
    print("Couples : (gamma_final,Qr), (zeta,Qr), (kappa,Qr)")
    print(
        f"Total par durée/source : "
        f"{len(SCAN_PAIRS) * n_x * n_y} simulations"
    )
    print(f"DG : Nx={scan_nx}, N_snapshot={scan_n_snapshot}")

    summary_rows = []

    for source in target_sources:
        for T_max in t_max_values:
            dt, nsteps, solve_kwargs = make_solver_data(
                T_max,
                train_params["cfl"],
                scan_nx,
                scan_n_snapshot,
                length,
                c,
                bc,
                phi0,
                y0,
                z0,
            )
            snapshot_times = (
                solve_kwargs["n_snaps"] + 1
            ) * solve_kwargs["dt"]

            print("\n" + "#" * 80)
            print(
                f"Source={source} | T_max={T_max:.4f} s | "
                f"dt={dt:.6e} | nsteps={nsteps}"
            )
            print("#" * 80)

            target_p, params_dg = make_target_for_source(
                source,
                params,
                geometry,
                c,
                solve_kwargs,
                {"T_max": T_max},
                snapshot_times,
                args,
            )

            pair_results = []
            for param_x, param_y in SCAN_PAIRS:
                result, summary = run_one_pair_scan(
                    param_x=param_x,
                    param_y=param_y,
                    source=source,
                    T_max=T_max,
                    params_dg=params_dg,
                    geometry=geometry,
                    c=c,
                    target_p=target_p,
                    solve_kwargs=solve_kwargs,
                    args=args,
                    output_dir=output_dir,
                    n_x=n_x,
                    n_y=n_y,
                    stft_resolutions=stft_resolutions,
                    stft_allow_padding=stft_allow_padding,
                )
                pair_results.append(result)
                summary_rows.append(summary)

            combined_stem = (
                f"scan_2D_gamma_zeta_kappa_Qr_MSTS_{source}_T{T_max:.3f}"
            ).replace(".", "p")
            combined_png = os.path.join(output_dir, combined_stem + ".png")
            plot_three_scans(pair_results, source, T_max, combined_png)

            combined_npz = os.path.join(output_dir, combined_stem + ".npz")
            np.savez(
                combined_npz,
                gamma_values=pair_results[0]["x_values"],
                gamma_qr_values=pair_results[0]["y_values"],
                gamma_losses=pair_results[0]["losses"],
                zeta_values=pair_results[1]["x_values"],
                zeta_qr_values=pair_results[1]["y_values"],
                zeta_losses=pair_results[1]["losses"],
                kappa_values=pair_results[2]["x_values"],
                kappa_qr_values=pair_results[2]["y_values"],
                kappa_losses=pair_results[2]["losses"],
                T_max=T_max,
            )

            print("\nImage combinée :", combined_png)
            print("NPZ combiné     :", combined_npz)

    summary_path = write_three_scan_summary(summary_rows, output_dir)
    print("\nCSV résumé :", summary_path)


if __name__ == "__main__":
    main_three_scans()