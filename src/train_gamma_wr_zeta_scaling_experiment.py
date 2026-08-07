import argparse
import copy
import csv
import json
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.res_openwind import run_openwind_reference
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)

P_CLOSED = 5e3
MIN_SCALE = 1e-8
DEFAULT_INIT_FACTOR = 0.8
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

PARAM_JSON_PATHS = {
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "Qr": ("left_bc_params", "Qr"),
    "fr": ("left_bc_params", "fr"),
    "zeta": ("left_bc_params", "zeta"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Optimisation simultanee de gamma, wr et zeta sur plusieurs signaux. "
            "Qr est fixe a la valeur du param.json. Les cibles OpenWind sont "
            "precalculees une seule fois par duree."
        )
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=5, help="Conserve pour compatibilite.")
    parser.add_argument(
        "--experiment_sizes",
        type=str,
        default="1,5,10",
        help="Nombres de signaux a tester, par exemple 1,5,10,20.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Nombre de tirages aleatoires independants pour chaque taille N.",
    )
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--stage1_iter", type=int, default=200)
    parser.add_argument("--stage2_iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-3, help="Conserve pour compatibilite.")
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage2_lr", type=float, default=5e-3)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)
    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="32:8,64:16,128:32",
        help=(
            "Resolutions STFT n_fft:hop separees par des virgules. "
            "Exemple: 32:8,64:16,128:32"
        ),
    )
    parser.add_argument(
        "--no_stft_padding",
        action="store_true",
        help="Ignore les resolutions STFT plus grandes que le signal au lieu de zero-padder.",
    )
    parser.add_argument(
        "--stft_dynamic_db",
        type=float,
        default=60.0,
        help="Dynamique en dB utilisee dans la partie log de la loss spectrale.",
    )
    parser.add_argument(
        "--random_factor_min",
        type=float,
        default=0.75,
        help="Borne basse du facteur aleatoire applique aux valeurs de reference.",
    )
    parser.add_argument(
        "--random_factor_max",
        type=float,
        default=1.25,
        help="Borne haute du facteur aleatoire applique aux valeurs de reference.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/gradient/results/gamma_wr_zeta_Qr_fixed",
    )
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


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
        raise ValueError("La liste --stft_resolutions est vide.")

    return tuple(resolutions)


def base_triplet_from_params(params):
    """Renvoie [gamma, wr, zeta]. Qr n'est pas optimise."""
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    fr = float(get_nested(params, PARAM_JSON_PATHS["fr"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    wr = 2.0 * np.pi * fr
    return np.asarray([gamma, wr, zeta], dtype=float)


def set_triplet_json(params, triplet):
    """Modifie seulement gamma, fr et zeta dans le json OpenWind."""
    gamma, wr, zeta = [float(v) for v in triplet]
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["fr"], wr / (2.0 * np.pi))
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    return params


def set_triplet_data(data, triplet):
    """Modifie seulement gamma, fr et zeta dans les donnees DG."""
    gamma, wr, zeta = triplet
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "fr", wr / (2.0 * jnp.pi), GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    return data


def set_trainable_triplet(params):
    """Qr reste fixe a la valeur presente dans param.json."""
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name in ("gamma_final", "fr", "zeta")
    return params


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


def make_optimizer(lr, n_iter):
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_iter,
        alpha=1e-2,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler, weight_decay=1e-5),
    )
    return optimizer, scheduler


def make_openwind_target(params_true, type_S, T_max, snapshot_times, args):
    t_ow, _, p_right, _, _, ow_params = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=args.ow_l_ele,
        order=args.ow_order,
    )

    target = np.interp(
        np.asarray(snapshot_times, dtype=float),
        np.asarray(t_ow, dtype=float),
        np.asarray(p_right, dtype=float),
    )

    print(
        f"  OpenWind: dt={ow_params['dt']:.6e}, "
        f"n={len(t_ow)}, h_eff={ow_params['h_eff']:.6e}"
    )
    return jnp.asarray(target / P_CLOSED, dtype=jnp.float64), ow_params


def sample_true_triplets(base_triplet, n_signals, factor_min, factor_max, seed):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(factor_min, factor_max, size=(n_signals, 3))
    return np.asarray(base_triplet[None, :] * factors, dtype=float)


def build_targets_for_T(
    T_max,
    params,
    true_values_np,
    data_ref,
    c,
    bc,
    phi0,
    y0,
    z0,
    train_params,
    args,
):
    """
    Construit les cibles OpenWind une seule fois pour une duree T_max.
    Ensuite les stages utilisent directement target_signals sans relancer OpenWind.
    """
    dt, nsteps, solve_kwargs = make_solver_data(
        T_max,
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        data_ref.section.L_tube + data_ref.section.L_bell,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    snapshot_times = (solve_kwargs["n_snaps"] + 1) * solve_kwargs["dt"]
    target_signals = []

    print(f"\n=== Pre-calcul des cibles OpenWind pour T = {T_max:.4f} s ===")
    for i, true_triplet in enumerate(true_values_np):
        print(f"\nSignal {i + 1}/{len(true_values_np)}")
        params_true = set_triplet_json(copy.deepcopy(params), true_triplet)
        target, _ = make_openwind_target(
            params_true,
            args.type_S,
            T_max,
            snapshot_times,
            args,
        )
        target_signals.append(target)
        print(
            f"  cible: max={float(jnp.max(jnp.abs(target))):.4e}, "
            f"std={float(jnp.std(target)):.4e}"
        )

    return jnp.stack(target_signals, axis=0), solve_kwargs, snapshot_times


def make_batched_loss(
    data_init,
    geometry,
    c,
    target_signals,
    scales,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    """Loss moyenne sur les signaux, vectorisee avec jax.vmap."""

    def loss_one_signal(theta_one, scale_one, target_one):
        params_phys = theta_one * scale_one
        data = set_triplet_data(data_init, params_phys)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(loss_one_signal, in_axes=(0, 0, 0))

    def loss(theta):
        losses = vmapped_loss(theta, scales, target_signals)
        return jnp.mean(losses)

    return loss


def make_losses_per_signal(
    data_init,
    geometry,
    c,
    target_signals,
    scales,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    """Renvoie une fonction theta -> vecteur des losses individuelles."""

    def loss_one_signal(theta_one, scale_one, target_one):
        params_phys = theta_one * scale_one
        data = set_triplet_data(data_init, params_phys)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(loss_one_signal, in_axes=(0, 0, 0))

    @jax.jit
    def losses_per_signal(theta):
        return vmapped_loss(theta, scales, target_signals)

    return losses_per_signal

def optimize_stage(theta, loss_fn, train_mask, lr, n_iter, print_every=10):
    optimizer, scheduler = make_optimizer(lr, n_iter)
    opt_state = optimizer.init(theta)
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    final_loss = None
    print_every = max(int(print_every), 1)

    for iteration in range(n_iter):
        t_it = time.time()
        loss_val, grad_val = loss_and_grad(theta)
        final_loss = loss_val

        grad_val = grad_val * train_mask
        updates, opt_state = optimizer.update(grad_val, opt_state, theta)
        updates = jax.tree_util.tree_map(lambda u: u * train_mask, updates)
        theta = optax.apply_updates(theta, updates)

        if iteration % print_every == 0 or iteration == n_iter - 1:
            current_lr = float(scheduler(iteration))
            print(
                f"    iter {iteration:4d} | loss = {float(loss_val):.4e} | "
                f"lr = {current_lr:.3e} | t/iter = {time.time() - t_it:.2f}s"
            )

    if final_loss is None:
        final_loss = loss_fn(theta)

    return theta, float(final_loss)


def relative_errors(params_current, true_values):
    denom = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    err = jnp.abs((params_current - true_values) / denom)
    return err, float(jnp.linalg.norm(err))


def write_results_csv(path, true_values, estimated_values, err_values, final_loss):
    fieldnames = [
        "signal_idx",
        "true_gamma",
        "true_wr",
        "true_fr",
        "true_zeta",
        "estimated_gamma",
        "estimated_wr",
        "estimated_fr",
        "estimated_zeta",
        "relerr_gamma",
        "relerr_wr",
        "relerr_zeta",
        "final_loss",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(true_values.shape[0]):
            writer.writerow(
                {
                    "signal_idx": i,
                    "true_gamma": float(true_values[i, 0]),
                    "true_wr": float(true_values[i, 1]),
                    "true_fr": float(true_values[i, 1] / (2.0 * np.pi)),
                    "true_zeta": float(true_values[i, 2]),
                    "estimated_gamma": float(estimated_values[i, 0]),
                    "estimated_wr": float(estimated_values[i, 1]),
                    "estimated_fr": float(estimated_values[i, 1] / (2.0 * np.pi)),
                    "estimated_zeta": float(estimated_values[i, 2]),
                    "relerr_gamma": float(err_values[i, 0]),
                    "relerr_wr": float(err_values[i, 1]),
                    "relerr_zeta": float(err_values[i, 2]),
                    "final_loss": float(final_loss),
                }
            )


def write_summary_csv(path, err_values):
    names = ["gamma", "wr", "zeta"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["parameter", "mean_relerr", "std_relerr", "max_relerr"],
        )
        writer.writeheader()
        err_np = np.asarray(err_values)
        for j, name in enumerate(names):
            writer.writerow(
                {
                    "parameter": name,
                    "mean_relerr": float(np.mean(err_np[:, j])),
                    "std_relerr": float(np.std(err_np[:, j])),
                    "max_relerr": float(np.max(err_np[:, j])),
                }
            )


def make_trained_predictions(data_init, geometry, c, estimated_values, solve_kwargs):
    preds = []
    for signal_idx in range(int(estimated_values.shape[0])):
        data = set_triplet_data(data_init, estimated_values[signal_idx])
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        preds.append(pred)
    return jnp.stack(preds, axis=0)


def plot_parameter_comparison(path, true_values, estimated_values, err_values):
    labels = [r"$\gamma$", r"$\omega_r$", r"$\zeta$"]
    n_signals = true_values.shape[0]
    x = np.arange(1, n_signals + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.2), squeeze=False)
    axes = axes.ravel()

    for param_idx, ax in enumerate(axes):
        y_true = true_values[:, param_idx]
        y_est = estimated_values[:, param_idx]
        x_true = x - 0.08
        x_est = x + 0.08

        ax.scatter(x_true, y_true, marker="o", s=48, label="Reel")
        ax.scatter(x_est, y_est, marker="s", s=48, label="Entraine")

        for x_i, y_i, err_i in zip(x_est, y_est, err_values[:, param_idx]):
            ax.annotate(
                f"{err_i:.2e}",
                (x_i, y_i),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(labels[param_idx])
        ax.set_xlabel("Signal")
        ax.set_ylabel("Valeur")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend()

        all_values = np.concatenate([y_true, y_est])
        ymin = float(np.min(all_values))
        ymax = float(np.max(all_values))
        margin = max(0.08 * (ymax - ymin), 1e-8)
        ax.set_ylim(ymin - margin, ymax + margin)

    fig.suptitle("Parametres reels vs parametres entraines", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_signal_comparison(path, times, target_signals, trained_signals):
    n_signals = target_signals.shape[0]
    n_cols = 1 if n_signals == 1 else 2
    n_rows = int(np.ceil(n_signals / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
    )
    axes_flat = axes.ravel()

    for signal_idx in range(n_signals):
        ax = axes_flat[signal_idx]
        ax.plot(times, target_signals[signal_idx], label="OpenWind", linewidth=1.5)
        ax.plot(
            times,
            trained_signals[signal_idx],
            "--",
            label="DG entraine",
            linewidth=1.3,
        )
        ax.set_title(f"Signal {signal_idx + 1}")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Pression / P_closed")
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[n_signals:]:
        ax.axis("off")

    fig.suptitle("Signaux OpenWind vs DG entraine", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)



def parse_experiment_sizes(value):
    sizes = sorted({int(v.strip()) for v in value.split(",") if v.strip()})
    if not sizes or any(v <= 0 for v in sizes):
        raise ValueError("--experiment_sizes doit contenir des entiers strictement positifs.")
    return sizes


def write_scaling_experiment_csv(path, rows):
    """Ecrit une ligne par couple (repetition, N)."""
    fieldnames = [
        "repeat_idx", "seed", "n_signals",
        "mean_relerr_gamma", "std_relerr_gamma",
        "mean_relerr_wr", "std_relerr_wr",
        "mean_relerr_zeta", "std_relerr_zeta",
        "global_relerr", "final_loss", "elapsed_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows):
    """
    Agrege les repetitions pour chaque valeur de N.

    Les moyennes/std ci-dessous sont calculees entre repetitions sur la
    moyenne d'erreur obtenue dans chaque repetition. Cela mesure la
    variabilite due au tirage aleatoire, ce qui est le but de l'experience.
    """
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["n_signals"]), []).append(row)

    aggregated = []
    for n_signals in sorted(grouped):
        group = grouped[n_signals]
        out = {
            "n_signals": n_signals,
            "n_repeats": len(group),
        }
        for name in ("gamma", "wr", "zeta"):
            values = np.asarray(
                [r[f"mean_relerr_{name}"] for r in group], dtype=float
            )
            out[f"mean_relerr_{name}"] = float(np.mean(values))
            out[f"std_relerr_{name}"] = float(np.std(values))
            out[f"min_relerr_{name}"] = float(np.min(values))
            out[f"max_relerr_{name}"] = float(np.max(values))

        for key in ("global_relerr", "final_loss", "elapsed_seconds"):
            values = np.asarray([r[key] for r in group], dtype=float)
            out[f"mean_{key}"] = float(np.mean(values))
            out[f"std_{key}"] = float(np.std(values))

        aggregated.append(out)

    return aggregated


def write_aggregated_csv(path, rows):
    fieldnames = [
        "n_signals", "n_repeats",
        "mean_relerr_gamma", "std_relerr_gamma",
        "min_relerr_gamma", "max_relerr_gamma",
        "mean_relerr_wr", "std_relerr_wr",
        "min_relerr_wr", "max_relerr_wr",
        "mean_relerr_zeta", "std_relerr_zeta",
        "min_relerr_zeta", "max_relerr_zeta",
        "mean_global_relerr", "std_global_relerr",
        "mean_final_loss", "std_final_loss",
        "mean_elapsed_seconds", "std_elapsed_seconds",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_scaling_summary(path, rows):
    n = np.asarray([r["n_signals"] for r in rows], dtype=int)
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))

    for name, label in [
        ("gamma", r"$\gamma$"),
        ("wr", r"$\omega_r$"),
        ("zeta", r"$\zeta$"),
    ]:
        mean = 100.0 * np.asarray([r[f"mean_relerr_{name}"] for r in rows])
        std = 100.0 * np.asarray([r[f"std_relerr_{name}"] for r in rows])
        axes[0].errorbar(n, mean, yerr=std, marker="o", capsize=4, label=label)

    axes[0].set_xlabel("Nombre de signaux")
    axes[0].set_ylabel("Erreur relative moyenne entre repetitions (%)")
    axes[0].set_xticks(n)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    elapsed_minutes = np.asarray([r["mean_elapsed_seconds"] for r in rows]) / 60.0
    elapsed_std = np.asarray([r["std_elapsed_seconds"] for r in rows]) / 60.0
    axes[1].errorbar(n, elapsed_minutes, yerr=elapsed_std, marker="o", capsize=4)
    axes[1].set_xlabel("Nombre de signaux")
    axes[1].set_ylabel("Temps moyen d'optimisation (min)")
    axes[1].set_xticks(n)
    axes[1].grid(True, alpha=0.3)

    n_repeats = rows[0]["n_repeats"] if rows else 0
    fig.suptitle(
        f"Robustesse et cout en fonction du nombre de signaux "
        f"({n_repeats} tirages)",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_error_boxplots(path, raw_rows):
    """Boxplots de la moyenne d'erreur obtenue a chaque repetition."""
    sizes = sorted({int(r["n_signals"]) for r in raw_rows})
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), squeeze=False)
    axes = axes.ravel()

    for ax, name, label in zip(
        axes,
        ("gamma", "wr", "zeta"),
        (r"$\gamma$", r"$\omega_r$", r"$\zeta$"),
    ):
        data = [
            100.0 * np.asarray([
                r[f"mean_relerr_{name}"]
                for r in raw_rows
                if int(r["n_signals"]) == n
            ])
            for n in sizes
        ]
        ax.boxplot(data, tick_labels=[str(n) for n in sizes], showmeans=True)
        ax.set_title(label)
        ax.set_xlabel("Nombre de signaux")
        ax.set_ylabel("Erreur moyenne par repetition (%)")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Distribution des erreurs sur les tirages aleatoires", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def print_experiment_table(rows):
    print("\n=== Synthese statistique de l'experience ===")
    print(
        f"{'N':>4} | {'R':>3} | {'gamma mean±std (%)':>22} | "
        f"{'wr mean±std (%)':>22} | {'zeta mean±std (%)':>22} | "
        f"{'loss mean±std':>21} | {'temps mean±std (min)':>24}"
    )
    print("-" * 132)
    for r in rows:
        print(
            f"{r['n_signals']:4d} | {r['n_repeats']:3d} | "
            f"{100*r['mean_relerr_gamma']:8.3f} ± {100*r['std_relerr_gamma']:7.3f} | "
            f"{100*r['mean_relerr_wr']:8.3f} ± {100*r['std_relerr_wr']:7.3f} | "
            f"{100*r['mean_relerr_zeta']:8.3f} ± {100*r['std_relerr_zeta']:7.3f} | "
            f"{r['mean_final_loss']:9.3e} ± {r['std_final_loss']:8.2e} | "
            f"{r['mean_elapsed_seconds']/60:8.2f} ± {r['std_elapsed_seconds']/60:7.2f}"
        )


def run_experiment(
    n_signals,
    repeat_idx,
    seed,
    true_values_all,
    targets_short_all,
    targets_long_all,
    data_ref,
    geometry,
    c,
    solve_kwargs_short,
    solve_kwargs_long,
    snapshot_times_long,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    stage1_lr,
    stage2_lr,
    stage1_iter,
    stage2_iter,
    print_every,
    output_dir,
):
    """Execute une optimisation pour une repetition et une taille N."""
    t_start = time.time()

    true_values = jnp.asarray(true_values_all[:n_signals], dtype=jnp.float64)
    targets_short = jnp.asarray(targets_short_all[:n_signals])
    targets_long = jnp.asarray(targets_long_all[:n_signals])

    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    theta = (true_values * DEFAULT_INIT_FACTOR) / scales
    train_mask = jnp.ones((1, 3), dtype=jnp.float64)

    loss_short = make_batched_loss(
        data_ref, geometry, c, targets_short, scales, solve_kwargs_short,
        stft_resolutions, stft_dynamic_db, stft_allow_padding,
    )
    loss_long = make_batched_loss(
        data_ref, geometry, c, targets_long, scales, solve_kwargs_long,
        stft_resolutions, stft_dynamic_db, stft_allow_padding,
    )

    print(f"\n{'=' * 76}")
    print(
        f"REPETITION {repeat_idx + 1} | SEED = {seed} | "
        f"N = {n_signals} SIGNAUX"
    )
    print(f"{'=' * 76}")
    print("\n--- Stage 1 : T = T_max ---")
    theta, loss_stage1 = optimize_stage(
        theta, loss_long, train_mask, stage1_lr, stage1_iter, print_every,
    )

    print("\n--- Stage 2 : T = 0.01 s ---")
    theta, loss_stage2 = optimize_stage(
        theta, loss_short, train_mask, stage2_lr, stage2_iter, print_every,
    )

    estimated_values = theta * scales
    err_values, err_global = relative_errors(estimated_values, true_values)
    losses_fn = make_losses_per_signal(
        data_ref, geometry, c, targets_long, scales, solve_kwargs_long,
        stft_resolutions, stft_dynamic_db, stft_allow_padding,
    )
    losses_per_signal = np.asarray(losses_fn(theta))
    final_loss = float(np.mean(losses_per_signal))

    print("\n=== Loss par signal ===")
    for i, value in enumerate(losses_per_signal):
        print(f"signal {i}: loss = {value:.6e}")

    # On ne produit les figures detaillees que pour chaque repetition dans
    # son propre repertoire afin d'eviter tout ecrasement de fichier.
    trained_signals = make_trained_predictions(
        data_ref, geometry, c, estimated_values, solve_kwargs_long,
    )

    run_dir = output_dir / f"repeat_{repeat_idx:03d}_seed_{seed}" / f"N_{n_signals:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_results_csv(
        run_dir / "gamma_wr_zeta_results.csv",
        true_values, estimated_values, err_values, final_loss,
    )
    write_summary_csv(run_dir / "gamma_wr_zeta_summary.csv", err_values)
    plot_parameter_comparison(
        run_dir / "params_true_vs_trained.png",
        np.asarray(true_values), np.asarray(estimated_values), np.asarray(err_values),
    )
    plot_signal_comparison(
        run_dir / "signals_openwind_vs_dg.png",
        np.asarray(snapshot_times_long), np.asarray(targets_long), np.asarray(trained_signals),
    )

    err_np = np.asarray(err_values)
    elapsed = time.time() - t_start
    row = {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "n_signals": n_signals,
        "mean_relerr_gamma": float(np.mean(err_np[:, 0])),
        "std_relerr_gamma": float(np.std(err_np[:, 0])),
        "mean_relerr_wr": float(np.mean(err_np[:, 1])),
        "std_relerr_wr": float(np.std(err_np[:, 1])),
        "mean_relerr_zeta": float(np.mean(err_np[:, 2])),
        "std_relerr_zeta": float(np.std(err_np[:, 2])),
        "global_relerr": float(err_global),
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
    }

    print(f"\n=== Resume repetition {repeat_idx + 1}, N = {n_signals} ===")
    print(f"gamma : {100*row['mean_relerr_gamma']:.3f} ± {100*row['std_relerr_gamma']:.3f} %")
    print(f"wr    : {100*row['mean_relerr_wr']:.3f} ± {100*row['std_relerr_wr']:.3f} %")
    print(f"zeta  : {100*row['mean_relerr_zeta']:.3f} ± {100*row['std_relerr_zeta']:.3f} %")
    print(f"loss finale : {final_loss:.4e}")
    print(f"temps       : {elapsed/60:.2f} min")
    print(f"loss stage 1: {loss_stage1:.4e}")
    print(f"loss stage 2: {loss_stage2:.4e}")

    return row


def main():
    args = parse_args()
    total_start = time.time()

    if args.n_repeats <= 0:
        raise ValueError("--n_repeats doit etre strictement positif.")

    experiment_sizes = parse_experiment_sizes(args.experiment_sizes)
    max_signals = max(experiment_sizes)
    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding
    stage2_iter = args.n_iter if args.stage2_iter is None else args.stage2_iter

    root = repo_root()
    with open(root / "experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]
    with open(root / "experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    train_params = solver_params["train"]
    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    params_trainable = set_trainable_triplet(params)
    data_ref = build_physical_data(params_trainable, args.type_S)
    geometry = build_solver_geometry(data_ref, train_params["Nx"], c)
    bc = BC(type="full")
    base_triplet = base_triplet_from_params(params)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Etude statistique multi-signaux avec repetitions ===")
    print("Tailles testees       :", experiment_sizes)
    print("Nombre de repetitions :", args.n_repeats)
    print("Seed de base          :", args.seed)
    print("Organisation          : pour chaque tirage, les tailles N sont imbriquees")
    print("Stage 1               : T=T_max, lr=", args.stage1_lr)
    print("Stage 2               : T=0.01 s, lr=", args.stage2_lr)
    print("Qr fixe depuis JSON   :", float(get_nested(params, PARAM_JSON_PATHS["Qr"])))
    print("STFT                  :", stft_resolutions)

    raw_rows = []

    for repeat_idx in range(args.n_repeats):
        seed = args.seed + repeat_idx
        repeat_start = time.time()
        print(f"\n{'#' * 80}")
        print(
            f"TIRAGE {repeat_idx + 1}/{args.n_repeats} "
            f"(seed={seed}, max_signals={max_signals})"
        )
        print(f"{'#' * 80}")

        # Un tirage independant par repetition. Les tailles N restent imbriquees
        # a l'interieur du tirage pour comparer N sur exactement les memes cas.
        true_values_all = sample_true_triplets(
            base_triplet,
            max_signals,
            args.random_factor_min,
            args.random_factor_max,
            seed,
        )

        # Les cibles OpenWind ne sont generees qu'une fois par repetition et
        # par duree, puis reutilisees pour toutes les valeurs de N.
        targets_short_all, solve_kwargs_short, _ = build_targets_for_T(
            0.01, params, true_values_all, data_ref, c, bc, phi0, y0, z0,
            train_params, args,
        )
        targets_long_all, solve_kwargs_long, snapshot_times_long = build_targets_for_T(
            train_params["T_max"], params, true_values_all, data_ref, c, bc,
            phi0, y0, z0, train_params, args,
        )

        for n_signals in experiment_sizes:
            row = run_experiment(
                n_signals=n_signals,
                repeat_idx=repeat_idx,
                seed=seed,
                true_values_all=true_values_all,
                targets_short_all=targets_short_all,
                targets_long_all=targets_long_all,
                data_ref=data_ref,
                geometry=geometry,
                c=c,
                solve_kwargs_short=solve_kwargs_short,
                solve_kwargs_long=solve_kwargs_long,
                snapshot_times_long=snapshot_times_long,
                stft_resolutions=stft_resolutions,
                stft_dynamic_db=args.stft_dynamic_db,
                stft_allow_padding=stft_allow_padding,
                stage1_lr=args.stage1_lr,
                stage2_lr=args.stage2_lr,
                stage1_iter=args.stage1_iter,
                stage2_iter=stage2_iter,
                print_every=args.print_every,
                output_dir=output_dir,
            )
            raw_rows.append(row)

        # Sauvegarde incrementale: on ne perd pas les repetitions terminees si
        # une longue experience est interrompue plus tard.
        write_scaling_experiment_csv(
            output_dir / "scaling_experiment_all_runs.csv",
            raw_rows,
        )
        current_aggregated = aggregate_rows(raw_rows)
        write_aggregated_csv(
            output_dir / "scaling_experiment_aggregated.csv",
            current_aggregated,
        )
        print(
            f"\nTemps du tirage {repeat_idx + 1}: "
            f"{(time.time() - repeat_start)/60:.2f} min"
        )

    aggregated_rows = aggregate_rows(raw_rows)

    raw_csv_path = output_dir / "scaling_experiment_all_runs.csv"
    agg_csv_path = output_dir / "scaling_experiment_aggregated.csv"
    summary_plot_path = output_dir / "scaling_experiment_summary.png"
    boxplot_path = output_dir / "scaling_experiment_boxplots.png"

    write_scaling_experiment_csv(raw_csv_path, raw_rows)
    write_aggregated_csv(agg_csv_path, aggregated_rows)
    plot_scaling_summary(summary_plot_path, aggregated_rows)
    plot_error_boxplots(boxplot_path, raw_rows)
    print_experiment_table(aggregated_rows)

    print(f"\nTemps total de l'etude : {(time.time() - total_start)/60:.2f} min")
    print("CSV tous les runs      :", raw_csv_path)
    print("CSV agrege             :", agg_csv_path)
    print("Figure moyenne ± std   :", summary_plot_path)
    print("Figure boxplots        :", boxplot_path)


if __name__ == "__main__":
    main()
