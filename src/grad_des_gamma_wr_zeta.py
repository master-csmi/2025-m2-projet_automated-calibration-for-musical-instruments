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
    parser.add_argument("--n_signals", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--stage1_iter", type=int, default=200)
    parser.add_argument("--stage2_iter", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5e-3)
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
        print(f"\nSignal {i + 1}/{args.n_signals}")
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
    def loss_one_signal(theta_one, scale_one, target_one):
        params_phys = theta_one * scale_one

        data = set_triplet_data(data_init, params_phys)

        pred = forward_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )

        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    loss_all_signals = jax.vmap(
        loss_one_signal,
        in_axes=(0, 0, 0),
    )

    def loss(theta):
        losses = loss_all_signals(theta, scales, target_signals)
        return jnp.mean(losses)

    return loss

def make_batched_losses_per_signal(
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

    vmapped = jax.vmap(loss_one_signal, in_axes=(0, 0, 0))

    @jax.jit
    def losses_per_signal(theta):
        return vmapped(theta, scales, target_signals)

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


def main():
    args = parse_args()
    start_time = time.time()
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
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell
    geometry = build_solver_geometry(data_ref, train_params["Nx"], c)
    bc = BC(type="full")

    dt_long, nsteps_long, _ = make_solver_data(
        train_params["T_max"],
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    base_triplet = base_triplet_from_params(params)
    true_values_np = sample_true_triplets(
        base_triplet,
        args.n_signals,
        args.random_factor_min,
        args.random_factor_max,
        args.seed,
    )

    print("\n=== Optimisation multi-signaux gamma, wr, zeta ; Qr fixe ===")
    print("Devices JAX :", jax.devices())
    print(
        f"Train long: dt={dt_long:.6e}, nsteps={nsteps_long}, "
        f"T_max={train_params['T_max']:.4f}"
    )
    print("STFT resolutions :", stft_resolutions)
    print("STFT dynamic dB  :", args.stft_dynamic_db)
    print("STFT padding     :", stft_allow_padding)
    print("Parametres de base [gamma, wr, zeta] :", base_triplet)
    print("Qr fixe depuis param.json :", float(get_nested(params, PARAM_JSON_PATHS["Qr"])))
    print("Valeurs vraies tirees aleatoirement :")
    for i, values in enumerate(true_values_np):
        print(
            f"  signal {i}: gamma={values[0]:.6g}, "
            f"wr={values[1]:.6g}, "
            f"fr={values[1] / (2.0 * np.pi):.6g}, "
            f"zeta={values[2]:.6g}"
        )

    true_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    theta = (true_values * DEFAULT_INIT_FACTOR) / scales
    train_mask = jnp.ones((1, 3), dtype=jnp.float64)

    print("\n=== Pre-calcul des cibles OpenWind ===")
    targets_short, solve_kwargs_short, snapshot_times_short = build_targets_for_T(
        0.01,
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
    )
    targets_long, solve_kwargs_long, snapshot_times_long = build_targets_for_T(
        train_params["T_max"],
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
    )

    loss_short = make_batched_loss(
        data_ref,
        geometry,
        c,
        targets_short,
        scales,
        solve_kwargs_short,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
    )
    loss_long = make_batched_loss(
        data_ref,
        geometry,
        c,
        targets_long,
        scales,
        solve_kwargs_long,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
    )

    print("\n=== Curriculum temporel ===")
    print("\n--- Stage 1 : T = 0.01, gamma / wr / zeta ; Qr fixe ---")
    theta, loss_stage1 = optimize_stage(
        theta,
        loss_short,
        train_mask=train_mask,
        lr=args.lr*2.0,
        n_iter=args.stage1_iter,
        print_every=args.print_every,
    )

    print("\n--- Stage 2 : T = T_max, gamma / wr / zeta ; Qr fixe ---")
    theta, loss_stage2 = optimize_stage(
        theta,
        loss_long,
        train_mask=train_mask,
        lr=args.lr,
        n_iter=stage2_iter,
        print_every=args.print_every,
    )

    estimated_values = theta * scales
    err_values, err_global = relative_errors(estimated_values, true_values)
    final_loss = float(loss_long(theta))

    losses_per_signal_fn = make_batched_losses_per_signal(
        data_ref,
        geometry,
        c,
        targets_long,
        scales,
        solve_kwargs_long,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
    )

    losses_per_signal = losses_per_signal_fn(theta)

    print("\n=== Loss par signal ===")
    for i, val in enumerate(losses_per_signal):
        print(f"signal {i}: loss = {float(val):.6e}")

    print("\n=== Generation des predictions finales DG ===")
    trained_signals = make_trained_predictions(
        data_ref,
        geometry,
        c,
        estimated_values,
        solve_kwargs_long,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    os.makedirs(output_dir, exist_ok=True)

    true_values_np_out = np.asarray(true_values)
    estimated_values_np = np.asarray(estimated_values)
    err_values_np = np.asarray(err_values)
    target_signals_np = np.asarray(targets_long)
    trained_signals_np = np.asarray(trained_signals)
    snapshot_times_np = np.asarray(snapshot_times_long)

    csv_path = output_dir / "gamma_wr_zeta_results.csv"
    summary_csv_path = output_dir / "gamma_wr_zeta_summary.csv"
    params_plot_path = output_dir / "gamma_wr_zeta_params_true_vs_trained.png"
    signals_plot_path = output_dir / "gamma_wr_zeta_signals_openwind_vs_dg.png"

    write_results_csv(csv_path, true_values, estimated_values, err_values, final_loss)
    write_summary_csv(summary_csv_path, err_values)
    plot_parameter_comparison(
        params_plot_path,
        true_values_np_out,
        estimated_values_np,
        err_values_np,
    )
    plot_signal_comparison(
        signals_plot_path,
        snapshot_times_np,
        target_signals_np,
        trained_signals_np,
    )

    print("\n=== Resultats finaux ===")
    for i in range(args.n_signals):
        print(
            f"signal {i}: "
            f"true={np.asarray(true_values[i])} | "
            f"est={np.asarray(estimated_values[i])} | "
            f"relerr={np.asarray(err_values[i])}"
        )

    err_np = np.asarray(err_values)
    print("\n=== Erreurs moyennes par parametre ===")
    print(f"gamma : mean={np.mean(err_np[:, 0]):.4e}, std={np.std(err_np[:, 0]):.4e}")
    print(f"wr    : mean={np.mean(err_np[:, 1]):.4e}, std={np.std(err_np[:, 1]):.4e}")
    print(f"zeta  : mean={np.mean(err_np[:, 2]):.4e}, std={np.std(err_np[:, 2]):.4e}")

    print(f"\nLoss stage 1           : {loss_stage1:.4e}")
    print(f"Loss stage 2           : {loss_stage2:.4e}")
    print(f"Loss finale verifiee   : {final_loss:.4e}")
    print(f"Erreur relative globale: {err_global:.4e}")
    print(f"Temps total            : {time.time() - start_time:.2f}s")
    print("CSV detail             :", csv_path)
    print("CSV resume             :", summary_csv_path)
    print("Plot parametres        :", params_plot_path)
    print("Plot signaux           :", signals_plot_path)


if __name__ == "__main__":
    main()
