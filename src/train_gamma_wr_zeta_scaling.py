import argparse
import copy
import csv
import json
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
PARAM_NAMES = ("gamma", "wr", "zeta")

PARAM_JSON_PATHS = {
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "Qr": ("left_bc_params", "Qr"),
    "fr": ("left_bc_params", "fr"),
    "zeta": ("left_bc_params", "zeta"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibre gamma, wr et zeta sur N signaux OpenWind, avec Qr fixe, "
            "puis repete l'experience sur plusieurs tirages aleatoires."
        )
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--stage1_iter", type=int, default=300)
    parser.add_argument("--stage2_iter", type=int, default=100)
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage2_lr", type=float, default=5e-3)
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)

    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="32:8,64:16,128:32",
        help="Liste n_fft:hop, par exemple 32:8,64:16,128:32.",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument(
        "--no_stft_padding",
        action="store_true",
        help="Ignore une resolution STFT trop grande au lieu de zero-padder.",
    )

    parser.add_argument("--random_factor_min", type=float, default=0.75)
    parser.add_argument("--random_factor_max", type=float, default=1.25)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/gradient/results/gamma_wr_zeta_10x30",
    )
    parser.add_argument(
        "--save_detailed_plots",
        action="store_true",
        help="Sauvegarde les comparaisons signal par signal pour chaque repetition.",
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
        separator = ":" if ":" in item else "x" if "x" in item else None
        if separator is None:
            raise ValueError(f"Resolution STFT invalide: {item}")
        n_fft, hop = map(int, item.split(separator, maxsplit=1))
        if n_fft <= 0 or hop <= 0:
            raise ValueError(f"Resolution STFT invalide: {item}")
        resolutions.append((n_fft, hop))
    if not resolutions:
        raise ValueError("La liste --stft_resolutions est vide.")
    return tuple(resolutions)


def base_triplet_from_params(params):
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    fr = float(get_nested(params, PARAM_JSON_PATHS["fr"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    return np.asarray([gamma, 2.0 * np.pi * fr, zeta], dtype=float)


def set_triplet_json(params, triplet):
    gamma, wr, zeta = map(float, triplet)
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["fr"], wr / (2.0 * np.pi))
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    return params


def set_triplet_data(data, triplet):
    gamma, wr, zeta = triplet
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "fr", wr / (2.0 * jnp.pi), GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    return data


def set_trainable_triplet(params):
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name in ("gamma_final", "fr", "zeta")
    return params


def sample_true_triplets(base_triplet, n_signals, factor_min, factor_max, seed):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(factor_min, factor_max, size=(n_signals, 3))
    return np.asarray(base_triplet[None, :] * factors, dtype=float)


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    x_left, x_right = cell_edges_from_nodes(x_nodes)
    dt = CFL * (x_right[0] - x_left[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))
    t_solver = jnp.arange(nsteps) * dt
    snapshot_steps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot)
    ).astype(jnp.int32)
    solve_kwargs = {
        "dt": dt,
        "nsteps": nsteps,
        "bc": bc,
        "phi0": phi0,
        "y0": y0,
        "z0": z0,
        "t_solver": t_solver,
        "n_snaps": snapshot_steps,
    }
    return solve_kwargs


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
    return jnp.asarray(target / P_CLOSED, dtype=jnp.float64), ow_params


def build_targets_for_T(
    T_max,
    params,
    true_values,
    data_ref,
    c,
    bc,
    phi0,
    y0,
    z0,
    train_params,
    args,
):
    solve_kwargs = make_solver_data(
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

    targets = []
    print(f"\n=== Cibles OpenWind, T = {T_max:.4f} s ===")
    for i, true_triplet in enumerate(true_values):
        params_true = set_triplet_json(copy.deepcopy(params), true_triplet)
        target, ow_params = make_openwind_target(
            params_true,
            args.type_S,
            T_max,
            snapshot_times,
            args,
        )
        targets.append(target)
        print(
            f"signal {i + 1:2d}/{len(true_values)} | "
            f"dt_ow={ow_params['dt']:.3e} | "
            f"max={float(jnp.max(jnp.abs(target))):.3e}"
        )

    return jnp.stack(targets), solve_kwargs, snapshot_times


def make_batched_loss(
    data_init,
    geometry,
    c,
    targets,
    scales,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(theta_one, scale_one, target_one):
        data = set_triplet_data(data_init, theta_one * scale_one)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(one_loss, in_axes=(0, 0, 0))

    def loss(theta):
        return jnp.mean(vmapped_loss(theta, scales, targets))

    return loss


def make_losses_per_signal(
    data_init,
    geometry,
    c,
    targets,
    scales,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(theta_one, scale_one, target_one):
        data = set_triplet_data(data_init, theta_one * scale_one)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(one_loss, in_axes=(0, 0, 0))

    @jax.jit
    def evaluate(theta):
        return vmapped_loss(theta, scales, targets)

    return evaluate


def optimize_stage(theta, loss_fn, lr, n_iter, print_every):
    optimizer, scheduler = make_optimizer(lr, n_iter)
    opt_state = optimizer.init(theta)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    final_loss = np.nan
    for iteration in range(n_iter):
        start = time.time()
        loss_value, gradient = value_and_grad(theta)
        updates, opt_state = optimizer.update(gradient, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        final_loss = float(loss_value)

        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            print(
                f"iter {iteration:4d} | loss={final_loss:.4e} | "
                f"lr={float(scheduler(iteration)):.3e} | "
                f"t={time.time() - start:.2f}s"
            )

    return theta, final_loss


def relative_errors(estimated, true_values):
    denominator = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    errors = jnp.abs((estimated - true_values) / denominator)
    return errors, float(jnp.linalg.norm(errors))


def make_trained_predictions(data_init, geometry, c, estimated_values, solve_kwargs):
    def one_prediction(params):
        data = set_triplet_data(data_init, params)
        return forward_snapshots(data, geometry, c, **solve_kwargs)

    return jax.jit(jax.vmap(one_prediction))(estimated_values)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def repeat_fieldnames():
    return [
        "repeat_idx",
        "seed",
        "n_signals",
        "mean_relerr_gamma",
        "std_relerr_gamma",
        "mean_relerr_wr",
        "std_relerr_wr",
        "mean_relerr_zeta",
        "std_relerr_zeta",
        "global_relerr",
        "final_loss",
        "elapsed_seconds",
    ]


def signal_fieldnames():
    return [
        "repeat_idx",
        "seed",
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
        "signal_loss",
        "elapsed_seconds_repeat",
    ]


def plot_signal_comparison(path, times, targets, predictions):
    n_signals = targets.shape[0]
    n_cols = 2
    n_rows = int(np.ceil(n_signals / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(7.0 * n_cols, 3.0 * n_rows),
        squeeze=False,
        sharex=True,
    )
    for i, ax in enumerate(axes.ravel()):
        if i >= n_signals:
            ax.axis("off")
            continue
        ax.plot(times, targets[i], label="OpenWind", linewidth=1.4)
        ax.plot(times, predictions[i], "--", label="DG entraine", linewidth=1.2)
        ax.set_title(f"Signal {i + 1}")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Pression / P_closed")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Signaux OpenWind vs DG entraine")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_histograms(path, signal_rows):
    arrays = {
        r"$\gamma$": 100.0 * np.asarray([r["relerr_gamma"] for r in signal_rows]),
        r"$\omega_r$": 100.0 * np.asarray([r["relerr_wr"] for r in signal_rows]),
        r"$\zeta$": 100.0 * np.asarray([r["relerr_zeta"] for r in signal_rows]),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, (label, values) in zip(axes, arrays.items()):
        ax.hist(values, bins=20, edgecolor="black", alpha=0.8)
        mean = np.mean(values)
        median = np.median(values)
        q95 = np.quantile(values, 0.95)
        ax.axvline(mean, linestyle="--", label=f"Moyenne={mean:.2f}%")
        ax.axvline(median, linestyle=":", label=f"Mediane={median:.2f}%")
        ax.set_title(label)
        ax.set_xlabel("Erreur relative (%)")
        ax.set_ylabel("Nombre de calibrations")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.text(0.97, 0.74, f"q95={q95:.2f}%", transform=ax.transAxes, ha="right")
    fig.suptitle(f"Distribution sur {len(signal_rows)} calibrations")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_boxplots(path, signal_rows):
    values = [
        100.0 * np.asarray([r["relerr_gamma"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_wr"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_zeta"] for r in signal_rows]),
    ]
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.boxplot(values, tick_labels=[r"$\gamma$", r"$\omega_r$", r"$\zeta$"], showmeans=True)
    ax.set_ylabel("Erreur relative individuelle (%)")
    ax.set_title(f"Erreurs sur {len(signal_rows)} calibrations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_repeat_means(path, repeat_rows):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    for ax, name, label in zip(axes, PARAM_NAMES, (r"$\gamma$", r"$\omega_r$", r"$\zeta$")):
        values = 100.0 * np.asarray([row[f"mean_relerr_{name}"] for row in repeat_rows])
        ax.boxplot(values, showmeans=True)
        ax.set_title(label)
        ax.set_ylabel("Erreur moyenne du tirage (%)")
        ax.set_xticks([1], [f"N={repeat_rows[0]['n_signals']}"])
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Variabilite entre tirages aleatoires")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def run_experiment(
    repeat_idx,
    seed,
    true_values_np,
    targets_short,
    targets_long,
    data_ref,
    geometry,
    c,
    solve_kwargs_short,
    solve_kwargs_long,
    snapshot_times_long,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    args,
    output_dir,
):
    start = time.time()
    true_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    theta = DEFAULT_INIT_FACTOR * true_values / scales

    loss_short = make_batched_loss(
        data_ref,
        geometry,
        c,
        targets_short,
        scales,
        solve_kwargs_short,
        stft_resolutions,
        stft_dynamic_db,
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
        stft_dynamic_db,
        stft_allow_padding,
    )

    print("\n--- Stage 1 : T=T_max ---")
    theta, loss_stage1 = optimize_stage(
        theta, loss_long, args.stage1_lr, args.stage1_iter, args.print_every
    )
    print("\n--- Stage 2 : T=0.01 s ---")
    theta, loss_stage2 = optimize_stage(
        theta, loss_short, args.stage2_lr, args.stage2_iter, args.print_every
    )

    estimated = theta * scales
    errors, global_error = relative_errors(estimated, true_values)
    losses_fn = make_losses_per_signal(
        data_ref,
        geometry,
        c,
        targets_long,
        scales,
        solve_kwargs_long,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )
    signal_losses = np.asarray(losses_fn(theta))
    final_loss = float(np.mean(signal_losses))
    elapsed = time.time() - start

    true_np = np.asarray(true_values)
    estimated_np = np.asarray(estimated)
    errors_np = np.asarray(errors)

    signal_rows = []
    for i in range(args.n_signals):
        signal_rows.append(
            {
                "repeat_idx": repeat_idx,
                "seed": seed,
                "signal_idx": i,
                "true_gamma": float(true_np[i, 0]),
                "true_wr": float(true_np[i, 1]),
                "true_fr": float(true_np[i, 1] / (2.0 * np.pi)),
                "true_zeta": float(true_np[i, 2]),
                "estimated_gamma": float(estimated_np[i, 0]),
                "estimated_wr": float(estimated_np[i, 1]),
                "estimated_fr": float(estimated_np[i, 1] / (2.0 * np.pi)),
                "estimated_zeta": float(estimated_np[i, 2]),
                "relerr_gamma": float(errors_np[i, 0]),
                "relerr_wr": float(errors_np[i, 1]),
                "relerr_zeta": float(errors_np[i, 2]),
                "signal_loss": float(signal_losses[i]),
                "elapsed_seconds_repeat": elapsed,
            }
        )

    repeat_row = {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "n_signals": args.n_signals,
        "mean_relerr_gamma": float(np.mean(errors_np[:, 0])),
        "std_relerr_gamma": float(np.std(errors_np[:, 0])),
        "mean_relerr_wr": float(np.mean(errors_np[:, 1])),
        "std_relerr_wr": float(np.std(errors_np[:, 1])),
        "mean_relerr_zeta": float(np.mean(errors_np[:, 2])),
        "std_relerr_zeta": float(np.std(errors_np[:, 2])),
        "global_relerr": global_error,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
    }

    run_dir = output_dir / f"repeat_{repeat_idx:03d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "signals.csv", signal_rows, signal_fieldnames())
    if args.save_detailed_plots:
        predictions = make_trained_predictions(
            data_ref, geometry, c, estimated, solve_kwargs_long
        )
        plot_signal_comparison(
            run_dir / "signals_openwind_vs_dg.png",
            np.asarray(snapshot_times_long),
            np.asarray(targets_long),
            np.asarray(predictions),
        )

    print(
        f"\nResume tirage {repeat_idx + 1}: "
        f"gamma={100*repeat_row['mean_relerr_gamma']:.3f}% | "
        f"wr={100*repeat_row['mean_relerr_wr']:.3f}% | "
        f"zeta={100*repeat_row['mean_relerr_zeta']:.3f}% | "
        f"loss={final_loss:.3e} | temps={elapsed/60:.2f} min"
    )
    print(f"Loss stage 1={loss_stage1:.3e}, stage 2={loss_stage2:.3e}")
    return repeat_row, signal_rows


def print_global_summary(signal_rows, repeat_rows):
    print("\n=== Resume global ===")
    print(f"Tirages termines      : {len(repeat_rows)}")
    print(f"Calibrations totales  : {len(signal_rows)}")
    for name in PARAM_NAMES:
        values = np.asarray([row[f"relerr_{name}"] for row in signal_rows])
        print(
            f"{name:5s}: mean={100*np.mean(values):.3f}% | "
            f"std={100*np.std(values):.3f}% | "
            f"median={100*np.median(values):.3f}% | "
            f"q95={100*np.quantile(values, 0.95):.3f}% | "
            f"max={100*np.max(values):.3f}%"
        )


def main():
    args = parse_args()
    total_start = time.time()

    if args.n_signals <= 0 or args.n_repeats <= 0:
        raise ValueError("--n_signals et --n_repeats doivent etre strictement positifs.")
    if args.random_factor_min <= 0 or args.random_factor_min >= args.random_factor_max:
        raise ValueError("Bornes aleatoires invalides.")

    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding

    root = repo_root()
    with open(root / "experiments/gradient/config/simu.json", "r") as file:
        train_params = json.load(file)["solver_params"]["train"]
    with open(root / "experiments/gradient/config/param.json", "r") as file:
        params = json.load(file)

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

    print("\n=== Experience statistique ===")
    print(f"Signaux par tirage : {args.n_signals}")
    print(f"Nombre de tirages  : {args.n_repeats}")
    print(f"Qr fixe            : {float(get_nested(params, PARAM_JSON_PATHS['Qr']))}")
    print(f"STFT               : {stft_resolutions}")

    repeat_rows = []
    signal_rows = []
    repeat_csv = output_dir / "repetitions.csv"
    signal_csv = output_dir / "all_signals.csv"

    for repeat_idx in range(args.n_repeats):
        seed = args.seed + repeat_idx
        print("\n" + "#" * 80)
        print(f"TIRAGE {repeat_idx + 1}/{args.n_repeats} | seed={seed}")
        print("#" * 80)

        true_values = sample_true_triplets(
            base_triplet,
            args.n_signals,
            args.random_factor_min,
            args.random_factor_max,
            seed,
        )
        targets_short, solve_short, _ = build_targets_for_T(
            0.01,
            params,
            true_values,
            data_ref,
            c,
            bc,
            phi0,
            y0,
            z0,
            train_params,
            args,
        )
        targets_long, solve_long, times_long = build_targets_for_T(
            train_params["T_max"],
            params,
            true_values,
            data_ref,
            c,
            bc,
            phi0,
            y0,
            z0,
            train_params,
            args,
        )

        repeat_row, new_signal_rows = run_experiment(
            repeat_idx,
            seed,
            true_values,
            targets_short,
            targets_long,
            data_ref,
            geometry,
            c,
            solve_short,
            solve_long,
            times_long,
            stft_resolutions,
            args.stft_dynamic_db,
            stft_allow_padding,
            args,
            output_dir,
        )
        repeat_rows.append(repeat_row)
        signal_rows.extend(new_signal_rows)

        # Sauvegarde incrementale, utile pour une experience longue.
        write_csv(repeat_csv, repeat_rows, repeat_fieldnames())
        write_csv(signal_csv, signal_rows, signal_fieldnames())
        print_global_summary(signal_rows, repeat_rows)

    plot_histograms(output_dir / "error_histograms.png", signal_rows)
    plot_boxplots(output_dir / "error_boxplots.png", signal_rows)
    plot_repeat_means(output_dir / "repeat_mean_boxplots.png", repeat_rows)

    print_global_summary(signal_rows, repeat_rows)
    print(f"\nTemps total : {(time.time() - total_start)/3600:.2f} h")
    print(f"CSV repetitions : {repeat_csv}")
    print(f"CSV signaux     : {signal_csv}")
    print(f"histogrammes d'erreurs : {output_dir / 'error_histograms.png'}")
    print(f"boxplots d'erreurs        : {output_dir / 'error_boxplots.png'}")
    print(f"boxplots erreurs moyennes : {output_dir / 'repeat_mean_boxplots.png'}")


if __name__ == "__main__":
    main()
