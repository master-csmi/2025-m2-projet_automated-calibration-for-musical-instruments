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
            "Optimisation simultanee de gamma, Qr et wr sur plusieurs signaux "
            "generes par le solveur DG."
        )
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n_iter", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)
    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="8:4,16:4,32:8,64:8,80:10,128:16,256:32",
        help=(
            "Resolutions STFT n_fft:hop separees par des virgules. "
            "Exemple: 8:4,16:4,32:8,64:8,80:10"
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
        default="experiments/gradient/results/gamma_Qr_wr_multi",
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
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    Qr = float(get_nested(params, PARAM_JSON_PATHS["Qr"]))
    fr = float(get_nested(params, PARAM_JSON_PATHS["fr"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    wr = 2.0 * np.pi * fr
    return np.asarray([gamma, Qr, wr, zeta], dtype=float)


def set_triplet_json(params, triplet):
    gamma, Qr, wr, zeta = [float(v) for v in triplet]
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["Qr"], Qr)
    set_nested(params, PARAM_JSON_PATHS["fr"], wr / (2.0 * np.pi))
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    return params


def set_triplet_data(data, triplet):
    gamma, Qr, wr, zeta = triplet
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "Qr", Qr, GEO_KEYS)
    data = set_param(data, "fr", wr / (2.0 * jnp.pi), GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    return data


def set_trainable_triplet(params):
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name in ("gamma_final", "Qr", "fr", "zeta")
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


def make_dg_target(params_true, type_S, geometry, c, solve_kwargs):
    """
    Genere une cible avec le meme solveur DG que celui utilise pendant
    l'optimisation. Cela permet de tester l'identifiabilite dans le cas DG/DG.

    Les parametres aleatoires sont deja inseres dans params_true avant l'appel.
    """
    data_true = build_physical_data(params_true, type_S)
    target = forward_snapshots(data_true, geometry, c, **solve_kwargs)

    print(
        f"  DG target: n_snapshots={target.shape[0]}, "
        f"max={float(jnp.max(jnp.abs(target))):.4e}, "
        f"std={float(jnp.std(target)):.4e}"
    )

    return jnp.asarray(target, dtype=jnp.float64)


def sample_true_triplets(base_triplet, n_signals, factor_min, factor_max, seed):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(factor_min, factor_max, size=(n_signals, 4))
    return np.asarray(base_triplet[None, :] * factors, dtype=float)


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
    n_signals = int(target_signals.shape[0])

    def loss(theta):
        params_phys = theta * scales
        total = jnp.asarray(0.0, dtype=jnp.float64)

        for signal_idx in range(n_signals):
            data = set_triplet_data(data_init, params_phys[signal_idx])
            pred = forward_snapshots(data, geometry, c, **solve_kwargs)
            total = total + loss_fn_signal(
                pred,
                target_signals[signal_idx],
                stft_resolutions=stft_resolutions,
                stft_dynamic_db=stft_dynamic_db,
                stft_allow_padding=stft_allow_padding,
            )

        return total / n_signals

    return loss

def build_loss_for_T(
    T_max,
    params,
    true_values_np,
    data_ref,
    geometry,
    c,
    bc,
    phi0,
    y0,
    z0,
    train_params,
    args,
    stft_resolutions,
    stft_allow_padding,
    scales,
):
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
    for true_triplet in true_values_np:
        params_true = set_triplet_json(copy.deepcopy(params), true_triplet)
        target = make_dg_target(
            params_true,
            args.type_S,
            geometry,
            c,
            solve_kwargs,
        )
        target_signals.append(target)

    target_signals = jnp.stack(target_signals, axis=0)

    loss_fn = make_batched_loss(
        data_ref,
        geometry,
        c,
        target_signals,
        scales,
        solve_kwargs,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
    )

    return loss_fn, solve_kwargs, snapshot_times, target_signals

def optimize_stage(theta, loss_fn, train_mask, lr, n_iter, print_every=10):
    """
    Optimise seulement les paramètres où train_mask == 1.
    theta shape: (n_signals, 4)
    train_mask shape: (1, 4) ou (n_signals, 4)
    """
    optimizer, scheduler = make_optimizer(lr, n_iter)
    opt_state = optimizer.init(theta)

    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    for iteration in range(n_iter):
        loss_val, grad_val = loss_and_grad(theta)

        # bloque les gradients des paramètres non entraînés
        grad_val = grad_val * train_mask

        updates, opt_state = optimizer.update(grad_val, opt_state, theta)
        updates = jax.tree_util.tree_map(lambda u: u * train_mask, updates)

        theta = optax.apply_updates(theta, updates)

        if iteration % print_every == 0 or iteration == n_iter - 1:
            print(f"    iter {iteration:4d} | loss = {float(loss_val):.4e}")

    return theta


def relative_errors(params_current, true_values):
    denom = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    err = jnp.abs((params_current - true_values) / denom)
    return err, float(jnp.linalg.norm(err))


def write_results_csv(path, true_values, estimated_values, err_values, final_loss):
    fieldnames = [
        "signal_idx",
        "true_gamma",
        "true_Qr",
        "true_wr",
        "true_fr",
        "true_zeta",
        "estimated_gamma",
        "estimated_Qr",
        "estimated_wr",
        "estimated_fr",
        "estimated_zeta",
        "relerr_gamma",
        "relerr_Qr",
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
                    "true_Qr": float(true_values[i, 1]),
                    "true_wr": float(true_values[i, 2]),
                    "true_zeta": float(true_values[i, 3]),
                    "true_fr": float(true_values[i, 2] / (2.0 * np.pi)),
                    "estimated_gamma": float(estimated_values[i, 0]),
                    "estimated_Qr": float(estimated_values[i, 1]),
                    "estimated_wr": float(estimated_values[i, 2]),
                    "estimated_zeta": float(estimated_values[i, 3]),
                    "estimated_fr": float(estimated_values[i, 2] / (2.0 * np.pi)),
                    "relerr_gamma": float(err_values[i, 0]),
                    "relerr_Qr": float(err_values[i, 1]),
                    "relerr_wr": float(err_values[i, 2]),
                    "relerr_zeta": float(err_values[i, 3]),
                    "final_loss": float(final_loss),
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
    labels = [r"$\gamma$", r"$Q_r$", r"$\omega_r$", r"$\zeta$"]
    n_signals = true_values.shape[0]
    x = np.arange(1, n_signals + 1)

    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), squeeze=False)
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
        ax.plot(times, target_signals[signal_idx], label="DG cible", linewidth=1.5)
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

    fig.suptitle("Signaux DG cible vs DG entraine", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    start_time = time.time()
    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding

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

    dt, nsteps, solve_kwargs = make_solver_data(
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
    snapshot_times = (solve_kwargs["n_snaps"] + 1) * solve_kwargs["dt"]

    base_triplet = base_triplet_from_params(params)
    true_values_np = sample_true_triplets(
        base_triplet,
        args.n_signals,
        args.random_factor_min,
        args.random_factor_max,
        args.seed,
    )

    print("\n=== Optimisation multi-signaux gamma, Qr, wr ===")
    print("Devices JAX :", jax.devices())
    print(f"Train: dt={dt:.6e}, nsteps={nsteps}, T_max={train_params['T_max']:.4f}")
    print("STFT resolutions :", stft_resolutions)
    print("STFT dynamic dB  :", args.stft_dynamic_db)
    print("STFT padding     :", stft_allow_padding)
    print("Parametres de base [gamma, Qr, wr, zeta] :", base_triplet)
    print("Valeurs vraies tirees aleatoirement :")
    for i, values in enumerate(true_values_np):
        print(
            f"  signal {i}: gamma={values[0]:.6g}, "
            f"Qr={values[1]:.6g}, wr={values[2]:.6g}, "
            f"fr={values[2] / (2.0 * np.pi):.6g}"
        )

    target_signals = []
    print("\n=== Generation des cibles DG ===")
    for i, true_triplet in enumerate(true_values_np):
        print(f"\nSignal {i + 1}/{args.n_signals}")
        params_true = set_triplet_json(copy.deepcopy(params), true_triplet)
        target = make_dg_target(
            params_true,
            args.type_S,
            geometry,
            c,
            solve_kwargs,
        )
        target_signals.append(target)
        print(
            f"  cible: max={float(jnp.max(jnp.abs(target))):.4e}, "
            f"std={float(jnp.std(target)):.4e}"
        )

    target_signals = jnp.stack(target_signals, axis=0)
    true_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    theta = (true_values * DEFAULT_INIT_FACTOR) / scales

    loss_fn = make_batched_loss(
        data_ref,
        geometry,
        c,
        target_signals,
        scales,
        solve_kwargs,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
    )
    loss_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    optimizer, scheduler = make_optimizer(args.lr, args.n_iter)
    opt_state = optimizer.init(theta)

    print("\n=== Entrainement simultane ===")
    print(f"{'iter':>5} | {'loss':>12} | {'err_glob':>12} | {'lr':>10} | {'t/iter':>8}")
    print("-" * 64)

    final_loss = None
    print_every = max(int(args.print_every), 1)
    print("\n=== Curriculum temporel ===")

    # Ordre des paramètres dans theta : [gamma, Qr, wr, zeta]
    mask_wr = jnp.array([[0.0, 0.0, 1.0, 0.0]])
    mask_gamma_zeta = jnp.array([[1.0, 0.0, 0.0, 1.0]])
    mask_Qr = jnp.array([[0.0, 1.0, 0.0, 0.0]])
    mask_all = jnp.array([[1.0, 1.0, 1.0, 1.0]])

    # Test : la loss avec les vrais paramètres doit être proche de zéro
    theta_true = jnp.asarray(true_values_np) / scales

    loss_at_true = loss_fn(theta_true)

    print("\n=== TEST LOSS AUX VRAIS PARAMÈTRES ===")
    print("theta_true =", theta_true)
    print("true params =", true_values_np)
    print("loss(theta_true) =", float(loss_at_true))
    # Étape 2 : temps intermédiaire.
    # On ajuste wr seul, car une erreur de fréquence produit un déphasage cumulatif.
    print("\n--- Stage 1 : T = 0.01, wr seul ---")
    loss_fn, solve_kwargs, snapshot_times, target_signals = build_loss_for_T(
        0.01, params, true_values_np, data_ref, geometry, c,
        bc, phi0, y0, z0, train_params, args,
        stft_resolutions, stft_allow_padding, scales
    )
    theta = optimize_stage(theta, loss_fn, mask_wr, args.lr * 0.2, 300, args.print_every)

    print("\n--- Stage 2 : T = 0.01, gamma / zeta ---")
    theta = optimize_stage(theta, loss_fn, mask_gamma_zeta, args.lr, 200, args.print_every)

    print("\n--- Stage 3 : T = 0.10, Qr seul ---")
    loss_fn, solve_kwargs, snapshot_times, target_signals = build_loss_for_T(
        0.10, params, true_values_np, data_ref, geometry, c,
        bc, phi0, y0, z0, train_params, args,
        stft_resolutions, stft_allow_padding, scales
    )

    
    theta = optimize_stage(theta, loss_fn, mask_Qr, args.lr * 0.2, args.n_iter, args.print_every)

    print("\n--- Stage 4 : T = 0.10, raffinement global ---")
    theta = optimize_stage(theta, loss_fn, mask_all, args.lr * 0.05, 300, args.print_every)

    estimated_values = theta * scales
    err_values, err_global = relative_errors(estimated_values, true_values)
    final_loss = float(final_loss if final_loss is not None else loss_fn(theta))
    trained_signals = make_trained_predictions(
        data_ref,
        geometry,
        c,
        estimated_values,
        solve_kwargs,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    os.makedirs(output_dir, exist_ok=True)

    true_values_np = np.asarray(true_values)
    estimated_values_np = np.asarray(estimated_values)
    err_values_np = np.asarray(err_values)
    target_signals_np = np.asarray(target_signals)
    
    trained_signals_np = np.asarray(trained_signals)
    snapshot_times_np = np.asarray(snapshot_times)

    csv_path = output_dir / "gamma_Qr_wr_multi_results.csv"
    write_results_csv(csv_path, true_values, estimated_values, err_values, final_loss)
    params_plot_path = output_dir / "gamma_Qr_wr_params_true_vs_trained.png"
    signals_plot_path = output_dir / "gamma_Qr_wr_signals_dg_vs_dg.png"
    plot_parameter_comparison(
        params_plot_path,
        true_values_np,
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

    print(f"\nErreur relative globale : {err_global:.4e}")
    print(f"Loss finale             : {final_loss:.4e}")
    print(f"Temps total             : {time.time() - start_time:.2f}s")
    print("CSV sauvegarde          :", csv_path)
    print("Plot parametres         :", params_plot_path)
    print("Plot signaux DG/DG       :", signals_plot_path)


if __name__ == "__main__":
    main()
