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
from numerics.time_integrators.rk2 import time_integrate_rk2
from physics.mouth_pressure import pressure_at_mouth_alexis


jax.config.update("jax_enable_x64", True)

P_CLOSED = 5e3
REED_OPENING = 5e-4
MIN_SCALE = 1e-8
MIN_POSITIVE = 1e-8
DEFAULT_INIT_FACTOR = 0.8
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
PARAM_NAMES = ("gamma", "wr", "zeta", "Qr")

PARAM_JSON_PATHS = {
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "Qr": ("left_bc_params", "Qr"),
    "fr": ("left_bc_params", "fr"),
    "zeta": ("left_bc_params", "zeta"),
    "alpha": ("right_bc_params", "alpha"),
    "beta": ("right_bc_params", "beta"),
    "Zt": ("right_bc_params", "Zt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibre gamma, wr, zeta et Qr sur N signaux OpenWind a partir de p(L,t) seul, "
            "puis repete l'experience sur plusieurs tirages aleatoires."
        )
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--stage1_iter", type=int, default=300)
    parser.add_argument("--stage2_iter", type=int, default=100)
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage2_lr", type=float, default=5e-3)
    parser.add_argument(
        "--stage1_lr_final_factor", type=float, default=0.8,
        help="LR final du stage 1, en fraction du LR initial.",
    )
    parser.add_argument(
        "--stage2_lr_final_factor", type=float, default=0.8,
        help="LR final du stage 2, en fraction du LR initial.",
    )
    parser.add_argument(
        "--l_lr", type=float, default=1e-4,
        help="Learning rate du reseau l(y), utilise par train_complete.py.",
    )
    parser.add_argument(
        "--l_lr_final_factor", type=float, default=0.8,
        help="LR final de l(y), en fraction de --l_lr.",
    )
    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)

    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="128:32,256:64,512:128",
        help="Liste n_fft:hop, par exemple 128:32,256:64,512:128.",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument("--pressure_weight", type=float, default=0.5)
    parser.add_argument("--reed_weight", type=float, default=0.5)
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
        default="experiments/gradient/results/Qr_wr_gamma_zeta_pressure_and_y",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="experiments/gradient/datasets/openwind_Qr_wr_gamma_zeta_300.npz",
        help="Cache OpenWind pregenere. L'entrainement ne relance pas OpenWind.",
    )
    parser.add_argument(
        "--save_detailed_plots",
        action="store_true",
        help="Sauvegarde les comparaisons signal par signal pour chaque repetition.",
    )
    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def load_openwind_dataset(path, expected):
    """Load and validate a pregenerated OpenWind training dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset OpenWind absent: {path}. Lance d'abord "
            "src/generate_openwind_Qr_wr_gamma_zeta_dataset.py."
        )

    with np.load(path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        for key, expected_value in expected.items():
            actual = metadata.get(key)
            if actual != expected_value:
                raise ValueError(
                    f"Dataset incompatible pour {key}: "
                    f"attendu={expected_value!r}, trouve={actual!r}."
                )

        arrays = {
            name: np.asarray(dataset[name])
            for name in (
                "true_values",
                "pressure_short",
                "reed_short",
                "pressure_long",
                "reed_long",
                "times_short",
                "times_long",
            )
        }
        radiation = {
            "alpha": float(dataset["radiation_alpha"]),
            "beta": float(dataset["radiation_beta"]),
        }

    expected_prefix = (expected["n_repeats"], expected["n_signals"])
    for name, array in arrays.items():
        if array.ndim >= 2 and array.shape[:2] != expected_prefix:
            raise ValueError(
                f"Forme invalide pour {name}: {array.shape}, "
                f"prefixe attendu={expected_prefix}."
            )
    return metadata, arrays, radiation


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


def params_with_openwind_radiation(base_params, ow_params, type_S):
    """Return DG parameters matched to OpenWind's radiation connector."""
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


def base_parameter_vector_from_params(params):
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    fr = float(get_nested(params, PARAM_JSON_PATHS["fr"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    Qr = float(get_nested(params, PARAM_JSON_PATHS["Qr"]))
    return np.asarray([gamma, 2.0 * np.pi * fr, zeta, Qr], dtype=float)


def set_parameter_vector_json(params, values):
    gamma, wr, zeta, Qr = map(float, values)
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["fr"], wr / (2.0 * np.pi))
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    set_nested(params, PARAM_JSON_PATHS["Qr"], Qr)
    return params


def set_parameter_vector_data(data, values):
    gamma, wr, zeta, Qr = values
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "fr", wr / (2.0 * jnp.pi), GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    data = set_param(data, "Qr", Qr, GEO_KEYS)
    return data


def set_trainable_parameters(params):
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name in ("gamma_final", "fr", "zeta", "Qr")
    return params


def sample_true_parameters(base_values, n_signals, factor_min, factor_max, seed):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(factor_min, factor_max, size=(n_signals, 4))
    return np.asarray(base_values[None, :] * factors, dtype=float)


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



def forward_complete_snapshots(data, geometry, c, **solve_kwargs):
    """Retourne simultanement p(L,t) et y(t) aux instants de snapshot."""
    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=solve_kwargs["t_solver"],
    )

    (
        _u_final,
        _phi_final,
        _y_final,
        _z_final,
        u_snaps,
        _phi_snaps,
        y_snaps,
        _z_snaps,
    ) = time_integrate_rk2(
        geometry.u0,
        geometry.x_nodes,
        c,
        solve_kwargs["dt"],
        solve_kwargs["nsteps"],
        geometry.Mp_inv,
        geometry.Mv_inv,
        solve_kwargs["bc"],
        solve_kwargs["phi0"],
        solve_kwargs["y0"],
        solve_kwargs["z0"],
        data,
        S_cells=geometry.S_cells,
        S_star=geometry.S_star,
        S_quad=geometry.S_quad,
        S_ext=geometry.S_ext,
        snapshot_steps=solve_kwargs["n_snaps"],
        gamma_target=gamma_t,
    )

    p_bell_snaps = (
        c * geometry.S_star / geometry.S_cells[-1]
    ) * u_snaps[:, -1, 0, 1]

    return p_bell_snaps, y_snaps


def normalized_mse(pred, target, eps=1e-12):
    """Mean squared error normalized by target energy."""
    numerator = jnp.mean((pred - target) ** 2)
    denominator = jnp.mean(target ** 2) + eps
    return numerator / denominator


def combined_signal_loss(
    pred_p,
    target_p,
    pred_y,
    target_y,
    pressure_weight,
    reed_weight,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    """
    Pressure loss: original temporal + multi-scale spectral loss.
    Reed loss: normalized MSE only.
    """
    loss_p = loss_fn_signal(
        pred_p,
        target_p,
        stft_resolutions=stft_resolutions,
        stft_dynamic_db=stft_dynamic_db,
        stft_allow_padding=stft_allow_padding,
    )

    loss_y = normalized_mse(pred_y, target_y)

    total = pressure_weight * loss_p + reed_weight * loss_y
    return total, loss_p, loss_y


def positive_parameters(raw_theta, scales):
    """Map unconstrained variables to strictly positive physical values."""
    positive_normalized = jax.nn.softplus(raw_theta) + MIN_POSITIVE
    return positive_normalized * scales


def inverse_softplus(x):
    """Stable inverse softplus for x > 0."""
    x = jnp.maximum(x, MIN_POSITIVE)
    return x + jnp.log(-jnp.expm1(-x))


def make_optimizer(lr, n_iter, final_lr_factor):
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_iter,
        alpha=final_lr_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=scheduler, weight_decay=1e-5),
    )
    return optimizer, scheduler


def make_openwind_target(params_true, type_S, T_max, snapshot_times, args):
    ow_l_ele = args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4
    t_ow, _, p_right, y_ow, _, ow_params = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=ow_l_ele,
        order=args.ow_order,
    )
    times = np.asarray(snapshot_times, dtype=float)
    t_ow_np = np.asarray(t_ow, dtype=float)

    target_p = np.interp(times, t_ow_np, np.asarray(p_right, dtype=float))
    target_y_physical = np.interp(
        times,
        t_ow_np,
        np.asarray(y_ow, dtype=float),
    )

    # OpenWind returns a physical reed displacement (m), whereas the DG
    # variable y is normalized by the equilibrium opening.
    target_y = target_y_physical / REED_OPENING

    return (
        jnp.asarray(target_p / P_CLOSED, dtype=jnp.float64),
        jnp.asarray(target_y, dtype=jnp.float64),
        ow_params,
    )


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

    pressure_targets = []
    reed_targets = []
    radiation_params = None
    print(f"\n=== Cibles OpenWind p(L,t) et y(t), T = {T_max:.4f} s ===")
    for i, true_parameter_vector in enumerate(true_values):
        params_true = set_parameter_vector_json(
            copy.deepcopy(params), true_parameter_vector
        )
        target_p, target_y, ow_params = make_openwind_target(
            params_true,
            args.type_S,
            T_max,
            snapshot_times,
            args,
        )
        pressure_targets.append(target_p)
        reed_targets.append(target_y)
        if radiation_params is None:
            radiation_params = ow_params
        print(
            f"signal {i + 1:2d}/{len(true_values)} | "
            f"dt_ow={ow_params['dt']:.3e} | "
            f"max|p|={float(jnp.max(jnp.abs(target_p))):.3e} | "
            f"std(y)={float(jnp.std(target_y)):.3e}"
        )

    return (
        jnp.stack(pressure_targets),
        jnp.stack(reed_targets),
        solve_kwargs,
        snapshot_times,
        radiation_params,
    )


def make_batched_loss(
    data_init,
    geometry,
    c,
    pressure_targets,
    reed_targets,
    scales,
    solve_kwargs,
    pressure_weight,
    reed_weight,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(raw_theta_one, scale_one, target_p_one, target_y_one):
        physical_values = positive_parameters(raw_theta_one, scale_one)
        data = set_parameter_vector_data(data_init, physical_values)

        pred_p, pred_y = forward_complete_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )
        pred_p = pred_p[: target_p_one.shape[0]]
        pred_y = pred_y[: target_y_one.shape[0]]

        total, _, _ = combined_signal_loss(
            pred_p,
            target_p_one,
            pred_y,
            target_y_one,
            pressure_weight,
            reed_weight,
            stft_resolutions,
            stft_dynamic_db,
            stft_allow_padding,
        )
        return total

    vmapped_loss = jax.vmap(one_loss, in_axes=(0, 0, 0, 0))

    def loss(raw_theta):
        return jnp.mean(
            vmapped_loss(
                raw_theta,
                scales,
                pressure_targets,
                reed_targets,
            )
        )

    return loss


def make_losses_per_signal(
    data_init,
    geometry,
    c,
    pressure_targets,
    reed_targets,
    scales,
    solve_kwargs,
    pressure_weight,
    reed_weight,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(raw_theta_one, scale_one, target_p_one, target_y_one):
        physical_values = positive_parameters(raw_theta_one, scale_one)
        data = set_parameter_vector_data(data_init, physical_values)

        pred_p, pred_y = forward_complete_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )
        pred_p = pred_p[: target_p_one.shape[0]]
        pred_y = pred_y[: target_y_one.shape[0]]

        return combined_signal_loss(
            pred_p,
            target_p_one,
            pred_y,
            target_y_one,
            pressure_weight,
            reed_weight,
            stft_resolutions,
            stft_dynamic_db,
            stft_allow_padding,
        )

    vmapped_loss = jax.vmap(one_loss, in_axes=(0, 0, 0, 0))

    @jax.jit
    def evaluate(raw_theta):
        return vmapped_loss(
            raw_theta,
            scales,
            pressure_targets,
            reed_targets,
        )

    return evaluate


def optimize_stage(theta, loss_fn, lr, n_iter, print_every, final_lr_factor):
    optimizer, scheduler = make_optimizer(lr, n_iter, final_lr_factor)
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
        data = set_parameter_vector_data(data_init, params)
        return forward_complete_snapshots(data, geometry, c, **solve_kwargs)

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
        "mean_relerr_Qr",
        "std_relerr_Qr",
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
        "true_Qr",
        "estimated_gamma",
        "estimated_wr",
        "estimated_fr",
        "estimated_zeta",
        "estimated_Qr",
        "relerr_gamma",
        "relerr_wr",
        "relerr_zeta",
        "relerr_Qr",
        "signal_loss",
        "pressure_loss",
        "reed_loss",
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



def plot_reed_comparison(path, times, targets, predictions):
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
        ax.set_ylabel("Ouverture y")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("Ouverture de l'anche OpenWind vs DG entraine")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_histograms(path, signal_rows):
    arrays = {
        r"$\gamma$": 100.0 * np.asarray(
            [r["relerr_gamma"] for r in signal_rows]
        ),
        r"$\omega_r$": 100.0 * np.asarray(
            [r["relerr_wr"] for r in signal_rows]
        ),
        r"$\zeta$": 100.0 * np.asarray(
            [r["relerr_zeta"] for r in signal_rows]
        ),
        r"$Q_r$": 100.0 * np.asarray(
            [r["relerr_Qr"] for r in signal_rows]
        ),
    }

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4))

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
        ax.text(
            0.97,
            0.74,
            f"q95={q95:.2f}%",
            transform=ax.transAxes,
            ha="right",
        )

    fig.suptitle(f"Distribution sur {len(signal_rows)} calibrations")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_boxplots(path, signal_rows):
    values = [
        100.0 * np.asarray([r["relerr_gamma"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_wr"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_zeta"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_Qr"] for r in signal_rows]),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.boxplot(
        values,
        tick_labels=[
            r"$\gamma$",
            r"$\omega_r$",
            r"$\zeta$",
            r"$Q_r$",
        ],
        showmeans=True,
    )
    ax.set_ylabel("Erreur relative individuelle (%)")
    ax.set_title(f"Erreurs sur {len(signal_rows)} calibrations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_repeat_means(path, repeat_rows):
    labels = (
        r"$\gamma$",
        r"$\omega_r$",
        r"$\zeta$",
        r"$Q_r$",
    )

    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.4))

    for ax, name, label in zip(axes, PARAM_NAMES, labels):
        values = 100.0 * np.asarray(
            [row[f"mean_relerr_{name}"] for row in repeat_rows]
        )
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
    pressure_targets_short,
    reed_targets_short,
    pressure_targets_long,
    reed_targets_long,
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
    initial_normalized = DEFAULT_INIT_FACTOR * true_values / scales
    theta = inverse_softplus(initial_normalized)

    loss_short = make_batched_loss(
        data_ref,
        geometry,
        c,
        pressure_targets_short,
        reed_targets_short,
        scales,
        solve_kwargs_short,
        args.pressure_weight,
        args.reed_weight,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )
    loss_long = make_batched_loss(
        data_ref,
        geometry,
        c,
        pressure_targets_long,
        reed_targets_long,
        scales,
        solve_kwargs_long,
        args.pressure_weight,
        args.reed_weight,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )

    print(f"\n--- Stage 1 : T={generation_T:.4f} s (signal généré) ---")
    theta, loss_stage1 = optimize_stage(
        theta, loss_long, args.stage1_lr, args.stage1_iter, args.print_every,
        args.stage1_lr_final_factor,
    )
    if args.skip_stage2:
        loss_stage2 = float(loss_short(theta))
        print("\n--- Stage 2 ignore (--skip_stage2) ---")
    else:
        print("\n--- Stage 2 : T=0.01 s ---")
        theta, loss_stage2 = optimize_stage(
            theta, loss_short, args.stage2_lr, args.stage2_iter, args.print_every,
            args.stage2_lr_final_factor,
        )

    estimated = positive_parameters(theta, scales)
    errors, global_error = relative_errors(estimated, true_values)
    losses_fn = make_losses_per_signal(
        data_ref,
        geometry,
        c,
        pressure_targets_long,
        reed_targets_long,
        scales,
        solve_kwargs_long,
        args.pressure_weight,
        args.reed_weight,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )
    loss_components = losses_fn(theta)
    signal_losses = np.asarray(loss_components[0])
    pressure_losses = np.asarray(loss_components[1])
    reed_losses = np.asarray(loss_components[2])
    final_loss = float(np.mean(signal_losses))
    elapsed = time.time() - start

    true_np = np.asarray(true_values)
    estimated_np = np.asarray(estimated)
    errors_np = np.asarray(errors)
    batch_size = true_np.shape[0]

    signal_rows = []
    for i in range(batch_size):
        signal_rows.append(
            {
                "repeat_idx": repeat_idx,
                "seed": seed,
                "signal_idx": i,
                "true_gamma": float(true_np[i, 0]),
                "true_wr": float(true_np[i, 1]),
                "true_fr": float(true_np[i, 1] / (2.0 * np.pi)),
                "true_zeta": float(true_np[i, 2]),
                "true_Qr": float(true_np[i, 3]),
                "estimated_gamma": float(estimated_np[i, 0]),
                "estimated_wr": float(estimated_np[i, 1]),
                "estimated_fr": float(estimated_np[i, 1] / (2.0 * np.pi)),
                "estimated_zeta": float(estimated_np[i, 2]),
                "estimated_Qr": float(estimated_np[i, 3]),
                "relerr_gamma": float(errors_np[i, 0]),
                "relerr_wr": float(errors_np[i, 1]),
                "relerr_zeta": float(errors_np[i, 2]),
                "relerr_Qr": float(errors_np[i, 3]),
                "signal_loss": float(signal_losses[i]),
                "pressure_loss": float(pressure_losses[i]),
                "reed_loss": float(reed_losses[i]),
                "elapsed_seconds_repeat": elapsed,
            }
        )

    repeat_row = {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "n_signals": batch_size,
        "mean_relerr_gamma": float(np.mean(errors_np[:, 0])),
        "std_relerr_gamma": float(np.std(errors_np[:, 0])),
        "mean_relerr_wr": float(np.mean(errors_np[:, 1])),
        "std_relerr_wr": float(np.std(errors_np[:, 1])),
        "mean_relerr_zeta": float(np.mean(errors_np[:, 2])),
        "std_relerr_zeta": float(np.std(errors_np[:, 2])),
        "mean_relerr_Qr": float(np.mean(errors_np[:, 3])),
        "std_relerr_Qr": float(np.std(errors_np[:, 3])),
        "global_relerr": global_error,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
    }

    run_dir = output_dir / f"repeat_{repeat_idx:03d}_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(run_dir / "signals.csv", signal_rows, signal_fieldnames())
    if args.save_detailed_plots:
        pressure_predictions, reed_predictions = make_trained_predictions(
            data_ref, geometry, c, estimated, solve_kwargs_long
        )
        pressure_predictions = pressure_predictions[:, : pressure_targets_long.shape[1]]
        reed_predictions = reed_predictions[:, : reed_targets_long.shape[1]]
        plot_signal_comparison(
            run_dir / "pressure_openwind_vs_dg.png",
            np.asarray(snapshot_times_long),
            np.asarray(pressure_targets_long),
            np.asarray(pressure_predictions),
        )
        plot_reed_comparison(
            run_dir / "reed_openwind_vs_dg.png",
            np.asarray(snapshot_times_long),
            np.asarray(reed_targets_long),
            np.asarray(reed_predictions),
        )

    print(
        f"\nResume tirage {repeat_idx + 1}: "
        f"gamma={100*repeat_row['mean_relerr_gamma']:.3f}% | "
        f"wr={100*repeat_row['mean_relerr_wr']:.3f}% | "
        f"zeta={100*repeat_row['mean_relerr_zeta']:.3f}% | "
        f"Qr={100*repeat_row['mean_relerr_Qr']:.3f}% | "
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
    for name, factor in (
        ("stage1_lr_final_factor", args.stage1_lr_final_factor),
        ("stage2_lr_final_factor", args.stage2_lr_final_factor),
        ("l_lr_final_factor", args.l_lr_final_factor),
    ):
        if not 0.0 <= factor <= 1.0:
            raise ValueError(f"--{name} doit etre compris entre 0 et 1.")

    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding

    root = repo_root()
    with open(root / "experiments/gradient/config/simu.json", "r") as file:
        train_params = json.load(file)["solver_params"]["train"]
    with open(root / "experiments/gradient/config/param.json", "r") as file:
        params = json.load(file)

    generation_T = float(train_params["T_max"])
    calibration_T = float(train_params.get("T_long", generation_T))
    if calibration_T <= 0.0 or generation_T <= 0.0:
        raise ValueError("Les fenetres temporelles doivent etre positives.")
    if calibration_T > generation_T:
        raise ValueError("T_long doit etre inferieur ou egal a T_max.")

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    params_trainable = set_trainable_parameters(params)
    data_ref = build_physical_data(params_trainable, args.type_S)
    geometry = build_solver_geometry(data_ref, train_params["Nx"], c)
    bc = BC(type="full")
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = root / dataset_path

    expected_dataset = {
        "format_version": 1,
        "type_S": args.type_S,
        "n_signals": args.n_signals,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "random_factor_min": args.random_factor_min,
        "random_factor_max": args.random_factor_max,
        "T_short": 0.01,
        "T_long": generation_T,
        "Nx": int(train_params["Nx"]),
        "N_snapshot": int(train_params["N_snapshot"]),
        "cfl": float(train_params["cfl"]),
        "c": float(c),
        "ow_order": int(args.ow_order),
        "ow_theta": float(args.ow_theta),
        "ow_l_ele": float(args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4),
    }
    dataset_metadata, dataset_arrays, radiation = load_openwind_dataset(
        dataset_path,
        expected_dataset,
    )

    long_keep = int(np.searchsorted(dataset_arrays["times_long"], calibration_T, side="right"))
    long_keep = max(1, long_keep)

    length = data_ref.section.L_tube + data_ref.section.L_bell
    solve_short = make_solver_data(
        0.01, train_params["cfl"], train_params["Nx"],
        train_params["N_snapshot"], length, c, bc, phi0, y0, z0,
    )
    solve_long = make_solver_data(
        train_params["T_max"], train_params["cfl"], train_params["Nx"],
        train_params["N_snapshot"], length, c, bc, phi0, y0, z0,
    )
    times_long = (solve_long["n_snaps"] + 1) * solve_long["dt"]

    matched_params = params_with_openwind_radiation(
        params_trainable,
        radiation,
        args.type_S,
    )
    data_matched = build_physical_data(matched_params, args.type_S)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Experience statistique ===")
    total_signals = args.n_repeats * args.n_signals
    print(f"Signaux dans le batch unique : {total_signals}")
    print("Parametres calibres : gamma, wr, zeta, Qr")
    print("Observation         : p(L,t) et y(t)")
    print(f"Poids loss          : p={args.pressure_weight}, y={args.reed_weight}")
    print(f"STFT               : {stft_resolutions}")
    print(
        "LR stage 1          : "
        f"{args.stage1_lr:.3e} -> "
        f"{args.stage1_lr * args.stage1_lr_final_factor:.3e}"
    )
    print(
        "LR stage 2          : "
        f"{args.stage2_lr:.3e} -> "
        f"{args.stage2_lr * args.stage2_lr_final_factor:.3e}"
    )
    print(
        "LR l(y)             : "
        f"{args.l_lr:.3e} -> {args.l_lr * args.l_lr_final_factor:.3e}"
    )
    print(f"Dataset OpenWind    : {dataset_path}")
    print(f"Simulations cachees : {dataset_metadata['n_repeats'] * dataset_metadata['n_signals']}")
    print(f"Calibration T_long  : {calibration_T:.4f} s (prefix of T_max={generation_T:.4f} s)")
    print("\n=== Radiation DG appariee a OpenWind ===")
    print(f"alpha={float(data_matched.alpha):.12g}")
    print(f"beta ={float(data_matched.beta):.12g}")
    print(f"Zt   ={float(data_matched.Zt):.12g}")

    repeat_rows = []
    signal_rows = []
    repeat_csv = output_dir / "repetitions.csv"
    signal_csv = output_dir / "all_signals.csv"

    print("\n" + "#" * 80)
    print(f"ENTRAINEMENT UNIQUE SUR {total_signals} SIGNAUX")
    print("#" * 80)
    true_values = jnp.asarray(dataset_arrays["true_values"].reshape(-1, 4))
    pressure_targets_short = jnp.asarray(
        dataset_arrays["pressure_short"].reshape(total_signals, -1)
    )
    reed_targets_short = jnp.asarray(
        dataset_arrays["reed_short"].reshape(total_signals, -1)
    )
    pressure_targets_long = jnp.asarray(
        dataset_arrays["pressure_long"].reshape(total_signals, -1)[:, :long_keep]
    )
    reed_targets_long = jnp.asarray(
        dataset_arrays["reed_long"].reshape(total_signals, -1)[:, :long_keep]
    )

    repeat_row, signal_rows = run_experiment(
        0, args.seed, true_values,
        pressure_targets_short, reed_targets_short,
        pressure_targets_long, reed_targets_long,
        data_matched, geometry, c, solve_short, solve_long, times_long[:long_keep],
        stft_resolutions, args.stft_dynamic_db, stft_allow_padding,
        args, output_dir,
    )
    repeat_rows = [repeat_row]
    write_csv(repeat_csv, repeat_rows, repeat_fieldnames())
    write_csv(signal_csv, signal_rows, signal_fieldnames())

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
