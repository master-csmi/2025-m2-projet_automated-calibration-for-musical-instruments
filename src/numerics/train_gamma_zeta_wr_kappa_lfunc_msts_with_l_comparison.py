
import os
import csv
import json
import time
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from utils.parse_args import parse_args
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.solve import forward_snapshots
from utils.param_func import set_param
from inverse.total_loss import loss_fn_signal
from inverse.l_func_nn import LFuncNN


jax.config.update("jax_enable_x64", True)
print(jax.devices())


# =============================================================================
# Configuration
# =============================================================================

N_SIGNALS = 200
N_EPOCHS_PER_SIGNAL = 10
N_CYCLES = 2

LR_PARAMS = 5e-3
LR_LFUNC = 1e-4
MIN_LR_FACTOR = 0.1

REG_WEIGHT = 1e-3
GRAD_CLIP_PARAMS = 1.0
GRAD_CLIP_LFUNC = 0.1

PRINT_EVERY_SIGNAL = 10
SEED = 0

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

PARAM_RANGES = {
    "gamma_final": (0.20, 0.80),
    "fr": (120.0, 240.0),
    "zeta": (0.20, 0.80),
    "kappa": (0.40, 1.20),
}

INIT_GAMMA = 0.35
INIT_FR = 160.0
INIT_ZETA = 0.35
INIT_KAPPA = 0.70

PARAM_NAMES = ("gamma_final", "fr", "zeta", "kappa")


@dataclass
class SignalExample:
    data_true: object
    target_pressure: jax.Array
    true_gamma: float
    true_fr: float
    true_zeta: float
    true_kappa: float


def replace_l(data, ell_nn):
    return eqx.tree_at(lambda d: d.l, data, ell_nn)


def set_sampled_params(data, sampled_params):
    for name, value in sampled_params.items():
        data = set_param(
            data,
            name,
            jnp.asarray(value, dtype=jnp.float64),
            GEO_KEYS,
        )
    return data


def sample_physical_params(key):
    keys = jax.random.split(key, len(PARAM_RANGES))
    sampled = {}

    for subkey, (name, (vmin, vmax)) in zip(keys, PARAM_RANGES.items()):
        value = jax.random.uniform(
            subkey,
            shape=(),
            minval=vmin,
            maxval=vmax,
            dtype=jnp.float64,
        )
        sampled[name] = float(value)

    return sampled


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps, dtype=jnp.float64) * dt
    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot)
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


def softplus_inverse(x, eps=1e-10):
    x = jnp.maximum(jnp.asarray(x, dtype=jnp.float64), eps)
    return x + jnp.log(-jnp.expm1(-x))


def raw_to_physical(raw_theta):
    positive = jax.nn.softplus(raw_theta) + 1e-8
    gamma = positive[0]
    fr = positive[1]
    zeta = positive[2]
    kappa = positive[3]
    return gamma, fr, zeta, kappa


def initial_raw_theta():
    physical = jnp.asarray(
        [INIT_GAMMA, INIT_FR, INIT_ZETA, INIT_KAPPA],
        dtype=jnp.float64,
    )
    return softplus_inverse(physical)


def apply_estimated_parameters(data, raw_theta):
    gamma, fr, zeta, kappa = raw_to_physical(raw_theta)

    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "fr", fr, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)

    return data


def physical_regularization(ell_nn):
    y_neg = jnp.linspace(-0.2, 0.0, 64)
    y_pos = jnp.linspace(0.0, 1.5, 128)

    pred_neg = jax.vmap(ell_nn)(y_neg)
    pred_pos = jax.vmap(ell_nn)(y_pos)

    loss_closed = jnp.mean(pred_neg**2)
    loss_positive = jnp.mean(jax.nn.relu(-pred_pos) ** 2)

    dy = y_pos[1] - y_pos[0]
    finite_diff = (pred_pos[1:] - pred_pos[:-1]) / dy
    loss_monotone = jnp.mean(jax.nn.relu(-finite_diff) ** 2)

    loss_anchor = jnp.square(ell_nn(jnp.asarray(1.0)) - 1.0)

    return (
        loss_closed
        + 0.1 * loss_positive
        + 0.1 * loss_monotone
        + loss_anchor
    )


def loss_one_signal(
    trainable,
    data_template,
    target_pressure,
    Nx,
    c,
    solve_kwargs,
    reg_weight,
):
    """
    trainable = (raw_theta_signal, ell_nn)

    Observation loss: MSTS on p(L,t) only.
    """
    raw_theta, ell_nn = trainable

    data = apply_estimated_parameters(data_template, raw_theta)
    data = replace_l(data, ell_nn)

    pred_pressure = forward_snapshots(
        data,
        Nx,
        c,
        **solve_kwargs,
    )

    loss_msts = loss_fn_signal(
        pred_pressure,
        target_pressure,
    )

    loss_reg = physical_regularization(ell_nn)
    total = loss_msts + reg_weight * loss_reg

    return total, (loss_msts, loss_reg)


def make_optimizer(lr, total_steps, clip_norm):
    schedule = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=max(total_steps, 1),
        alpha=MIN_LR_FACTOR,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(
            learning_rate=schedule,
            weight_decay=1e-7,
        ),
    )

    return optimizer, schedule


def run_joint_training(
    ell_nn,
    raw_params,
    dataset,
    Nx,
    c,
    solve_kwargs,
):
    total_steps = N_CYCLES * N_EPOCHS_PER_SIGNAL * len(dataset)

    optimizer_params, schedule_params = make_optimizer(
        LR_PARAMS,
        total_steps,
        GRAD_CLIP_PARAMS,
    )
    optimizer_lfunc, schedule_lfunc = make_optimizer(
        LR_LFUNC,
        total_steps,
        GRAD_CLIP_LFUNC,
    )

    param_states = [
        optimizer_params.init(raw_params[i])
        for i in range(len(dataset))
    ]
    lfunc_state = optimizer_lfunc.init(
        eqx.filter(ell_nn, eqx.is_array)
    )

    def wrapped_loss(trainable, data_template, target_pressure):
        return loss_one_signal(
            trainable=trainable,
            data_template=data_template,
            target_pressure=target_pressure,
            Nx=Nx,
            c=c,
            solve_kwargs=solve_kwargs,
            reg_weight=REG_WEIGHT,
        )

    loss_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(
            wrapped_loss,
            has_aux=True,
        )
    )

    history = {
        "iteration": [],
        "cycle": [],
        "signal": [],
        "total_loss": [],
        "msts_loss": [],
        "regularization": [],
        "lr_params": [],
        "lr_lfunc": [],
    }

    global_step = 0

    print("\n=== Joint training: gamma, fr, zeta, kappa and l(y) ===")
    print(f"N signals      = {len(dataset)}")
    print(f"cycles         = {N_CYCLES}")
    print(f"epochs/signal  = {N_EPOCHS_PER_SIGNAL}")
    print("observation loss = MSTS on p(L,t) only")

    for cycle in range(N_CYCLES):
        print(f"\n--- Cycle {cycle + 1}/{N_CYCLES} ---")

        for signal_idx, example in enumerate(dataset):
            raw_theta = raw_params[signal_idx]

            for _ in range(N_EPOCHS_PER_SIGNAL):
                trainable = (raw_theta, ell_nn)

                (loss_value, aux), grads = loss_and_grad(
                    trainable,
                    example.data_true,
                    example.target_pressure,
                )

                loss_msts, loss_reg = aux
                grad_theta, grad_lfunc = grads

                updates_theta, new_param_state = optimizer_params.update(
                    grad_theta,
                    param_states[signal_idx],
                    raw_theta,
                )
                raw_theta = optax.apply_updates(
                    raw_theta,
                    updates_theta,
                )
                param_states[signal_idx] = new_param_state

                filtered_lfunc = eqx.filter(ell_nn, eqx.is_array)
                updates_lfunc, lfunc_state = optimizer_lfunc.update(
                    grad_lfunc,
                    lfunc_state,
                    filtered_lfunc,
                )
                ell_nn = eqx.apply_updates(
                    ell_nn,
                    updates_lfunc,
                )

                history["iteration"].append(global_step)
                history["cycle"].append(cycle)
                history["signal"].append(signal_idx)
                history["total_loss"].append(float(loss_value))
                history["msts_loss"].append(float(loss_msts))
                history["regularization"].append(float(loss_reg))
                history["lr_params"].append(float(schedule_params(global_step)))
                history["lr_lfunc"].append(float(schedule_lfunc(global_step)))

                global_step += 1

            raw_params = raw_params.at[signal_idx].set(raw_theta)

            if (
                signal_idx % PRINT_EVERY_SIGNAL == 0
                or signal_idx == len(dataset) - 1
            ):
                gamma_est, fr_est, zeta_est, kappa_est = raw_to_physical(raw_theta)
                print(
                    f"signal={signal_idx:03d} | "
                    f"loss={float(loss_value):.4e} | "
                    f"MSTS={float(loss_msts):.4e} | "
                    f"gamma={float(gamma_est):.4f} | "
                    f"fr={float(fr_est):.2f} Hz | "
                    f"zeta={float(zeta_est):.4f} | "
                    f"kappa={float(kappa_est):.4f}"
                )

    return ell_nn, raw_params, history


def save_history(history, path):
    fieldnames = list(history.keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(len(history["iteration"])):
            writer.writerow({
                name: history[name][i]
                for name in fieldnames
            })


def save_parameter_results(raw_params, dataset, path):
    rows = []

    for i, example in enumerate(dataset):
        gamma_est, fr_est, zeta_est, kappa_est = raw_to_physical(raw_params[i])

        true_wr = 2.0 * np.pi * example.true_fr
        est_wr = 2.0 * np.pi * float(fr_est)

        rows.append({
            "signal_idx": i,
            "true_gamma": example.true_gamma,
            "true_fr": example.true_fr,
            "true_wr": true_wr,
            "true_zeta": example.true_zeta,
            "true_kappa": example.true_kappa,
            "estimated_gamma": float(gamma_est),
            "estimated_fr": float(fr_est),
            "estimated_wr": est_wr,
            "estimated_zeta": float(zeta_est),
            "estimated_kappa": float(kappa_est),
            "relerr_gamma": abs(float(gamma_est) - example.true_gamma)
                            / abs(example.true_gamma),
            "relerr_fr": abs(float(fr_est) - example.true_fr)
                         / abs(example.true_fr),
            "relerr_wr": abs(est_wr - true_wr) / abs(true_wr),
            "relerr_zeta": abs(float(zeta_est) - example.true_zeta)
                           / abs(example.true_zeta),
            "relerr_kappa": abs(float(kappa_est) - example.true_kappa)
                            / abs(example.true_kappa),
        })

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    return rows


def evaluate_l_function(ell_nn, l_true, y_min=-0.2, y_max=1.5, n_points=400):
    """
    Compare the learned law l_theta(y) with the default DG law l(y).

    Returns the evaluation grid, both curves, and several error indicators.
    """
    y_grid = jnp.linspace(y_min, y_max, n_points)

    ell_pred = jax.vmap(ell_nn)(y_grid)
    ell_true = jax.vmap(l_true)(y_grid)

    difference = ell_pred - ell_true

    rel_l2 = (
        jnp.linalg.norm(difference)
        / (jnp.linalg.norm(ell_true) + 1e-12)
    )
    rmse = jnp.sqrt(jnp.mean(difference**2))
    max_abs_error = jnp.max(jnp.abs(difference))

    return y_grid, ell_true, ell_pred, {
        "relative_l2_error": float(rel_l2),
        "rmse": float(rmse),
        "max_absolute_error": float(max_abs_error),
    }


def plot_l_function_comparison(ell_nn, l_true, path, metrics_path):
    """
    Save a dedicated comparison plot between l_theta(y) and the default DG l(y),
    together with a CSV file containing the error metrics.
    """
    y_grid, ell_true, ell_pred, metrics = evaluate_l_function(
        ell_nn,
        l_true,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_true),
        label=r"Default DG law $\ell(y)$",
        linewidth=2.3,
    )
    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_pred),
        label=r"Learned law $\ell_\theta(y)$",
        linewidth=2.0,
        linestyle="--",
    )

    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$\ell(y)$")
    ax.set_title(
        "Comparison of the reed-opening laws\n"
        rf"relative $L^2$ error = {100.0 * metrics['relative_l2_error']:.3f}\%"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "relative_l2_error",
                "relative_l2_error_percent",
                "rmse",
                "max_absolute_error",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "relative_l2_error": metrics["relative_l2_error"],
            "relative_l2_error_percent": 100.0 * metrics["relative_l2_error"],
            "rmse": metrics["rmse"],
            "max_absolute_error": metrics["max_absolute_error"],
        })

    print("\n=== Error on the learned reed-opening law ===")
    print(
        f"Relative L2 error : "
        f"{100.0 * metrics['relative_l2_error']:.3f}%"
    )
    print(f"RMSE              : {metrics['rmse']:.6e}")
    print(
        f"Maximum abs. error: "
        f"{metrics['max_absolute_error']:.6e}"
    )

    return metrics


def plot_training_summary(history, ell_nn, l_true, rows, path):
    y_grid, ell_true, ell_pred, l_metrics = evaluate_l_function(
        ell_nn,
        l_true,
    )
    rel_err_l = l_metrics["relative_l2_error"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    axes[0, 0].semilogy(
        history["iteration"],
        history["msts_loss"],
    )
    axes[0, 0].set_title("MSTS training loss")
    axes[0, 0].set_xlabel("Iteration")
    axes[0, 0].set_ylabel("MSTS")
    axes[0, 0].grid(True)

    axes[0, 1].plot(
        np.asarray(y_grid),
        np.asarray(ell_true),
        label="True l(y)",
        linewidth=2,
    )
    axes[0, 1].plot(
        np.asarray(y_grid),
        np.asarray(ell_pred),
        label=f"Learned l(y), rel. error={100*rel_err_l:.2f}%",
        linewidth=2,
    )
    axes[0, 1].set_xlabel("y")
    axes[0, 1].set_ylabel("l(y)")
    axes[0, 1].set_title("Shared reed-opening law")
    axes[0, 1].grid(True)
    axes[0, 1].legend()

    relerr_gamma = 100.0 * np.asarray([r["relerr_gamma"] for r in rows])
    relerr_wr = 100.0 * np.asarray([r["relerr_wr"] for r in rows])
    relerr_zeta = 100.0 * np.asarray([r["relerr_zeta"] for r in rows])
    relerr_kappa = 100.0 * np.asarray([r["relerr_kappa"] for r in rows])

    axes[1, 0].boxplot(
        [relerr_gamma, relerr_wr, relerr_zeta, relerr_kappa],
        tick_labels=[r"$\gamma$", r"$\omega_r$", r"$\zeta$", r"$\kappa$"],
        showfliers=True,
    )
    axes[1, 0].set_ylabel("Relative error (%)")
    axes[1, 0].set_title("Parameter calibration errors")
    axes[1, 0].grid(True, axis="y")

    axes[1, 1].scatter(
        [r["true_fr"] for r in rows],
        [r["estimated_fr"] for r in rows],
        s=16,
        alpha=0.7,
    )
    min_fr = min(r["true_fr"] for r in rows)
    max_fr = max(r["true_fr"] for r in rows)
    axes[1, 1].plot(
        [min_fr, max_fr],
        [min_fr, max_fr],
        linestyle="--",
    )
    axes[1, 1].set_xlabel("True fr (Hz)")
    axes[1, 1].set_ylabel("Estimated fr (Hz)")
    axes[1, 1].set_title("True vs estimated reed frequency")
    axes[1, 1].grid(True)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Relative error on l(y): {100*rel_err_l:.3f}%")


def main():
    start_time = time.time()

    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    train_params = solver_params["train"]

    T_max = train_params["T_max"]
    CFL = train_params["cfl"]
    Nx = train_params["Nx"]
    N_snapshot = train_params["N_snapshot"]

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    args = parse_args()
    type_S = args.type_S

    result_dir = "../experiments/gradient/results/train_gamma_zeta_wr_lfunc_msts"
    os.makedirs(result_dir, exist_ok=True)

    for name in params["trainable"]:
        params["trainable"][name] = name in PARAM_NAMES

    data_ref = build_physical_data(params, type_S)
    l_true = data_ref.l
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell

    bc = BC(type="full")

    dt, nsteps, solve_kwargs = make_solver_data(
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
    )

    print(
        f"Simulation: dt={dt:.6e}, nsteps={nsteps}, "
        f"T_max={T_max:.4f}, Nx={Nx}"
    )

    print("\n=== Target generation ===")

    key = jax.random.PRNGKey(SEED)
    forward_jit = eqx.filter_jit(forward_snapshots)
    dataset = []

    for signal_idx in range(N_SIGNALS):
        key, sample_key = jax.random.split(key)
        sampled = sample_physical_params(sample_key)

        data_true = build_physical_data(params, type_S)
        data_true = set_sampled_params(data_true, sampled)

        target_pressure = forward_jit(
            data_true,
            Nx,
            c,
            **solve_kwargs,
        )

        dataset.append(
            SignalExample(
                data_true=data_true,
                target_pressure=target_pressure,
                true_gamma=sampled["gamma_final"],
                true_fr=sampled["fr"],
                true_zeta=sampled["zeta"],
                true_kappa=sampled["kappa"],
            )
        )

        if (
            signal_idx % PRINT_EVERY_SIGNAL == 0
            or signal_idx == N_SIGNALS - 1
        ):
            print(
                f"signal={signal_idx:03d} | "
                f"gamma={sampled['gamma_final']:.4f} | "
                f"fr={sampled['fr']:.2f} Hz | "
                f"zeta={sampled['zeta']:.4f} | "
                f"kappa={sampled['kappa']:.4f} | "
                f"max|p|={float(jnp.max(jnp.abs(target_pressure))):.4e}"
            )

    key, nn_key = jax.random.split(key)

    ell_nn = LFuncNN(
        [1, 8, 8, 1],
        activation=jax.nn.tanh,
        key=nn_key,
    )

    raw0 = initial_raw_theta()
    raw_params = jnp.tile(raw0[None, :], (N_SIGNALS, 1))

    ell_nn, raw_params, history = run_joint_training(
        ell_nn=ell_nn,
        raw_params=raw_params,
        dataset=dataset,
        Nx=Nx,
        c=c,
        solve_kwargs=solve_kwargs,
    )

    history_path = os.path.join(result_dir, "training_history.csv")
    results_path = os.path.join(result_dir, "all_signals.csv")
    figure_path = os.path.join(result_dir, "training_summary.png")
    l_comparison_path = os.path.join(
        result_dir,
        "l_theta_vs_default_l.png",
    )
    l_metrics_path = os.path.join(
        result_dir,
        "l_function_errors.csv",
    )
    network_path = os.path.join(result_dir, "ell_nn.eqx")
    raw_params_path = os.path.join(result_dir, "raw_signal_parameters.npy")

    save_history(history, history_path)
    rows = save_parameter_results(
        raw_params,
        dataset,
        results_path,
    )

    plot_training_summary(
        history,
        ell_nn,
        l_true,
        rows,
        figure_path,
    )

    plot_l_function_comparison(
        ell_nn=ell_nn,
        l_true=l_true,
        path=l_comparison_path,
        metrics_path=l_metrics_path,
    )

    eqx.tree_serialise_leaves(
        network_path,
        ell_nn,
    )
    np.save(
        raw_params_path,
        np.asarray(raw_params),
    )

    print("\n=== Summary ===")
    for name in ("relerr_gamma", "relerr_wr", "relerr_zeta", "relerr_kappa"):
        values = 100.0 * np.asarray([row[name] for row in rows])
        print(
            f"{name}: mean={np.mean(values):.3f}% | "
            f"median={np.median(values):.3f}% | "
            f"q95={np.quantile(values, 0.95):.3f}%"
        )

    print(f"Total time : {time.time() - start_time:.2f} s")
    print(f"Results    : {result_dir}")


if __name__ == "__main__":
    main()
