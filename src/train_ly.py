import os
import json
import time

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


# ======================================================================================
# Paramètres d'entraînement
# ======================================================================================

N_SIGNALS = 200
N_EPOCHS_PER_SIGNAL = 10
N_CYCLES = 1

LR_TRAIN = 1e-4
TOL_TRAIN = 1e-5

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

PARAM_RANGES = {
    "gamma_final": (0.20, 0.80),
    "zeta": (0.20, 0.80),
    "kappa": (0.40, 1.20),
    "fr": (120.0, 240.0),
    "Qr": (80.0, 180.0),
}


# ======================================================================================
# Outils
# ======================================================================================

def replace_l(data, ell_nn):
    return eqx.tree_at(lambda d: d.l, data, ell_nn)


def set_sampled_params(data, sampled_params):
    for name, value in sampled_params.items():
        data = set_param(
            data,
            name,
            jnp.array(value),
            GEO_KEYS,
        )
    return data


def sample_physical_params(key):
    sampled = {}
    keys = jax.random.split(key, len(PARAM_RANGES))

    for subkey, (name, (vmin, vmax)) in zip(keys, PARAM_RANGES.items()):
        value = jax.random.uniform(
            subkey,
            shape=(),
            minval=vmin,
            maxval=vmax,
        )
        sampled[name] = float(value)

    return sampled


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
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


def make_optimizer(lr, n_iter):
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_iter,
        alpha=1e-2,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=1e-7,
        ),
    )

    return optimizer, scheduler


def physical_regularization(ell_nn):
    y_neg = jnp.linspace(-0.2, 0.0, 64)
    y_pos = jnp.linspace(0.0, 1.5, 128)

    pred_neg = jax.vmap(ell_nn)(y_neg)
    pred_pos = jax.vmap(ell_nn)(y_pos)

    loss_neg = jnp.mean(pred_neg**2)

    dy = y_pos[1] - y_pos[0]
    grad = (pred_pos[1:] - pred_pos[:-1]) / dy
    loss_mono = jnp.mean(jax.nn.relu(-grad)**2)

    return loss_neg + 0.1 * loss_mono


# ======================================================================================
# Loss
# ======================================================================================

def loss_l_dataset(
    ell_nn,
    target_dataset,
    Nx,
    c,
    solve_kwargs,
    reg_weight=1e-3,
):
    total_loss = 0.0

    for data_true, target_snaps in target_dataset:
        data = replace_l(data_true, ell_nn)

        pred = forward_snapshots(
            data,
            Nx,
            c,
            **solve_kwargs,
        )

        total_loss = total_loss + loss_fn_signal(pred, target_snaps)

    loss_signal = total_loss / len(target_dataset)
    loss_reg = physical_regularization(ell_nn)

    return loss_signal + reg_weight * loss_reg


# ======================================================================================
# Entraînement
# ======================================================================================

def run_l_training(
    ell_nn,
    loss_and_grad,
    target_dataset,
    lr,
    n_epochs_per_signal,
    n_cycles,
    tol,
):
    total_iter = n_epochs_per_signal * len(target_dataset) * n_cycles

    optimizer, scheduler = make_optimizer(lr, total_iter)
    opt_state = optimizer.init(eqx.filter(ell_nn, eqx.is_array))

    print("\n=== Entraînement séquentiel de l(y) ===")
    print(f"N signals      = {len(target_dataset)}")
    print(f"epochs/signal  = {n_epochs_per_signal}")
    print(f"cycles         = {n_cycles}")
    print(f"{'iter':>5} | {'signal':>6} | {'loss':>12} | {'lr':>10} | {'t/iter':>8}")
    print("-" * 70)

    history = {"iter": [], "signal": [], "loss": [], "lr": []}
    global_iter = 0

    for cycle in range(n_cycles):
        print(f"\n--- Cycle {cycle + 1}/{n_cycles} ---")

        for signal_id, (data_true, target_snaps) in enumerate(target_dataset):
            single_dataset = [(data_true, target_snaps)]

            for local_iter in range(n_epochs_per_signal):
                t_it = time.time()
                current_lr = float(scheduler(global_iter))

                loss_val, grads = loss_and_grad(
                    ell_nn,
                    single_dataset,
                )

                updates, opt_state = optimizer.update(
                    grads,
                    opt_state,
                    eqx.filter(ell_nn, eqx.is_array),
                )

                ell_nn = eqx.apply_updates(ell_nn, updates)

                elapsed = time.time() - t_it

                history["iter"].append(global_iter)
                history["signal"].append(signal_id)
                history["loss"].append(float(loss_val))
                history["lr"].append(current_lr)

                if local_iter == 0:
                    print(
                        f"{global_iter:>5} | "
                        f"{signal_id:>6} | "
                        f"{float(loss_val):>12.4e} | "
                        f"{current_lr:>10.3e} | "
                        f"{elapsed:>7.2f}s"
                    )

                global_iter += 1

                if float(loss_val) < tol:
                    print(f"\nConvergence atteinte à l'itération {global_iter}.")
                    return ell_nn, history

    return ell_nn, history


# ======================================================================================
# Plot principal
# ======================================================================================

def plot_training_summary(history, ell_nn, l_true, path):
    y_grid = jnp.linspace(-0.2, 1.5, 400)

    ell_pred = jax.vmap(ell_nn)(y_grid)
    ell_true = jax.vmap(l_true)(y_grid)

    rel_err = (
        jnp.linalg.norm(ell_pred - ell_true)
        / jnp.linalg.norm(ell_true)
    )

    max_err = jnp.max(jnp.abs(ell_pred - ell_true))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].semilogy(
        history["iter"],
        history["loss"],
    )
    ax[0].set_title("Training loss")
    ax[0].set_xlabel("Iteration")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True)

    ax[1].plot(
        np.array(y_grid),
        np.array(ell_true),
        label="ReedOpening vraie",
        linewidth=2,
    )

    ax[1].plot(
        np.array(y_grid),
        np.array(ell_pred),
        label=(
            f"MLP appris\n"
            f"rel={100 * float(rel_err):.2f}%\n"
            f"max={float(max_err):.3e}"
        ),
        linewidth=2,
    )

    ax[1].set_xlabel("y")
    ax[1].set_ylabel("l(y)")
    ax[1].set_title("Comparaison de l(y)")
    ax[1].grid(True)
    ax[1].legend()

    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

    print(f"Erreur relative sur l(y) : {100 * float(rel_err):.3f}%")


# ======================================================================================
# Main
# ======================================================================================

def main():
    start_time = time.time()

    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    train_params = solver_params["train"]

    T_max_train = train_params["T_max"]
    CFL_train = train_params["cfl"]
    Nx_train = train_params["Nx"]
    N_snapshot_time_train = train_params["N_snapshot"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    for name in PARAM_RANGES:
        params["trainable"][name] = True

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    args = parse_args()
    type_S = args.type_S

    result_dir = "../experiments/gradient/results"
    os.makedirs(result_dir, exist_ok=True)

    data_ref = build_physical_data(params, type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell

    bc = BC(type="full")
    forward_snapshots_jit = eqx.filter_jit(forward_snapshots)

    dt_train, nsteps_train, solve_kwargs_train = make_solver_data(
        T_max_train,
        CFL_train,
        Nx_train,
        N_snapshot_time_train,
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"Train: dt={dt_train:.6e}, nsteps={nsteps_train}, T_max={T_max_train:.4f}")

    print("\n=== Génération du dataset cible aléatoire ===")

    key = jax.random.PRNGKey(0)
    target_dataset = []

    for i in range(N_SIGNALS):
        key, subkey = jax.random.split(key)
        sampled_params = sample_physical_params(subkey)

        data_true = build_physical_data(params, type_S)
        data_true = set_sampled_params(data_true, sampled_params)

        target = forward_snapshots_jit(
            data_true,
            Nx_train,
            c,
            **solve_kwargs_train,
        )

        target_dataset.append((data_true, target))

        print(
            f"signal={i:03d} | "
            f"p max={float(jnp.max(jnp.abs(target))):.4e}"
        )

    print(f"Nombre de signaux : {len(target_dataset)}")

    key, subkey = jax.random.split(key)

    ell_nn = LFuncNN(
        [1, 8, 8, 1],
        activation=jax.nn.tanh,
        key=subkey,
    )

    data_plot = build_physical_data(params, type_S)

    def train_loss(ell_nn, target_dataset):
        return loss_l_dataset(
            ell_nn=ell_nn,
            target_dataset=target_dataset,
            Nx=Nx_train,
            c=c,
            solve_kwargs=solve_kwargs_train,
            reg_weight=1e-3,
        )

    loss_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(train_loss)
    )

    ell_nn, history = run_l_training(
        ell_nn=ell_nn,
        loss_and_grad=loss_and_grad,
        target_dataset=target_dataset,
        lr=LR_TRAIN,
        n_epochs_per_signal=N_EPOCHS_PER_SIGNAL,
        n_cycles=N_CYCLES,
        tol=TOL_TRAIN,
    )

    plot_training_summary(
        history,
        ell_nn,
        data_plot.l,
        f"{result_dir}/training_summary.png",
    )

    eqx.tree_serialise_leaves(
        f"{result_dir}/ell_nn_random_dataset.eqx",
        ell_nn,
    )

    print("\nAfter training l(y):")
    print("=" * 50)
    print(f"Temps total : {time.time() - start_time:.2f}s")
    print(f"Modèle sauvegardé dans {result_dir}/ell_nn_random_dataset.eqx")


if __name__ == "__main__":
    main()