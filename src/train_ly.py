import os
import copy
import csv
import json
import time
from pathlib import Path

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
from utils.build_solver import build_solver_geometry
from utils.solve import forward_snapshots
from utils.param_func import set_param
from inverse.total_loss import loss_fn_signal
from inverse.l_func_nn import LFuncNN


jax.config.update("jax_enable_x64", True)
print(jax.devices())


# ======================================================================================
# Paramètres d'entraînement de l(y) uniquement
# ======================================================================================

N_SIGNALS = 300


N_EPOCHS_PHASE1 = 10
LR_PHASE1 = 3e-4

N_EPOCHS_PHASE2 = 8
LR_PHASE2 = 3e-5
LR_FINAL_FACTOR = 0.8

BATCH_SIZE = 16
SHUFFLE_BATCHES = True
TOL_TRAIN = 1e-5

REG_WEIGHT = 1e-3
ANCHOR_WEIGHT = 1e-1

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
P_CLOSED = 5e3

PARAM_JSON_PATHS = {
    "gamma_final": (
        "left_bc_params",
        "mouth_pressure_params",
        "gamma_final",
    ),
    "zeta": ("left_bc_params", "zeta"),
    "kappa": ("left_bc_params", "kappa"),
    "fr": ("left_bc_params", "fr"),
    "Qr": ("left_bc_params", "Qr"),
    "alpha": ("right_bc_params", "alpha"),
    "beta": ("right_bc_params", "beta"),
    "Zt": ("right_bc_params", "Zt"),
}

PARAM_RANGES = {
    "gamma_final": (0.20, 0.80),
    "zeta": (0.20, 0.80),
    "kappa": (0.40, 1.20),
    "fr": (120.0, 240.0),
    "Qr": (80.0, 180.0),
}


# All per-signal quantities passed through the JIT as one fixed-shape array.
# Using one common PhysicalData template avoids recompilation caused by
# different Python/static objects for every signal.
DYNAMIC_CASE_PARAMS = (
    "gamma_final",
    "zeta",
    "kappa",
    "fr",
    "Qr",
    "alpha",
    "beta",
    "Zt",
)


def apply_case_vector(data_template, case_vector):
    """Insert one signal's physical parameters into a common data template."""
    data = data_template

    for idx, name in enumerate(DYNAMIC_CASE_PARAMS):
        data = set_param(
            data,
            name,
            case_vector[idx],
            GEO_KEYS,
        )

    return data


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


def set_nested(mapping, path, value):
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = float(value)


def params_with_openwind_radiation(base_params, ow_params, type_S):
    """Apply the same DG/OpenWind radiation matching as scan_loss_1_D."""
    params = copy.deepcopy(base_params)
    if ow_params.get("alpha") is not None:
        set_nested(params, PARAM_JSON_PATHS["alpha"], ow_params["alpha"])
    if ow_params.get("beta") is not None:
        set_nested(params, PARAM_JSON_PATHS["beta"], ow_params["beta"])

    data = build_physical_data(params, type_S)
    length = data.section.L_tube + data.section.L_bell
    zt_geometry = float(data.section(0.0) / data.section(length))
    set_nested(params, PARAM_JSON_PATHS["Zt"], zt_geometry)
    return params, zt_geometry


def load_openwind_dataset(path, expected):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset OpenWind absent: {path}. Lance d'abord "
            "src/generate_openwind_Qr_wr_gamma_zeta_dataset.py."
        )
    with np.load(path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        for name, expected_value in expected.items():
            if metadata.get(name) != expected_value:
                raise ValueError(
                    f"Dataset incompatible pour {name}: "
                    f"attendu={expected_value!r}, trouve={metadata.get(name)!r}."
                )
        true_values = np.asarray(dataset["true_values"]).reshape(-1, 4)
        pressure = np.asarray(dataset["pressure_long"]).reshape(
            true_values.shape[0], -1
        )
        radiation = {
            "alpha": float(dataset["radiation_alpha"]),
            "beta": float(dataset["radiation_beta"]),
        }
    return metadata, true_values, pressure, radiation


def set_sampled_params_json(params, sampled_params):
    params_true = copy.deepcopy(params)

    for name, value in sampled_params.items():
        set_nested(
            params_true,
            PARAM_JSON_PATHS[name],
            value,
        )

    return params_true


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
        alpha=LR_FINAL_FACTOR,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(0.1),
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=1e-7,
        ),
    )

    return optimizer, scheduler

def make_const_optimizer(lr, n_iter):
    scheduler = optax.constant_schedule(lr)

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

def loss_l_single(
    ell_nn,
    case_vector,
    target_snaps,
    data_template,
    geometry,
    c,
    solve_kwargs,
    reg_weight=1e-3,
):
    """
    Loss for one signal.

    case_vector is a fixed-shape JAX array. data_template is the same static
    object for every call, so XLA can reuse one compiled executable.
    """
    data = apply_case_vector(
        data_template,
        case_vector,
    )
    data = replace_l(data, ell_nn)

    pred = forward_snapshots(
        data,
        geometry,
        c,
        **solve_kwargs,
    )

    loss_signal = loss_fn_signal(pred, target_snaps)

    # La régularisation de la loi commune est ajoutée une seule fois
    # dans la loss de batch, et non répétée pour chaque signal.
    return loss_signal


def loss_l_dataset(
    ell_nn,
    target_dataset,
    geometry,
    c,
    solve_kwargs,
    reg_weight=1e-3,
):
    total_loss = 0.0

    for data_true, target_snaps in target_dataset:
        data = replace_l(data_true, ell_nn)

        pred = forward_snapshots(
            data,
            geometry,
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

def make_minibatches(target_dataset, batch_size, permutation):
    """
    Construit des mini-batches sous forme de deux tableaux JAX empilés :

        case_batch.shape   = (B, n_case_params)
        target_batch.shape = (B, n_snapshots)

    Le dernier batch est complété par répétition du dernier élément afin
    de conserver une taille statique et d'éviter une recompilation JAX.
    La taille réelle du batch est également renvoyée.
    """
    batches = []

    for start_idx in range(0, len(permutation), batch_size):
        indices = permutation[start_idx:start_idx + batch_size]
        real_size = len(indices)

        if real_size == 0:
            continue

        if real_size < batch_size:
            pad_value = indices[-1]
            padding = np.full(
                batch_size - real_size,
                pad_value,
                dtype=indices.dtype,
            )
            indices = np.concatenate([indices, padding])

        case_batch = jnp.stack(
            [target_dataset[int(i)][0] for i in indices],
            axis=0,
        )
        target_batch = jnp.stack(
            [target_dataset[int(i)][1] for i in indices],
            axis=0,
        )

        batches.append((case_batch, target_batch, real_size))

    return batches


def tree_add(tree_a, tree_b):
    """Add two gradient PyTrees while preserving None leaves."""
    return jax.tree_util.tree_map(
        lambda a, b: (
            None
            if a is None
            else a + b
        ),
        tree_a,
        tree_b,
        is_leaf=lambda x: x is None,
    )


def tree_scale(tree, factor):
    """Multiply every non-None gradient leaf by a scalar."""
    return jax.tree_util.tree_map(
        lambda x: (
            None
            if x is None
            else factor * x
        ),
        tree,
        is_leaf=lambda x: x is None,
    )



def run_l_training_minibatch(
    ell_nn,
    batch_loss_and_grad,
    target_dataset,
    l_true,
    lr,
    n_epochs,
    batch_size,
    tol,
    seed=0,
    shuffle=True,
    phase_name="phase",
    start_iter=0,
    start_epoch=0,
):
    """
    Entraînement mini-batch vectorisé.

    Les signaux d'un même batch sont traités ensemble avec jax.vmap.
    Une seule passe forward/backward est lancée par batch, puis une seule
    mise à jour AdamW est appliquée.
    """
    n_signals = len(target_dataset)
    n_batches = int(np.ceil(n_signals / batch_size))
    total_iter = n_epochs * n_batches

    optimizer, scheduler = make_const_optimizer(lr, total_iter)
    opt_state = optimizer.init(eqx.filter(ell_nn, eqx.is_array))

    rng = np.random.default_rng(seed)

    print(
        f"\n=== {phase_name}: entraînement mini-batch vectorisé de l(y) ==="
    )
    print(f"N signals      = {n_signals}")
    print(f"batch size     = {batch_size}")
    print(f"batches/epoch  = {n_batches}")
    print(f"epochs         = {n_epochs}")
    print(f"updates total  = {total_iter}")
    print(
        f"{'iter':>5} | {'epoch':>5} | {'batch':>5} | "
        f"{'size':>4} | {'loss':>12} | {'lr':>10} | {'t/iter':>8}"
    )
    print("-" * 88)

    history = {
        "iter": [],
        "epoch": [],
        "batch": [],
        "batch_size": [],
        "loss": [],
        "lr": [],
        "epoch_index": [],
        "epoch_mean_loss": [],
        "epoch_median_loss": [],
        "epoch_relerr_l": [],
        "epoch_maxerr_l": [],
    }

    global_iter = start_iter

    for epoch in range(n_epochs):
        permutation = (
            rng.permutation(n_signals)
            if shuffle
            else np.arange(n_signals)
        )

        batches = make_minibatches(
            target_dataset,
            batch_size,
            permutation,
        )

        global_epoch = start_epoch + epoch + 1
        print(
            f"\n--- {phase_name} | epoch local {epoch + 1}/{n_epochs} "
            f"| epoch global {global_epoch} ---"
        )

        epoch_losses = []

        for batch_id, (case_batch, target_batch, real_size) in enumerate(batches):
            t_it = time.time()
            current_lr = float(scheduler(global_iter))

            loss_val, grads = batch_loss_and_grad(
                ell_nn,
                case_batch,
                target_batch,
                jnp.asarray(real_size, dtype=jnp.int32),
            )

            updates, opt_state = optimizer.update(
                grads,
                opt_state,
                eqx.filter(ell_nn, eqx.is_array),
            )
            ell_nn = eqx.apply_updates(ell_nn, updates)

            loss_val = jax.block_until_ready(loss_val)

            elapsed = time.time() - t_it
            loss_float = float(loss_val)
            epoch_losses.append(loss_float)

            history["iter"].append(global_iter)
            history["epoch"].append(global_epoch)
            history["batch"].append(batch_id)
            history["batch_size"].append(real_size)
            history["loss"].append(loss_float)
            history["lr"].append(current_lr)

            print(
                f"{global_iter:>5} | "
                f"{global_epoch:>5} | "
                f"{batch_id:>5} | "
                f"{real_size:>4} | "
                f"{loss_float:>12.4e} | "
                f"{current_lr:>10.3e} | "
                f"{elapsed:>7.2f}s"
            )

            global_iter += 1

        mean_epoch_loss = float(np.mean(epoch_losses))
        median_epoch_loss = float(np.median(epoch_losses))

        y_grid_eval = jnp.linspace(-0.2, 1.5, 400)
        ell_pred_eval = jax.vmap(ell_nn)(y_grid_eval)
        ell_true_eval = jax.vmap(l_true)(y_grid_eval)

        relerr_l = (
            jnp.linalg.norm(ell_pred_eval - ell_true_eval)
            / (jnp.linalg.norm(ell_true_eval) + 1e-12)
        )
        maxerr_l = jnp.max(jnp.abs(ell_pred_eval - ell_true_eval))

        relerr_l = float(jax.block_until_ready(relerr_l))
        maxerr_l = float(jax.block_until_ready(maxerr_l))

        history["epoch_index"].append(global_epoch)
        history["epoch_mean_loss"].append(mean_epoch_loss)
        history["epoch_median_loss"].append(median_epoch_loss)
        history["epoch_relerr_l"].append(relerr_l)
        history["epoch_maxerr_l"].append(maxerr_l)

        print(
            f"Epoch globale {global_epoch}: "
            f"mean loss={mean_epoch_loss:.4e}, "
            f"median loss={median_epoch_loss:.4e}, "
            f"relerr l(y)={100.0 * relerr_l:.3f}%, "
            f"maxerr l(y)={maxerr_l:.4e}"
        )

        if mean_epoch_loss < tol:
            print(
                f"\nConvergence atteinte à l'époque globale {global_epoch} "
                f"(loss moyenne={mean_epoch_loss:.4e})."
            )
            return ell_nn, history

    return ell_nn, history


def merge_histories(*histories):
    merged = {key: [] for key in histories[0]}
    for history in histories:
        for key in merged:
            merged[key].extend(history[key])
    return merged


# ======================================================================================
# Plot principal
# ======================================================================================


def plot_training_summary(history, ell_nn, l_true, path):
    """
    Figure finale à deux sous-figures :
      1. évolution de la loss par batch et moyenne par époque ;
      2. comparaison entre la loi apprise et la loi théorique.
    """
    y_grid = jnp.linspace(-0.2, 1.5, 400)

    ell_pred = jax.vmap(ell_nn)(y_grid)
    ell_true = jax.vmap(l_true)(y_grid)

    rel_err = (
        jnp.linalg.norm(ell_pred - ell_true)
        / (jnp.linalg.norm(ell_true) + 1e-12)
    )
    max_err = jnp.max(jnp.abs(ell_pred - ell_true))
    anchor_value = ell_nn(jnp.asarray(1.0))

    rel_err = float(jax.block_until_ready(rel_err))
    max_err = float(jax.block_until_ready(max_err))
    anchor_value = float(jax.block_until_ready(anchor_value))

    batch_iterations = np.asarray(history["iter"])
    batch_losses = np.asarray(history["loss"])
    epoch_indices = np.asarray(history["epoch_index"])
    epoch_mean_losses = np.asarray(history["epoch_mean_loss"])
    epoch_median_losses = np.asarray(history["epoch_median_loss"])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))

    ax[0].semilogy(
        batch_iterations,
        batch_losses,
        alpha=0.30,
        linewidth=0.9,
        label="Loss des mini-batches",
    )

    n_batches = int(np.ceil(N_SIGNALS / BATCH_SIZE))
    epoch_end_iterations = epoch_indices * n_batches - 1

    ax[0].semilogy(
        epoch_end_iterations,
        epoch_mean_losses,
        marker="o",
        linewidth=2.0,
        label="Loss moyenne par époque",
    )
    ax[0].semilogy(
        epoch_end_iterations,
        epoch_median_losses,
        marker="s",
        linewidth=1.5,
        linestyle="--",
        label="Loss médiane par époque",
    )

    phase_transition_iter = N_EPOCHS_PHASE1 * n_batches
    ax[0].axvline(
        phase_transition_iter,
        linestyle=":",
        linewidth=1.6,
        label="Début de la phase fine",
    )

    ax[0].set_title("Évolution de la loss MSTS")
    ax[0].set_xlabel("Mise à jour")
    ax[0].set_ylabel("Loss")
    ax[0].grid(True, alpha=0.3)
    ax[0].legend(fontsize=8)

    ax[1].plot(
        np.asarray(y_grid),
        np.asarray(ell_true),
        label=r"$\ell_{\mathrm{th}}(y)=\max(y,0)$",
        linewidth=2.2,
    )
    ax[1].plot(
        np.asarray(y_grid),
        np.asarray(ell_pred),
        linestyle="--",
        label=(
            r"$\ell_{\mathrm{NN}}(y)$"
            f"\nerr. rel.={100.0 * rel_err:.3f}%"
            f"\nerr. max.={max_err:.3e}"
        ),
        linewidth=2.2,
    )
    ax[1].scatter(
        [1.0],
        [anchor_value],
        marker="o",
        s=45,
        label=rf"$\ell_{{\mathrm{{NN}}}}(1)={anchor_value:.4f}$",
        zorder=4,
    )
    ax[1].scatter(
        [1.0],
        [1.0],
        marker="x",
        s=60,
        linewidths=1.8,
        label=r"Ancrage théorique $(1,1)$",
        zorder=4,
    )

    ax[1].set_xlabel(r"$y$")
    ax[1].set_ylabel(r"$\ell(y)$")
    ax[1].set_title("Loi apprise et loi théorique")
    ax[1].grid(True, alpha=0.3)
    ax[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Erreur relative sur l(y) : {100.0 * rel_err:.3f}%")
    print(f"Erreur maximale sur l(y) : {max_err:.4e}")
    print(f"Valeur de l(1) apprise   : {anchor_value:.6f}")


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

    # Les paramètres échantillonnés doivent rester des feuilles JAX afin que
    # set_param puisse construire chaque jeu de données cible.
    # Cela ne signifie pas qu'ils sont optimisés : seul ell_nn est transmis
    # à l'optimiseur plus bas dans le script.
    # These entries must be JAX leaves because they are modified later with
    # set_param. Only ell_nn is passed to the optimizer, so setting these flags
    # to True does not make them trainable in this experiment.
    mutable_physical_params = set(PARAM_RANGES) | {"alpha", "beta", "Zt"}

    for name in params["trainable"]:
        params["trainable"][name] = name in mutable_physical_params

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    args = parse_args()
    type_S = args.type_S

    result_dir = "../experiments/gradient/results/train_l_only_msts_openwind_matched"
    os.makedirs(result_dir, exist_ok=True)

    data_ref = build_physical_data(params, type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell

    geometry = build_solver_geometry(
        data_ref,
        Nx_train,
        c,
    )

    bc = BC(type="full")

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
    print(f"LR phase 1={LR_PHASE1:.3e}")
    print(f"LR phase fine={LR_PHASE2:.3e}")
    print(f"Poids régularisation={REG_WEIGHT:.3e}")
    print(f"Poids ancrage l(1)=1={ANCHOR_WEIGHT:.3e}")
    print(
        "OpenWind l_ele="
        f"{args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4}"
    )

    print("\n=== Chargement des cibles OpenWind pregenerees ===")

    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_absolute():
        dataset_path = Path(__file__).resolve().parents[1] / dataset_path
    expected_dataset = {
        "format_version": 1,
        "type_S": type_S,
        "T_long": float(T_max_train),
        "Nx": int(Nx_train),
        "N_snapshot": int(N_snapshot_time_train),
        "cfl": float(CFL_train),
        "c": float(c),
        "ow_order": int(args.ow_order),
        "ow_theta": float(args.ow_theta),
        "ow_l_ele": float(args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4),
    }
    dataset_metadata, true_values, pressure_targets, radiation = (
        load_openwind_dataset(dataset_path, expected_dataset)
    )
    if true_values.shape[0] != N_SIGNALS:
        raise ValueError(
            f"Le cache contient {true_values.shape[0]} signaux; "
            f"train_ly en attend {N_SIGNALS}."
        )

    key = jax.random.PRNGKey(0)
    target_dataset = []
    dataset_rows = []
    params_matched, Zt_signal = params_with_openwind_radiation(
        params, radiation, type_S
    )
    data_template = build_physical_data(params_matched, type_S)
    kappa_value = float(params["left_bc_params"]["kappa"])

    for i, (values, target_np) in enumerate(zip(true_values, pressure_targets)):
        gamma, wr, zeta, Qr = map(float, values)
        fr = wr / (2.0 * np.pi)
        target = jnp.asarray(target_np, dtype=jnp.float64)

        case_vector = jnp.asarray(
            [
                gamma, zeta, kappa_value, fr, Qr,
                radiation["alpha"], radiation["beta"],
                Zt_signal,
            ],
            dtype=jnp.float64,
        )

        target_dataset.append((case_vector, target))

        dataset_rows.append({
            "signal_idx": i,
            "gamma_final": gamma,
            "zeta": zeta,
            "kappa": kappa_value,
            "fr": fr,
            "Qr": Qr,
            "ow_alpha": radiation["alpha"],
            "ow_beta": radiation["beta"],
            "ow_dt": dataset_metadata.get("ow_dt"),
            "dg_Zt": float(Zt_signal),
            "target_max_abs_pressure": float(
                jnp.max(jnp.abs(target))
            ),
        })

        print(
            f"signal={i:03d} | "
            f"gamma={gamma:.4f} | fr={fr:.2f} Hz | "
            f"zeta={zeta:.4f} | kappa={kappa_value:.4f} | "
            f"Qr={Qr:.2f} | "
            f"max|p_OW|={float(jnp.max(jnp.abs(target))):.4e}"
        )

    print(f"Dataset OpenWind : {dataset_path}")

    dataset_csv_path = os.path.join(
        result_dir,
        "openwind_dataset_parameters.csv",
    )
    with open(dataset_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(dataset_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"Nombre de signaux : {len(target_dataset)}")

    # Check the OW/DG mismatch with the default DG law before training.
    baseline_losses = []
    for case_vector, target_snaps in target_dataset[:min(10, len(target_dataset))]:
        data_true = apply_case_vector(
            data_template,
            case_vector,
        )
        pred_true_l = forward_snapshots(
            data_true,
            geometry,
            c,
            **solve_kwargs_train,
        )
        baseline_losses.append(
            float(loss_fn_signal(pred_true_l, target_snaps))
        )

    print(
        "Baseline MSTS with default DG l(y): "
        f"mean={np.mean(baseline_losses):.4e}, "
        f"median={np.median(baseline_losses):.4e}"
    )

    key, subkey = jax.random.split(key)

    ell_nn = LFuncNN(
        [1, 8, 8, 1],
        activation=jax.nn.tanh,
        key=subkey,
    )

    data_plot = build_physical_data(params, type_S)

    def train_loss_single(ell_nn, case_vector, target_snaps):
        return loss_l_single(
            ell_nn=ell_nn,
            case_vector=case_vector,
            target_snaps=target_snaps,
            data_template=data_template,
            geometry=geometry,
            c=c,
            solve_kwargs=solve_kwargs_train,
            reg_weight=REG_WEIGHT,
        )

    # Vectorisation des signaux d'un même batch.
    # Le réseau ell_nn est partagé entre tous les signaux du batch.
    vmapped_train_loss = jax.vmap(
        train_loss_single,
        in_axes=(None, 0, 0),
    )

    def train_loss_batch(
        ell_nn,
        case_batch,
        target_batch,
        real_size,
    ):
        losses = vmapped_train_loss(
            ell_nn,
            case_batch,
            target_batch,
        )

        # Le dernier batch est éventuellement complété. On masque les
        # éléments artificiels pour ne moyenner que les vrais signaux.
        mask = (
            jnp.arange(losses.shape[0])
            < real_size
        ).astype(losses.dtype)

        loss_signal = jnp.sum(losses * mask) / jnp.maximum(
            jnp.sum(mask),
            1.0,
        )

        loss_shape = physical_regularization(ell_nn)
        anchor_error = ell_nn(jnp.asarray(1.0)) - 1.0
        loss_anchor = anchor_error**2

        return (
            loss_signal
            + REG_WEIGHT * loss_shape
            + ANCHOR_WEIGHT * loss_anchor
        )

    batch_loss_and_grad = eqx.filter_jit(
        eqx.filter_value_and_grad(train_loss_batch)
    )

    ell_nn, history_phase1 = run_l_training_minibatch(
        ell_nn=ell_nn,
        batch_loss_and_grad=batch_loss_and_grad,
        target_dataset=target_dataset,
        l_true=data_plot.l,
        lr=LR_PHASE1,
        n_epochs=N_EPOCHS_PHASE1,
        batch_size=BATCH_SIZE,
        tol=TOL_TRAIN,
        seed=0,
        shuffle=SHUFFLE_BATCHES,
        phase_name="Phase 1",
        start_iter=0,
        start_epoch=0,
    )

    phase1_model_path = os.path.join(
        result_dir,
        "ell_nn_after_phase1.eqx",
    )
    eqx.tree_serialise_leaves(phase1_model_path, ell_nn)

    ell_nn, history_phase2 = run_l_training_minibatch(
        ell_nn=ell_nn,
        batch_loss_and_grad=batch_loss_and_grad,
        target_dataset=target_dataset,
        l_true=data_plot.l,
        lr=LR_PHASE2,
        n_epochs=N_EPOCHS_PHASE2,
        batch_size=BATCH_SIZE,
        tol=TOL_TRAIN,
        seed=10,
        shuffle=SHUFFLE_BATCHES,
        phase_name="Phase 2 (finition)",
        start_iter=len(history_phase1["iter"]),
        start_epoch=len(history_phase1["epoch_index"]),
    )

    history = merge_histories(history_phase1, history_phase2)

    epoch_history_path = os.path.join(
        result_dir,
        "epoch_history.csv",
    )

    with open(epoch_history_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "phase",
                "learning_rate",
                "mean_loss",
                "median_loss",
                "relerr_l",
                "relerr_l_percent",
                "maxerr_l",
            ],
        )
        writer.writeheader()

        for epoch_idx, mean_loss, median_loss, relerr_l, maxerr_l in zip(
            history["epoch_index"],
            history["epoch_mean_loss"],
            history["epoch_median_loss"],
            history["epoch_relerr_l"],
            history["epoch_maxerr_l"],
        ):
            is_phase1 = epoch_idx <= N_EPOCHS_PHASE1
            writer.writerow({
                "epoch": epoch_idx,
                "phase": "phase1" if is_phase1 else "phase2_fine",
                "learning_rate": LR_PHASE1 if is_phase1 else LR_PHASE2,
                "mean_loss": mean_loss,
                "median_loss": median_loss,
                "relerr_l": relerr_l,
                "relerr_l_percent": 100.0 * relerr_l,
                "maxerr_l": maxerr_l,
            })

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

    print("\nAprès entraînement de l(y) uniquement :")
    print("=" * 50)
    print(f"Temps total : {time.time() - start_time:.2f}s")
    print(f"Checkpoint phase 1 sauvegardé dans {phase1_model_path}")
    print(f"Modèle final sauvegardé dans {result_dir}/ell_nn_random_dataset.eqx")
    print(f"Paramètres OpenWind sauvegardés dans {dataset_csv_path}")
    print(f"Historique par époque sauvegardé dans {epoch_history_path}")


if __name__ == "__main__":
    main()