import argparse
import copy
import csv
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse.l_func_nn import LFuncNN
from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)

MIN_SCALE = 1e-8
DEFAULT_INIT_FACTOR = 0.8
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
PARAM_NAMES = ("gamma", "kappa", "zeta")

def inverse_softplus(value):
    value = jnp.maximum(jnp.asarray(value, dtype=jnp.float64), 1e-12)
    return value + jnp.log(-jnp.expm1(-value))


def positive_parameters(raw_theta, scales):
    return scales * (jax.nn.softplus(raw_theta) + 1e-12)



class NormalizedLFuncNN(eqx.Module):
    """Réseau pour ell(y) normalisé exactement par ell(1)=1."""

    network: LFuncNN

    def __init__(self, layer_sizes, activation, key):
        self.network = LFuncNN(
            layer_sizes,
            activation=activation,
            key=key,
        )

    def __call__(self, y):
        y = jnp.asarray(y, dtype=jnp.float64)
        one = jnp.asarray(1.0, dtype=jnp.float64)
        denominator = self.network(one)
        return self.network(y) / (denominator + 1e-12)

PARAM_JSON_PATHS = {
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "Qr": ("left_bc_params", "Qr"),
    "kappa": ("left_bc_params", "kappa"),
    "zeta": ("left_bc_params", "zeta"),
    "alpha": ("right_bc_params", "alpha"),
    "beta": ("right_bc_params", "beta"),
    "Zt": ("right_bc_params", "Zt"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibration sequentielle de gamma, kappa et zeta avec une loi "
            "partagee ell(y). Chaque cycle parcourt les repeats; chaque repeat "
            "effectue plusieurs mises a jour conjointes et modifie immediatement "
            "la loi globale, comme dans l ancien code performant."
        )
    )

    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--train_params",
        nargs="+",
        choices=PARAM_NAMES,
        default=list(PARAM_NAMES),
        help=(
            "Parametres physiques a calibrer parmi gamma, kappa et zeta. "
            "Qr reste connu et fixe a la valeur OpenWind de chaque signal."
        ),
    )

    # Entrainement unique : fenetre longue.
    parser.add_argument(
        "--stage1_iter",
        "--n_cycles",
        dest="stage1_iter",
        type=int,
        default=2,
        help="Nombre de cycles complets sur les repeats.",
    )
    parser.add_argument("--stage1_lr_params", type=float, default=3e-3)
    parser.add_argument("--stage1_lr_lfunc", type=float, default=1e-4)
    parser.add_argument(
        "--local_steps",
        "--param_steps",
        dest="local_steps",
        type=int,
        default=100,
        help=(
            "Nombre de mises a jour locales des parametres physiques "
            "par repeat et par cycle externe, avec ell(y) gelee."
        ),
    )
    parser.add_argument("--law_steps", type=int, default=5)
    parser.add_argument("--params_lr_final_factor", type=float, default=0.1)
    parser.add_argument("--lfunc_lr_final_factor", type=float, default=0.1)

    # Apprentissage de ell(y).
    parser.add_argument("--lfunc_hidden_width", type=int, default=8)
    parser.add_argument("--lfunc_hidden_layers", type=int, default=2)
    parser.add_argument("--reg_weight", type=float, default=1e-3)
    parser.add_argument("--grad_clip_params", type=float, default=1.0)
    parser.add_argument("--grad_clip_lfunc", type=float, default=0.1)

    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help=(
            "Nombre de signaux traites simultanement. "
            "Les valeurs 5 ou 10 sont recommandees sur CPU ou GPU."
        ),
    )
    parser.add_argument(
        "--shuffle_seed",
        type=int,
        default=1234,
        help="Graine utilisee pour melanger les mini-batches a chaque epoque.",
    )
    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument(
        "--fast_test",
        action="store_true",
        help="Active un mode de validation rapide avec moins de signaux et d'iterations.",
    )
    parser.add_argument(
        "--log_every",
        type=int,
        default=5,
        help=(
            "Enregistre l'historique de mise a jour tous les N pas "
            "pour reduire la surcharge Python pendant l'apprentissage."
        ),
    )
    parser.add_argument(
        "--local_print_every",
        type=int,
        default=10,
        help=(
            "Frequence d affichage des iterations du bloc parametres. "
            "Une valeur de 10 affiche les iterations 1, 10, 20, ..., 100."
        ),
    )

    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)

    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="64:16,128:32,256:64",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument("--no_stft_padding", action="store_true")

    parser.add_argument("--random_factor_min", type=float, default=0.75)
    parser.add_argument("--random_factor_max", type=float, default=1.25)

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "experiments/gradient/results/"
            "gamma_kappa_zeta_lfunc_alternating_blocks_optimized"
        ),
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=(
            "experiments/gradient/datasets/"
            "openwind_Qr_kappa_gamma_zeta_300.npz"
        ),
        help=(
            "Cache OpenWind precalcule. Le fichier peut encore contenir Qr : "
            "Qr est alors utilise comme parametre connu propre a chaque signal, "
            "mais il n'est jamais optimise."
        ),
    )
    parser.add_argument("--save_detailed_plots", action="store_true")

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


def load_openwind_dataset(path, expected):
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset OpenWind absent : {path}. "
            "Genere d'abord le cache OpenWind."
        )

    with np.load(path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))

        layout_keys = {"n_signals", "n_repeats"}
        for key, expected_value in expected.items():
            if key in layout_keys:
                continue
            actual = metadata.get(key)
            if actual != expected_value:
                raise ValueError(
                    f"Dataset incompatible pour {key}: "
                    f"attendu={expected_value!r}, trouve={actual!r}."
                )

        requested_n_repeats = int(expected["n_repeats"])
        requested_n_signals = int(expected["n_signals"])
        dataset_n_repeats = int(metadata["n_repeats"])
        dataset_n_signals = int(metadata["n_signals"])

        if requested_n_repeats > dataset_n_repeats:
            raise ValueError(
                "Nombre de repeats incompatible : "
                f"attendu={requested_n_repeats}, trouve={dataset_n_repeats}."
            )
        if requested_n_signals > dataset_n_signals:
            raise ValueError(
                "Nombre de signaux incompatible : "
                f"attendu={requested_n_signals}, trouve={dataset_n_signals}."
            )

        selected_n_repeats = min(requested_n_repeats, dataset_n_repeats)
        selected_n_signals = min(requested_n_signals, dataset_n_signals)

        arrays = {}
        for name in (
            "true_values",
            "pressure_short",
            "pressure_long",
            "times_short",
            "times_long",
        ):
            array = np.asarray(dataset[name])
            if name in {"true_values", "pressure_short", "pressure_long"}:
                array = array[:selected_n_repeats, :selected_n_signals]
            arrays[name] = array

        radiation = {
            "alpha": float(dataset["radiation_alpha"]),
            "beta": float(dataset["radiation_beta"]),
        }

    metadata = dict(metadata)
    metadata["n_repeats"] = selected_n_repeats
    metadata["n_signals"] = selected_n_signals
    metadata["simulation_count"] = selected_n_repeats * selected_n_signals

    expected_prefix = (selected_n_repeats, selected_n_signals)
    for name, array in arrays.items():
        if array.ndim >= 2 and array.shape[:2] != expected_prefix:
            raise ValueError(
                f"Forme invalide pour {name}: {array.shape}; "
                f"prefixe attendu={expected_prefix}."
            )

    if arrays["true_values"].shape[-1] < 4:
        raise ValueError(
            "Le cache doit contenir au minimum "
            "[gamma, kappa, zeta, Qr] dans true_values."
        )

    return metadata, arrays, radiation


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


def parse_stft_resolutions(value):
    resolutions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "x" if "x" in item else None
        if separator is None:
            raise ValueError(f"Resolution STFT invalide : {item}")
        n_fft, hop = map(int, item.split(separator, maxsplit=1))
        if n_fft <= 0 or hop <= 0:
            raise ValueError(f"Resolution STFT invalide : {item}")
        resolutions.append((n_fft, hop))

    if not resolutions:
        raise ValueError("La liste --stft_resolutions est vide.")

    return tuple(resolutions)


def set_trainable_parameters(params):
    params = copy.deepcopy(params)

    # Qr n'est pas optimise, mais il doit rester une feuille dynamique
    # du PyTree Equinox, car sa valeur connue change d'un signal à l'autre
    # dans le vmap. Le masque d'optimisation ne contient toujours que
    # gamma, kappa et zeta.
    dynamic_names = (
        "gamma_final",
        "kappa",
        "zeta",
        "Qr",
    )

    for name in params["trainable"]:
        params["trainable"][name] = name in dynamic_names

    return params


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    x_left, x_right = cell_edges_from_nodes(x_nodes)

    dt = CFL * (x_right[0] - x_left[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps, dtype=jnp.float64) * dt
    snapshot_steps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot)
    ).astype(jnp.int32)

    return {
        "dt": dt,
        "nsteps": nsteps,
        "bc": bc,
        "phi0": phi0,
        "y0": y0,
        "z0": z0,
        "t_solver": t_solver,
        "n_snaps": snapshot_steps,
    }


def replace_l(data, ell_nn):
    return eqx.tree_at(lambda d: d.l, data, ell_nn)


def set_parameter_vector_data(data, values, qr_value):
    gamma, kappa, zeta = values

    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)

    # Qr n'est pas calibre. Il est impose a la valeur connue du signal.
    data = set_param(data, "Qr", qr_value, GEO_KEYS)

    return data


def physical_regularization(ell_nn, y_neg, y_pos):
    """
    Régularisation de forme de la loi partagée ell(y).

    La contrainte ell(1)=1 n'apparaît pas ici : elle est imposée exactement
    par la classe NormalizedLFuncNN.
    """

    pred_neg = jax.vmap(ell_nn)(y_neg)
    pred_pos = jax.vmap(ell_nn)(y_pos)

    loss_closed = jnp.mean(pred_neg**2)
    loss_positive = jnp.mean(jax.nn.relu(-pred_pos) ** 2)

    dy = y_pos[1] - y_pos[0]
    finite_diff = (pred_pos[1:] - pred_pos[:-1]) / dy
    loss_monotone = jnp.mean(jax.nn.relu(-finite_diff) ** 2)

    return (
        loss_closed
        + 0.1 * loss_positive
        + 0.1 * loss_monotone
    )


def make_minibatch_loss(
    data_init,
    geometry,
    c,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    reg_weight,
):
    y_neg = jnp.linspace(-0.2, 0.0, 64, dtype=jnp.float64)
    y_pos = jnp.linspace(0.0, 1.5, 128, dtype=jnp.float64)
    """
    Construit une loss JIT-compatible pour un mini-batch de taille fixe.

    Les donnees du batch sont passees comme arguments pour eviter que JAX
    capture les 300 signaux dans un unique graphe de calcul.
    """
    def one_signal_loss(theta_one, scale_one, qr_one, target_one, ell_nn):
        physical_params = positive_parameters(theta_one, scale_one)

        data = set_parameter_vector_data(
            data_init,
            physical_params,
            qr_one,
        )
        data = replace_l(data, ell_nn)

        pred = forward_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )
        pred = pred[: target_one.shape[0]]

        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(
        one_signal_loss,
        in_axes=(0, 0, 0, 0, None),
    )

    def observation_terms(
        theta_batch,
        ell_nn,
        scales_batch,
        qr_batch,
        targets_batch,
        batch_weights,
    ):
        per_signal = vmapped_loss(
            theta_batch,
            scales_batch,
            qr_batch,
            targets_batch,
            ell_nn,
        )
        weight_sum = jnp.maximum(jnp.sum(batch_weights), 1.0)
        observation_loss = (
            jnp.sum(batch_weights * per_signal) / weight_sum
        )
        return observation_loss, per_signal

    def parameter_loss(
        theta_batch,
        ell_nn,
        scales_batch,
        qr_batch,
        targets_batch,
        batch_weights,
    ):
        """
        Loss du bloc paramètres.

        ell(y) est gelée : la régularisation de forme ne dépend pas de theta
        et ne doit donc pas être recalculée à chaque itération.
        """
        observation_loss, per_signal = observation_terms(
            theta_batch,
            ell_nn,
            scales_batch,
            qr_batch,
            targets_batch,
            batch_weights,
        )
        zero_regularization = jnp.asarray(0.0, dtype=observation_loss.dtype)
        return observation_loss, (
            observation_loss,
            zero_regularization,
            per_signal,
        )

    def law_loss(
        ell_nn,
        theta_batch,
        scales_batch,
        qr_batch,
        targets_batch,
        batch_weights,
        reg_weight_override=None,
    ):
        """Loss du bloc loi, avec paramètres physiques gelés."""
        observation_loss, per_signal = observation_terms(
            theta_batch,
            ell_nn,
            scales_batch,
            qr_batch,
            targets_batch,
            batch_weights,
        )
        shape_regularization = physical_regularization(ell_nn, y_neg, y_pos)
        effective_reg_weight = (
            reg_weight
            if reg_weight_override is None
            else reg_weight_override
        )
        total = observation_loss + effective_reg_weight * shape_regularization
        return total, (
            observation_loss,
            shape_regularization,
            per_signal,
        )

    def loss(
        trainable,
        scales_batch,
        qr_batch,
        targets_batch,
        batch_weights,
    ):
        theta_batch, ell_nn = trainable
        return law_loss(
            ell_nn,
            theta_batch,
            scales_batch,
            qr_batch,
            targets_batch,
            batch_weights,
        )

    # Les deux objectifs spécialisés sont utilisés par l'optimisation alternée.
    loss.parameter_loss = parameter_loss
    loss.law_loss = law_loss

    return loss


def make_padded_batches(n_samples, batch_size, permutation):
    """
    Retourne des mini-batches de taille fixe.

    Le dernier batch est complete en repetant son premier indice. Les poids
    associes aux elements ajoutes valent zero, donc ils ne contribuent ni a la
    loss ni aux gradients.
    """
    if batch_size <= 0:
        raise ValueError("batch_size doit etre strictement positif.")

    batches = []
    for start in range(0, n_samples, batch_size):
        indices = np.asarray(
            permutation[start : start + batch_size],
            dtype=np.int32,
        )
        valid_count = len(indices)

        if valid_count == 0:
            continue

        weights = np.ones(batch_size, dtype=np.float64)

        if valid_count < batch_size:
            pad_count = batch_size - valid_count
            indices = np.concatenate(
                [
                    indices,
                    np.full(pad_count, indices[0], dtype=np.int32),
                ]
            )
            weights[valid_count:] = 0.0

        batches.append((indices, weights))

    return batches

def make_optimizer(lr, clip_norm, total_steps, final_factor):
    total_steps = max(int(total_steps), 1)
    schedule = optax.cosine_decay_schedule(
        init_value=float(lr), decay_steps=total_steps, alpha=float(final_factor)
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adamw(learning_rate=schedule, weight_decay=1e-7),
    )
    return optimizer, schedule


def get_cycle_lr(lr, cycle_index, n_cycles, final_factor):
    """Atténue légèrement le pas au fil des cycles pour stabiliser l'alternance."""
    lr = jnp.asarray(lr)
    cycle_ratio = jnp.where(
        jnp.asarray(n_cycles <= 1, dtype=jnp.bool_),
        jnp.asarray(1.0, dtype=lr.dtype),
        (cycle_index + 1) / jnp.asarray(n_cycles, dtype=lr.dtype),
    )
    decay = 0.5 + 0.5 * (1.0 - cycle_ratio)
    return lr * jnp.maximum(decay, 0.1) * (1.0 + 0.1 * final_factor)



def take_optimizer_state(state, indices, n_samples):
    """Extrait les moments Adam des seuls signaux du mini-batch."""
    def take_leaf(leaf):
        if eqx.is_array(leaf) and leaf.ndim > 0 and leaf.shape[0] == n_samples:
            return leaf[indices]
        return leaf

    return jax.tree_util.tree_map(take_leaf, state)


def scatter_optimizer_state(state, indices, batch_state, n_samples):
    """Réinsère les moments du batch sans modifier les autres signaux."""
    def scatter_leaf(full_leaf, batch_leaf):
        if (
            eqx.is_array(full_leaf)
            and full_leaf.ndim > 0
            and full_leaf.shape[0] == n_samples
        ):
            return full_leaf.at[indices].set(batch_leaf)
        return full_leaf

    return jax.tree_util.tree_map(scatter_leaf, state, batch_state)


def optimize_joint_stage(
    theta, ell_nn, loss_fn, targets, scales, known_qr, train_mask,
    lr_params, lr_lfunc, n_epochs, local_steps, law_steps,
    params_lr_final_factor, lfunc_lr_final_factor, shuffle_seed,
    print_every, local_print_every,
    grad_clip_params, grad_clip_lfunc, reg_weight, stage_name, log_every,
):
    """
    Optimisation alternée optimisée.

    Bloc A :
      - ell(y) gelée ;
      - gradient calculé uniquement par rapport aux paramètres ;
      - toutes les itérations locales sont compilées dans un lax.scan.

    Bloc B :
      - paramètres gelés ;
      - gradient calculé uniquement par rapport à ell(y) ;
      - accumulation séquentielle des gradients des repeats dans un lax.scan.

    Les conversions vers NumPy/Python ne sont effectuées qu'une fois par repeat
    ou par mise à jour globale de la loi.
    """
    n_repeats = int(theta.shape[0])
    n_signals = int(theta.shape[1])

    optimizer_params, params_schedule = make_optimizer(
        lr_params,
        grad_clip_params,
        max(n_epochs * local_steps, 1),
        params_lr_final_factor,
    )
    optimizer_lfunc, lfunc_schedule = make_optimizer(
        lr_lfunc,
        grad_clip_lfunc,
        max(n_epochs * law_steps, 1),
        lfunc_lr_final_factor,
    )

    state_params = jax.vmap(optimizer_params.init)(theta)
    state_lfunc = optimizer_lfunc.init(
        eqx.filter(ell_nn, eqx.is_array)
    )

    parameter_value_and_grad = jax.value_and_grad(
        loss_fn.parameter_loss,
        argnums=0,
        has_aux=True,
    )

    unit_weights = jnp.ones(n_signals, dtype=jnp.float64)

    @eqx.filter_jit
    def run_parameter_steps(
        theta_group,
        params_state_group,
        ell_model,
        scales_group,
        qr_group,
        targets_group,
        cycle_index,
        repeat_position,
        repeat_index,
    ):
        """Exécute toutes les mises à jour locales dans un seul graphe JAX."""
        step_indices = jnp.arange(local_steps, dtype=jnp.int32)

        def scan_step(carry, local_step):
            theta_current, state_current = carry

            (loss_value, aux), grad_theta = parameter_value_and_grad(
                theta_current,
                ell_model,
                scales_group,
                qr_group,
                targets_group,
                unit_weights,
            )
            observation_loss, shape_regularization, _ = aux

            updates, state_next = optimizer_params.update(
                grad_theta,
                state_current,
                theta_current,
            )
            updates = updates * train_mask
            theta_next = optax.apply_updates(theta_current, updates)

            local_schedule_step = cycle_index * local_steps + local_step
            current_lr = params_schedule(local_schedule_step)
            current_lr = get_cycle_lr(
                current_lr,
                cycle_index,
                n_epochs,
                params_lr_final_factor,
            )

            metrics = (
                loss_value,
                observation_loss,
                shape_regularization,
                current_lr,
            )
            return (theta_next, state_next), metrics

        (theta_final, state_final), metrics = jax.lax.scan(
            scan_step,
            (theta_group, params_state_group),
            step_indices,
        )
        return theta_final, state_final, metrics

    @eqx.filter_jit
    def run_global_law_step(
        ell_model,
        law_state,
        theta_all,
        scales_all,
        qr_all,
        targets_all,
        law_value_and_grad,
    ):
        """
        Calcule le gradient moyen de la loi sur les repeats sans revenir
        dans Python entre deux repeats.
        """
        filtered_lfunc = eqx.filter(ell_model, eqx.is_array)
        zero_grad = jax.tree_util.tree_map(
            jnp.zeros_like,
            filtered_lfunc,
        )

        initial_carry = (
            zero_grad,
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
            jnp.asarray(0.0, dtype=jnp.float64),
        )

        def repeat_step(carry, repeat_index):
            grad_sum, loss_sum, obs_sum, shape_sum = carry

            (loss_value, aux), grad_lfunc = law_value_and_grad(
                ell_model,
                theta_all[repeat_index],
                scales_all[repeat_index],
                qr_all[repeat_index],
                targets_all[repeat_index],
                unit_weights,
            )
            observation_loss, shape_regularization, _ = aux

            grad_sum = jax.tree_util.tree_map(
                lambda accumulated, gradient: accumulated + gradient,
                grad_sum,
                grad_lfunc,
            )
            return (
                grad_sum,
                loss_sum + loss_value,
                obs_sum + observation_loss,
                shape_sum + shape_regularization,
            ), None

        repeat_indices = jnp.arange(n_repeats, dtype=jnp.int32)
        (
            grad_sum,
            loss_sum,
            obs_sum,
            shape_sum,
        ), _ = jax.lax.scan(
            repeat_step,
            initial_carry,
            repeat_indices,
        )

        mean_grad = jax.tree_util.tree_map(
            lambda gradient: gradient / float(n_repeats),
            grad_sum,
        )

        updates, law_state_next = optimizer_lfunc.update(
            mean_grad,
            law_state,
            filtered_lfunc,
        )
        ell_next = eqx.apply_updates(ell_model, updates)

        return (
            ell_next,
            law_state_next,
            loss_sum / float(n_repeats),
            obs_sum / float(n_repeats),
            shape_sum / float(n_repeats),
        )

    history = []
    update_history = []
    final_loss = np.nan
    rng = np.random.default_rng(shuffle_seed)
    global_update = 0
    stage_start = time.time()

    print(f"\n--- {stage_name} ---")
    print(f"Organisation : {n_repeats} repeats x {n_signals} signaux")
    print(f"Cycles externes : {n_epochs}")
    print(
        f"Bloc A parametres : {local_steps} mises a jour/repeat, "
        "ell(y) gelee, lax.scan"
    )
    print(
        f"Bloc B loi : {law_steps} mises a jour globales/cycle, "
        "parametres geles, gradient moyen"
    )

    for cycle in range(n_epochs):
        cycle_start = time.time()
        repeat_order = rng.permutation(n_repeats)

        param_losses = []
        param_observations = []
        param_shapes = []

        for repeat_position, repeat_idx_np in enumerate(
            repeat_order,
            start=1,
        ):
            repeat_idx = int(repeat_idx_np)
            repeat_start = time.time()

            params_state_group = jax.tree_util.tree_map(
                lambda leaf: (
                    leaf[repeat_idx]
                    if (
                        eqx.is_array(leaf)
                        and leaf.ndim > 0
                        and leaf.shape[0] == n_repeats
                    )
                    else leaf
                ),
                state_params,
            )

            (
                theta_group,
                params_state_group,
                metrics,
            ) = run_parameter_steps(
                theta[repeat_idx],
                params_state_group,
                ell_nn,
                scales[repeat_idx],
                known_qr[repeat_idx],
                targets[repeat_idx],
                jnp.asarray(cycle, dtype=jnp.int32),
                jnp.asarray(repeat_position, dtype=jnp.int32),
                jnp.asarray(repeat_idx, dtype=jnp.int32),
            )

            # Une seule synchronisation et un seul transfert par repeat.
            loss_values, obs_values, shape_values, lr_values = (
                np.asarray(value)
                for value in metrics
            )

            theta = theta.at[repeat_idx].set(theta_group)
            state_params = jax.tree_util.tree_map(
                lambda full_leaf, group_leaf: (
                    full_leaf.at[repeat_idx].set(group_leaf)
                    if (
                        eqx.is_array(full_leaf)
                        and full_leaf.ndim > 0
                        and full_leaf.shape[0] == n_repeats
                    )
                    else full_leaf
                ),
                state_params,
                params_state_group,
            )

            param_losses.append(loss_values)
            param_observations.append(obs_values)
            param_shapes.append(shape_values)

            for local_step in range(local_steps):
                global_update += 1
                should_log_update = (
                    (local_step % log_every == 0)
                    or (local_step == local_steps - 1)
                )
                if should_log_update:
                    update_history.append(
                        {
                            "iteration": global_update,
                            "cycle": cycle + 1,
                            "repeat_idx": repeat_idx,
                            "local_step": local_step + 1,
                            "phase": "parameters",
                            "total_loss": float(loss_values[local_step]),
                            "observation_loss": float(obs_values[local_step]),
                            "shape_regularization": float(
                                shape_values[local_step]
                            ),
                            "lr_params": float(lr_values[local_step]),
                            "lr_lfunc": 0.0,
                        }
                    )

            elapsed_cycle = time.time() - cycle_start
            mean_repeat_seconds = elapsed_cycle / float(repeat_position)
            remaining_repeats = n_repeats - repeat_position
            eta_block_a_seconds = (
                mean_repeat_seconds * remaining_repeats
            )

            print(
                f"repeat {repeat_position:2d}/{n_repeats:2d} termine | "
                f"temps_repeat="
                f"{(time.time() - repeat_start) / 60.0:.2f}min | "
                f"temps_bloc_A={elapsed_cycle / 60.0:.2f}min | "
                f"reste_bloc_A~{eta_block_a_seconds / 60.0:.2f}min",
                flush=True,
            )

        param_loss_array = np.concatenate(param_losses)
        param_obs_array = np.concatenate(param_observations)
        param_shape_array = np.concatenate(param_shapes)

        mean_param_loss = float(np.mean(param_loss_array))
        mean_param_obs = float(np.mean(param_obs_array))
        mean_param_shape = float(np.mean(param_shape_array))

        law_losses = []
        law_observations = []
        law_shapes = []

        effective_reg_weight = reg_weight * (
            0.5 + 0.5 * (cycle + 1) / max(n_epochs, 1)
        )
        law_value_and_grad = eqx.filter_value_and_grad(
            lambda ell_model, theta_batch, scales_batch, qr_batch, targets_batch, batch_weights: loss_fn.law_loss(
                ell_model,
                theta_batch,
                scales_batch,
                qr_batch,
                targets_batch,
                batch_weights,
                reg_weight_override=effective_reg_weight,
            ),
            has_aux=True,
        )

        for law_step in range(law_steps):
            (
                ell_nn,
                state_lfunc,
                law_loss_value,
                law_obs_value,
                law_shape_value,
            ) = run_global_law_step(
                ell_nn,
                state_lfunc,
                theta,
                scales,
                known_qr,
                targets,
                law_value_and_grad,
            )

            # Une seule synchronisation par mise à jour globale de la loi.
            law_loss_float = float(law_loss_value)
            law_obs_float = float(law_obs_value)
            law_shape_float = float(law_shape_value)
            local_law_schedule_step = cycle * law_steps + law_step
            current_lr_lfunc = float(
                lfunc_schedule(local_law_schedule_step)
            )
            current_lr_lfunc = float(
                get_cycle_lr(
                    current_lr_lfunc,
                    cycle,
                    n_epochs,
                    lfunc_lr_final_factor,
                )
            )

            law_losses.append(law_loss_float)
            law_observations.append(law_obs_float)
            law_shapes.append(law_shape_float)

            global_update += 1
            should_log_update = (
                (law_step % log_every == 0)
                or (law_step == law_steps - 1)
            )
            if should_log_update:
                update_history.append(
                    {
                        "iteration": global_update,
                        "cycle": cycle + 1,
                        "repeat_idx": -1,
                        "local_step": law_step + 1,
                        "phase": "global_law",
                        "total_loss": law_loss_float,
                        "observation_loss": law_obs_float,
                        "shape_regularization": law_shape_float,
                        "lr_params": 0.0,
                        "lr_lfunc": current_lr_lfunc,
                    }
                )

            print(
                f"cycle {cycle + 1:2d}/{n_epochs:2d} | "
                f"loi iter {law_step + 1:3d}/{law_steps:3d} | "
                f"loss={law_loss_float:.4e} | "
                f"obs={law_obs_float:.4e} | "
                f"lr_l={current_lr_lfunc:.3e} | "
                f"t_cycle="
                f"{(time.time() - cycle_start) / 60.0:.2f}min",
                flush=True,
            )

        mean_law_loss = float(np.mean(law_losses))
        mean_law_obs = float(np.mean(law_observations))
        mean_law_shape = float(np.mean(law_shapes))
        final_loss = mean_law_loss

        epoch_seconds = time.time() - cycle_start
        last_param_schedule_step = (
            (cycle + 1) * local_steps - 1
        )
        last_law_schedule_step = (
            (cycle + 1) * law_steps - 1
        )

        history.append(
            {
                "iteration": cycle,
                "total_loss": final_loss,
                "observation_loss": mean_law_obs,
                "shape_regularization": mean_law_shape,
                "lr_params": float(
                    params_schedule(last_param_schedule_step)
                ),
                "lr_lfunc": float(
                    lfunc_schedule(last_law_schedule_step)
                ),
                "epoch_seconds": float(epoch_seconds),
                "epoch_minutes": float(epoch_seconds / 60.0),
            }
        )

        if cycle % max(print_every, 1) == 0 or cycle == n_epochs - 1:
            completed_epochs = cycle + 1
            elapsed_stage = time.time() - stage_start
            mean_epoch_seconds = elapsed_stage / float(completed_epochs)
            remaining_epochs = n_epochs - completed_epochs
            eta_total_seconds = (
                mean_epoch_seconds * remaining_epochs
            )

            print("=" * 90, flush=True)
            print(
                f"cycle {cycle + 1:3d}/{n_epochs:3d} termine | "
                f"param_loss={mean_param_loss:.4e} | "
                f"law_loss={mean_law_loss:.4e} | "
                f"lr_param="
                f"{float(params_schedule(last_param_schedule_step)):.3e} | "
                f"lr_l="
                f"{float(lfunc_schedule(last_law_schedule_step)):.3e}",
                flush=True,
            )
            print(
                f"temps_epoch={epoch_seconds / 60.0:.2f}min | "
                f"temps_epoch_moy="
                f"{mean_epoch_seconds / 60.0:.2f}min | "
                f"temps_total={elapsed_stage / 3600.0:.2f}h | "
                f"reste_total~={eta_total_seconds / 3600.0:.2f}h",
                flush=True,
            )
            print("=" * 90, flush=True)

    return theta, ell_nn, final_loss, history, update_history



def relative_errors(estimated, true_values):
    denominator = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    errors = jnp.abs((estimated - true_values) / denominator)
    return errors, float(jnp.linalg.norm(errors))


def evaluate_per_signal_losses_batched(
    loss_fn,
    theta,
    ell_nn,
    scales,
    known_qr,
    targets,
    batch_size,
):
    """
    Evalue les losses individuelles sans lancer les 300 solveurs en parallele.
    """
    n_samples = int(theta.shape[0])
    losses = np.empty(n_samples, dtype=np.float64)

    permutation = np.arange(n_samples, dtype=np.int32)
    batches = make_padded_batches(
        n_samples,
        batch_size,
        permutation,
    )

    @eqx.filter_jit
    def evaluate_batch(
        theta_batch,
        ell_model,
        scales_batch,
        qr_batch,
        targets_batch,
        weights_batch,
    ):
        _, aux = loss_fn(
            (theta_batch, ell_model),
            scales_batch,
            qr_batch,
            targets_batch,
            weights_batch,
        )
        return aux[2]

    cursor = 0
    for indices_np, weights_np in batches:
        valid_count = int(np.sum(weights_np))
        indices = jnp.asarray(indices_np, dtype=jnp.int32)
        batch_losses = evaluate_batch(
            theta[indices],
            ell_nn,
            scales[indices],
            known_qr[indices],
            targets[indices],
            jnp.asarray(weights_np, dtype=jnp.float64),
        )
        losses[cursor : cursor + valid_count] = np.asarray(
            batch_losses[:valid_count]
        )
        cursor += valid_count

    return losses

def make_trained_predictions_batched(
    data_init,
    geometry,
    c,
    estimated_values,
    known_qr,
    ell_nn,
    solve_kwargs,
    batch_size,
):
    def one_prediction(params, qr_value):
        data = set_parameter_vector_data(data_init, params, qr_value)
        data = replace_l(data, ell_nn)
        return forward_snapshots(data, geometry, c, **solve_kwargs)

    predict_batch = eqx.filter_jit(
        jax.vmap(one_prediction, in_axes=(0, 0))
    )

    n_samples = int(estimated_values.shape[0])
    outputs = []
    for start in range(0, n_samples, batch_size):
        stop = min(start + batch_size, n_samples)
        outputs.append(
            np.asarray(
                predict_batch(
                    estimated_values[start:stop],
                    known_qr[start:stop],
                )
            )
        )

    return np.concatenate(outputs, axis=0)

def evaluate_l_function(ell_nn, l_true, y_min=-0.2, y_max=1.5, n_points=400):
    y_grid = jnp.linspace(y_min, y_max, n_points)
    ell_pred = jax.vmap(ell_nn)(y_grid)
    ell_reference = jax.vmap(l_true)(y_grid)

    difference = ell_pred - ell_reference
    rel_l2 = (
        jnp.linalg.norm(difference)
        / (jnp.linalg.norm(ell_reference) + 1e-12)
    )
    rmse = jnp.sqrt(jnp.mean(difference**2))
    max_abs = jnp.max(jnp.abs(difference))

    metrics = {
        "relative_l2_error": float(rel_l2),
        "relative_l2_error_percent": 100.0 * float(rel_l2),
        "rmse": float(rmse),
        "max_absolute_error": float(max_abs),
        "ell_at_one": float(ell_nn(jnp.asarray(1.0, dtype=jnp.float64))),
    }
    return y_grid, ell_reference, ell_pred, metrics


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_history(path, stage1_history):
    rows = []
    for global_iteration, row in enumerate(stage1_history):
        rows.append({"global_iteration": global_iteration, "stage": "long", **row})
    write_csv(path, rows, [
        "global_iteration", "stage", "iteration", "total_loss",
        "observation_loss", "shape_regularization",
        "lr_params", "lr_lfunc", "epoch_seconds", "epoch_minutes",
    ])


def signal_fieldnames():
    return [
        "signal_idx",
        "true_gamma",
        "true_kappa",
        "true_zeta",
        "known_Qr",
        "estimated_gamma",
        "estimated_kappa",
        "estimated_zeta",
        "relerr_gamma",
        "relerr_kappa",
        "relerr_zeta",
        "signal_loss",
    ]


def plot_histograms(path, signal_rows):
    arrays = {
        r"$\gamma$": 100.0 * np.asarray(
            [r["relerr_gamma"] for r in signal_rows]
        ),
        r"$\kappa$": 100.0 * np.asarray(
            [r["relerr_kappa"] for r in signal_rows]
        ),
        r"$\zeta$": 100.0 * np.asarray(
            [r["relerr_zeta"] for r in signal_rows]
        ),
    }

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

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
        100.0 * np.asarray([r["relerr_kappa"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_zeta"] for r in signal_rows]),
    ]

    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.boxplot(
        values,
        tick_labels=[r"$\gamma$", r"$\kappa$", r"$\zeta$"],
        showmeans=True,
    )
    ax.set_ylabel("Erreur relative individuelle (%)")
    ax.set_title(f"Erreurs sur {len(signal_rows)} calibrations")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_l_function(path, metrics_path, ell_nn, l_true):
    y_grid, ell_reference, ell_pred, metrics = evaluate_l_function(
        ell_nn,
        l_true,
    )

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_reference),
        label=r"Loi de reference $\ell(y)$",
        linewidth=2.3,
    )
    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_pred),
        "--",
        label=r"Loi apprise $\ell_\theta(y)$",
        linewidth=2.0,
    )
    ax.set_xlabel(r"$y$")
    ax.set_ylabel(r"$\ell(y)$")
    ax.set_title(
        "Apprentissage de la loi d'ouverture\n"
        rf"erreur relative $L^2$ = "
        rf"{metrics['relative_l2_error_percent']:.3f}%"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    write_csv(
        metrics_path,
        [metrics],
        [
            "relative_l2_error",
            "relative_l2_error_percent",
            "rmse",
            "max_absolute_error",
            "ell_at_one",
        ],
    )

    return metrics


def save_update_history(path, update_history):
    write_csv(
        path,
        update_history,
        [
            "iteration",
            "cycle",
            "repeat_idx",
            "local_step",
            "phase",
            "total_loss",
            "observation_loss",
            "shape_regularization",
            "lr_params",
            "lr_lfunc",
        ],
    )


def plot_training_and_l_function(
    path,
    update_history,
    ell_nn,
    l_true,
    fine_start_iteration=None,
):
    """Figure récapitulative : loss à gauche, loi ell(y) à droite."""
    if not update_history:
        raise ValueError("update_history est vide : impossible de tracer la loss.")

    iterations = np.asarray(
        [row["iteration"] for row in update_history], dtype=np.int64
    )
    losses = np.asarray(
        [row["total_loss"] for row in update_history], dtype=np.float64
    )

    y_grid, ell_reference, ell_pred, metrics = evaluate_l_function(
        ell_nn,
        l_true,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.0))

    ax = axes[0]
    ax.semilogy(iterations, losses, linewidth=1.5)
    if fine_start_iteration is not None:
        ax.axvline(
            fine_start_iteration,
            linestyle="--",
            linewidth=1.6,
            label="Début phase fine",
        )
        ax.legend()
    ax.set_xlabel("Itération")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.grid(True, alpha=0.45)

    ax = axes[1]
    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_reference),
        linewidth=2.2,
        label="ReedOpening vraie",
    )
    ax.plot(
        np.asarray(y_grid),
        np.asarray(ell_pred),
        linewidth=2.2,
        label=(
            "MLP appris\n"
            f"rel={metrics['relative_l2_error_percent']:.2f}%\n"
            f"max={metrics['max_absolute_error']:.3e}"
        ),
    )
    ax.set_xlabel("y")
    ax.set_ylabel("l(y)")
    ax.set_title("Comparaison de l(y)")
    ax.grid(True, alpha=0.45)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return metrics


def plot_training_curves(path, stage1_history):
    iterations = np.arange(len(stage1_history))
    total = np.asarray([row["total_loss"] for row in stage1_history])
    observation = np.asarray(
        [row["observation_loss"] for row in stage1_history]
    )
    shape_regularization = np.asarray(
        [row["shape_regularization"] for row in stage1_history]
    )

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.semilogy(iterations, total, label="Loss totale")
    ax.semilogy(iterations, observation, label="Loss MSTS")

    if np.any(shape_regularization > 0.0):
        ax.semilogy(
            iterations,
            shape_regularization,
            label="Régularisation de forme",
        )

    ax.set_xlabel("Époque")
    ax.set_ylabel("Loss")
    ax.set_title("Historique d'entraînement sur la fenêtre longue")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


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
        ax.plot(
            times,
            predictions[i],
            "--",
            label="DG calibre",
            linewidth=1.2,
        )
        ax.set_title(f"Signal {i + 1}")
        ax.set_xlabel("Temps (s)")
        ax.set_ylabel("Pression / P_closed")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Signaux OpenWind et DG calibre")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def initialize_joint_state(
    true_values,
    scales,
    train_mask,
    lfunc_hidden_width,
    lfunc_hidden_layers,
    seed,
):
    """Initialise les paramètres locaux par signal et la loi globale partagée."""
    normalized_true = true_values / scales
    normalized_init = normalized_true * (
        train_mask * DEFAULT_INIT_FACTOR
        + (1.0 - train_mask)
    )
    local_params = inverse_softplus(normalized_init)

    key = jax.random.PRNGKey(seed)
    hidden = [lfunc_hidden_width] * lfunc_hidden_layers
    layer_sizes = [1, *hidden, 1]
    global_law = NormalizedLFuncNN(
        layer_sizes,
        activation=jax.nn.tanh,
        key=key,
    )
    return local_params, global_law, layer_sizes


def validate_args(args):
    if args.n_signals <= 0 or args.n_repeats <= 0:
        raise ValueError(
            "--n_signals et --n_repeats doivent etre strictement positifs."
        )

    for name in ("stage1_iter",):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} doit etre positif ou nul.")

    for name in (
        "stage1_lr_params",
        "stage1_lr_lfunc",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} doit etre strictement positif.")


    if args.local_steps <= 0:
        raise ValueError("--local_steps doit etre strictement positif.")
    if args.fast_test:
        args.n_signals = min(args.n_signals, 20)
        args.n_repeats = min(args.n_repeats, 6)
        args.local_steps = min(args.local_steps, 4)
        args.law_steps = min(args.law_steps, 1)
        args.stage1_iter = min(args.stage1_iter, 1)
    if args.law_steps <= 0:
        raise ValueError("--law_steps doit etre strictement positif.")
    for name in ("params_lr_final_factor", "lfunc_lr_final_factor"):
        value = getattr(args, name)
        if not (0.0 < value <= 1.0):
            raise ValueError(f"--{name} doit appartenir a ]0, 1].")

    if args.reg_weight < 0.0:
        raise ValueError("--reg_weight doit etre positif ou nul.")

    if args.batch_size <= 0:
        raise ValueError("--batch_size doit etre strictement positif.")

    if args.local_print_every <= 0:
        raise ValueError(
            "--local_print_every doit etre strictement positif."
        )
    if args.log_every <= 0:
        raise ValueError("--log_every doit etre strictement positif.")

    if args.lfunc_hidden_width <= 0:
        raise ValueError("--lfunc_hidden_width doit etre strictement positif.")

    if args.lfunc_hidden_layers < 0:
        raise ValueError("--lfunc_hidden_layers doit etre positif ou nul.")


def main():
    args = parse_args()
    validate_args(args)
    total_start = time.time()

    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding

    root = repo_root()

    with open(
        root / "experiments/gradient/config/simu.json",
        "r",
    ) as file:
        train_config = json.load(file)["solver_params"]["train"]

    with open(
        root / "experiments/gradient/config/param.json",
        "r",
    ) as file:
        params = json.load(file)

    generation_T = float(train_config["T_max"])
    calibration_T = float(train_config.get("T_long", generation_T))

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
    l_true = data_ref.l

    geometry = build_solver_geometry(
        data_ref,
        train_config["Nx"],
        c,
    )
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
        "T_long": generation_T,
        "Nx": int(train_config["Nx"]),
        "N_snapshot": int(train_config["N_snapshot"]),
        "cfl": float(train_config["cfl"]),
        "c": float(c),
        "ow_order": int(args.ow_order),
        "ow_theta": float(args.ow_theta),
        "ow_l_ele": float(
            args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4
        ),
    }

    metadata, arrays, radiation = load_openwind_dataset(
        dataset_path,
        expected_dataset,
    )

    long_keep = int(
        np.searchsorted(
            arrays["times_long"],
            calibration_T,
            side="right",
        )
    )
    long_keep = max(1, long_keep)

    length = data_ref.section.L_tube + data_ref.section.L_bell

    solve_long = make_solver_data(
        train_config["T_max"],
        train_config["cfl"],
        train_config["Nx"],
        train_config["N_snapshot"],
        length,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    times_long = (solve_long["n_snaps"] + 1) * solve_long["dt"]

    matched_params = params_with_openwind_radiation(
        params_trainable,
        radiation,
        args.type_S,
    )
    data_matched = build_physical_data(
        matched_params,
        args.type_S,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    total_signals = int(metadata["simulation_count"])
    dataset_n_repeats = int(metadata["n_repeats"])
    dataset_n_signals = int(metadata["n_signals"])

    # Structure conservée : (repeat, signal, parametre).
    # Cache historique : [gamma, kappa, zeta, Qr].
    all_true_values = np.asarray(arrays["true_values"])
    true_values_np = all_true_values[..., :3]
    known_qr_np = all_true_values[..., 3]

    targets_long = jnp.asarray(
        np.asarray(arrays["pressure_long"])[..., :long_keep],
        dtype=jnp.float64,
    )

    true_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    known_qr = jnp.asarray(known_qr_np, dtype=jnp.float64)

    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)

    train_mask = jnp.asarray(
        [name in args.train_params for name in PARAM_NAMES],
        dtype=jnp.float64,
    )[None, :]

    theta, ell_nn, layer_sizes = initialize_joint_state(
        true_values=true_values,
        scales=scales,
        train_mask=train_mask,
        lfunc_hidden_width=args.lfunc_hidden_width,
        lfunc_hidden_layers=args.lfunc_hidden_layers,
        seed=args.seed,
    )

    loss_long = make_minibatch_loss(
        data_matched,
        geometry,
        c,
        solve_long,
        stft_resolutions,
        args.stft_dynamic_db,
        stft_allow_padding,
        args.reg_weight,
    )


    print("\n=== Experience gamma-kappa-zeta + ell(y) ===")
    print(f"Signaux dans le batch : {total_signals}")
    print(f"Parametres calibres   : {', '.join(args.train_params)}")
    print("Qr                     : connu et fixe pour chaque signal")
    print("Observation            : p(L,t) uniquement")
    print("Protocole              : optimisation alternee par blocs")
    print("Bloc A                  : parametres locaux, ell(y) gelee")
    print("Bloc B                  : loi globale, parametres geles")
    print("Etat Adam parametres    : independant et persistant pour chaque repeat")
    print("Etat Adam ell(y)        : unique et persistant")
    print(f"Groupes                 : {dataset_n_repeats} repeats x {dataset_n_signals} signaux")
    print(f"Cycles externes          : {args.stage1_iter}")
    print(f"Mises a jour param/repeat: {args.local_steps}")
    print(f"Mises a jour loi/cycle   : {args.law_steps}")
    print(
        f"Affichage local          : toutes les "
        f"{args.local_print_every} iterations"
    )
    print(f"Mises a jour/parametre   : {args.stage1_iter * args.local_steps}")
    print(f"Architecture ell(y)    : {layer_sizes}")
    print(f"Poids regularisation   : {args.reg_weight:.3e}")
    print("Normalisation ell(1)=1 : exacte, par construction")
    print(f"Dataset OpenWind       : {dataset_path}")
    print(f"Fenetre longue         : {calibration_T:.4f} s")
    print(f"STFT                    : {stft_resolutions}")

    theta, ell_nn, loss_stage1, history_stage1, update_history = optimize_joint_stage(
        theta=theta,
        ell_nn=ell_nn,
        loss_fn=loss_long,
        targets=targets_long,
        scales=scales,
        known_qr=known_qr,
        train_mask=train_mask,
        lr_params=args.stage1_lr_params,
        lr_lfunc=args.stage1_lr_lfunc,
        n_epochs=args.stage1_iter,
        local_steps=args.local_steps,
        law_steps=args.law_steps,
        params_lr_final_factor=args.params_lr_final_factor,
        lfunc_lr_final_factor=args.lfunc_lr_final_factor,
        shuffle_seed=args.shuffle_seed,
        print_every=args.print_every,
        local_print_every=args.local_print_every,
        grad_clip_params=args.grad_clip_params,
        grad_clip_lfunc=args.grad_clip_lfunc,
        reg_weight=args.reg_weight,
        stage_name=(
            f"Etape 1 : optimisation alternee parametres/loi, "
            f"T={calibration_T:.4f} s"
        ),
        log_every=args.log_every,
    )


    estimated = positive_parameters(theta, scales)
    errors, global_error = relative_errors(
        estimated,
        true_values,
    )

    theta_flat = theta.reshape(total_signals, len(PARAM_NAMES))
    scales_flat = scales.reshape(total_signals, len(PARAM_NAMES))
    known_qr_flat = known_qr.reshape(total_signals)
    targets_flat = targets_long.reshape(
        total_signals,
        targets_long.shape[-1],
    )

    signal_losses = evaluate_per_signal_losses_batched(
        loss_fn=loss_long,
        theta=theta_flat,
        ell_nn=ell_nn,
        scales=scales_flat,
        known_qr=known_qr_flat,
        targets=targets_flat,
        batch_size=args.batch_size,
    )
    final_loss = float(np.mean(signal_losses))

    true_np = np.asarray(true_values).reshape(
        total_signals,
        len(PARAM_NAMES),
    )
    estimated_np = np.asarray(estimated).reshape(
        total_signals,
        len(PARAM_NAMES),
    )
    errors_np = np.asarray(errors).reshape(
        total_signals,
        len(PARAM_NAMES),
    )
    known_qr_flat_np = np.asarray(known_qr_np).reshape(total_signals)

    signal_rows = []
    for i in range(total_signals):
        signal_rows.append(
            {
                "signal_idx": i,
                "true_gamma": float(true_np[i, 0]),
                "true_kappa": float(true_np[i, 1]),
                "true_zeta": float(true_np[i, 2]),
                "known_Qr": float(known_qr_flat_np[i]),
                "estimated_gamma": float(estimated_np[i, 0]),
                "estimated_kappa": float(estimated_np[i, 1]),
                "estimated_zeta": float(estimated_np[i, 2]),
                "relerr_gamma": float(errors_np[i, 0]),
                "relerr_kappa": float(errors_np[i, 1]),
                "relerr_zeta": float(errors_np[i, 2]),
                "signal_loss": float(signal_losses[i]),
            }
        )

    all_signals_path = output_dir / "all_signals.csv"
    history_path = output_dir / "training_history.csv"
    l_metrics_path = output_dir / "l_function_errors.csv"
    l_plot_path = output_dir / "l_theta_vs_reference.png"
    training_plot_path = output_dir / "training_curves.png"
    summary_plot_path = output_dir / "training_loss_and_l_function.png"
    update_history_path = output_dir / "training_updates.csv"
    histogram_path = output_dir / "error_histograms.png"
    boxplot_path = output_dir / "error_boxplots.png"
    network_path = output_dir / "ell_nn.eqx"
    theta_path = output_dir / "estimated_parameters.npy"

    write_csv(
        all_signals_path,
        signal_rows,
        signal_fieldnames(),
    )
    # Sauvegarde prioritaire des resultats couteux avant les exports CSV/figures.
    # Une erreur d export ne peut ainsi plus faire perdre l entrainement.
    eqx.tree_serialise_leaves(network_path, ell_nn)
    np.save(theta_path, estimated_np)

    save_history(
        history_path,
        history_stage1,
    )
    save_update_history(update_history_path, update_history)
    plot_histograms(histogram_path, signal_rows)
    plot_boxplots(boxplot_path, signal_rows)
    plot_training_curves(
        training_plot_path,
        history_stage1,
    )
    l_metrics = plot_training_and_l_function(
        summary_plot_path,
        update_history,
        ell_nn,
        l_true,
        fine_start_iteration=None,
    )
    plot_l_function(
        l_plot_path,
        l_metrics_path,
        ell_nn,
        l_true,
    )


    if args.save_detailed_plots:
        predictions = make_trained_predictions_batched(
            data_matched,
            geometry,
            c,
            estimated.reshape(total_signals, len(PARAM_NAMES)),
            known_qr.reshape(total_signals),
            ell_nn,
            solve_long,
            args.batch_size,
        )[:, : targets_long.shape[1]]

        plot_signal_comparison(
            output_dir / "signals_openwind_vs_dg.png",
            np.asarray(times_long[:long_keep]),
            np.asarray(targets_flat),
            np.asarray(predictions),
        )

    print("\n=== Resume global ===")
    for name in PARAM_NAMES:
        values = np.asarray(
            [row[f"relerr_{name}"] for row in signal_rows]
        )
        print(
            f"{name:5s}: "
            f"mean={100*np.mean(values):.3f}% | "
            f"std={100*np.std(values):.3f}% | "
            f"median={100*np.median(values):.3f}% | "
            f"q95={100*np.quantile(values, 0.95):.3f}% | "
            f"max={100*np.max(values):.3f}%"
        )

    print(f"Erreur globale         : {global_error:.6e}")
    print(f"Loss entrainement long : {loss_stage1:.6e}")
    print(f"Loss finale longue     : {final_loss:.6e}")
    print(
        "Erreur relative L2 ell: "
        f"{l_metrics['relative_l2_error_percent']:.3f}%"
    )
    print(f"Temps total            : {(time.time() - total_start)/3600:.2f} h")
    print(f"Resultats               : {output_dir}")
    print(f"CSV signaux             : {all_signals_path}")
    print(f"Historique cycles       : {history_path}")
    print(f"Historique iterations   : {update_history_path}")
    print(f"Figure loss + ell(y)    : {summary_plot_path}")
    print(f"Histogrammes erreurs    : {histogram_path}")
    print(f"Reseau ell(y)           : {network_path}")


if __name__ == "__main__":
    main()
