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
from inverse.spectral_loss import multi_resolution_spectral_residuals
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)

MIN_SCALE = 1e-8
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
PARAM_NAMES = ("gamma", "kappa", "zeta")

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
            "Calibration de gamma, kappa et zeta avec Qr connu et fixe pour chaque signal : "
            "optimisation conjointe sur fenetre longue, fenetre courte, "
            "puis raffinement conjoint optionnel."
        )
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument("--n_signals", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--init_gamma", type=float, default=0.30)
    parser.add_argument("--init_kappa", type=float, default=0.50)
    parser.add_argument("--init_zeta", type=float, default=0.30)
    parser.add_argument("--scale_gamma", type=float, default=0.10)
    parser.add_argument("--scale_kappa", type=float, default=0.10)
    parser.add_argument("--scale_zeta", type=float, default=0.10)

    # Recherche multi-start avant l'entraînement principal.
    parser.add_argument(
        "--multistart_inits",
        type=str,
        default="",
        help=(
            "Liste de triplets gamma:kappa:zeta séparés par des ';'. "
            "Exemple : "
            "'0.30:0.50:0.30;0.30:0.65:0.30;"
            "0.30:0.80:0.30;0.30:0.95:0.30'. "
            "Chaîne vide : multi-start désactivé."
        ),
    )
    parser.add_argument(
        "--multistart_iter",
        type=int,
        default=0,
        help=(
            "Nombre d'itérations Adam courtes effectuées pour chaque "
            "initialisation du multi-start."
        ),
    )
    parser.add_argument(
        "--multistart_lr",
        type=float,
        default=1e-2,
        help="Learning rate de la phase exploratoire multi-start.",
    )
    parser.add_argument(
        "--multistart_lr_final_factor",
        type=float,
        default=0.9,
        help="Facteur final du learning rate pendant le multi-start.",
    )
    parser.add_argument(
        "--multistart_print_every",
        type=int,
        default=25,
        help="Fréquence d'affichage pendant chaque départ du multi-start.",
    )

    # Second multi-start ciblé sur les signaux encore mal calibrés.
    parser.add_argument(
        "--hard_loss_threshold",
        type=float,
        default=0.5,
        help=(
            "Après le Stage 1, les signaux dont la loss individuelle dépasse "
            "ce seuil sont relancés avec un second multi-start."
        ),
    )
    parser.add_argument(
        "--hard_relative_offsets",
        type=str,
        default=(
            "0:0:0;"
            "-0.08:0:0;0.08:0:0;"
            "0:-0.15:0;0:0.15:0;"
            "0:0:-0.10;0:0:0.10;"
            "-0.06:0.12:0.08;0.06:-0.12:-0.08;"
            "-0.06:-0.12:0.08;0.06:0.12:-0.08"
        ),
        help=(
            "Perturbations physiques dgamma:dkappa:dzeta utilisées autour "
            "de l'estimation de chaque signal difficile. Les triplets sont "
            "séparés par des ';'."
        ),
    )
    parser.add_argument(
        "--hard_param_mins",
        type=str,
        default="0.30:0.53:0.30",
        help="Bornes inférieures gamma:kappa:zeta du second multi-start relatif.",
    )
    parser.add_argument(
        "--hard_param_maxs",
        type=str,
        default="0.53:0.90:0.50",
        help="Bornes supérieures gamma:kappa:zeta du second multi-start relatif.",
    )
    parser.add_argument(
        "--hard_multistart_iter",
        type=int,
        default=0,
        help=(
            "Nombre d'itérations Adam par départ pour le second multi-start "
            "ciblé. Mettre 0 pour désactiver cette phase."
        ),
    )
    parser.add_argument(
        "--hard_multistart_lr",
        type=float,
        default=5e-3,
        help="Learning rate du second multi-start ciblé.",
    )
    parser.add_argument(
        "--hard_multistart_lr_final_factor",
        type=float,
        default=0.5,
        help="Facteur final du learning rate du second multi-start ciblé.",
    )
    parser.add_argument(
        "--hard_multistart_print_every",
        type=int,
        default=10,
        help="Fréquence d'affichage du second multi-start ciblé.",
    )
    parser.add_argument(
        "--hard_refine_iter",
        type=int,
        default=0,
        help=(
            "Nombre d'itérations Adam faible LR appliquées uniquement aux "
            "signaux relancés après le second multi-start."
        ),
    )
    parser.add_argument(
        "--hard_refine_lr",
        type=float,
        default=1e-3,
        help="Learning rate du raffinement ciblé après le second multi-start.",
    )
    parser.add_argument(
        "--hard_refine_lr_final_factor",
        type=float,
        default=0.1,
        help="Facteur final du learning rate du raffinement ciblé.",
    )
    parser.add_argument(
        "--train_params",
        nargs="+",
        choices=PARAM_NAMES,
        default=list(PARAM_NAMES),
        help=(
            "Parametres a entrainer parmi gamma kappa zeta. "
            "Qr reste connu et fixe à sa valeur OpenWind pour chaque signal."
        ),
    )

    parser.add_argument("--stage1_iter", type=int, default=400)
    parser.add_argument("--stage2_iter", type=int, default=100)
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage2_lr", type=float, default=5e-3)
    parser.add_argument("--stage1_lr_final_factor", type=float, default=0.9)
    parser.add_argument("--stage2_lr_final_factor", type=float, default=0.8)

    # Stage 3 : petit raffinement conjoint facultatif.
    parser.add_argument("--final_joint_iter", type=int, default=0)
    parser.add_argument("--final_joint_lr", type=float, default=5e-4)
    parser.add_argument("--final_joint_lr_final_factor", type=float, default=0.5)
    parser.add_argument(
        "--skip_final_joint",
        action="store_true",
        help="Desactive le raffinement conjoint final.",
    )

    parser.add_argument("--print_every", type=int, default=50)

    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=None)

    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="16:4,32:8,64:16,128:32",
        help="Liste n_fft:hop, par exemple 64:16,128:32,256:64.",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument(
        "--no_stft_padding",
        action="store_true",
        help="Ignore une resolution STFT trop grande au lieu de zero-padder.",
    )

    # Raffinement Gauss-Newton spectral sur les résidus MSTS.
    parser.add_argument("--gauss_newton_iter", type=int, default=0,
                        help="Nombre d'iterations de Gauss-Newton apres Adam.")
    parser.add_argument("--gauss_newton_step", type=float, default=0.25,
                        help="Facteur multiplicatif du pas de Gauss-Newton.")
    parser.add_argument("--gauss_newton_ridge", type=float, default=1e-4,
                        help="Amortissement relatif ajoute a J^T J.")
    parser.add_argument(
        "--gauss_newton_fd_eps",
        type=float,
        default=1e-3,
        help=(
            "Pas de differences finies centrales dans l'espace normalise "
            "pour construire le Jacobien GN."
        ),
    )
    parser.add_argument("--gauss_newton_batch_size", type=int, default=10,
                        help="Nombre de signaux traites simultanement par GN.")
    parser.add_argument("--gauss_newton_print_every", type=int, default=1)
    parser.add_argument("--skip_gauss_newton", action="store_true")

    # Raffinement final L-BFGS sur la loss scalaire deja utilisee par Adam.
    parser.add_argument(
        "--lbfgs_iter",
        type=int,
        default=0,
        help="Nombre maximal d'iterations L-BFGS apres les stages Adam.",
    )
    parser.add_argument(
        "--lbfgs_memory",
        type=int,
        default=5,
        help="Nombre de couples (pas, gradient) conserves par L-BFGS.",
    )
    parser.add_argument(
        "--lbfgs_tol",
        type=float,
        default=1e-5,
        help="Arret si la norme du gradient devient inferieure a ce seuil.",
    )
    parser.add_argument(
        "--lbfgs_max_linesearch_steps",
        type=int,
        default=5,
        help="Nombre maximal d'essais de la recherche lineaire par iteration.",
    )
    parser.add_argument(
        "--lbfgs_print_every",
        type=int,
        default=1,
        help="Frequence d'affichage des iterations L-BFGS.",
    )
    parser.add_argument(
        "--skip_lbfgs",
        action="store_true",
        help="Desactive le raffinement final L-BFGS.",
    )

    parser.add_argument("--random_factor_min", type=float, default=0.75)
    parser.add_argument("--random_factor_max", type=float, default=1.25)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/gradient/results/gamma_kappa_zeta_Qr_fixed_pressure_only",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="experiments/gradient/datasets/openwind_Qr_kappa_gamma_zeta_300.npz",
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
    """Load and validate the shared pregenerated OpenWind dataset."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset OpenWind absent: {path}. Lance d'abord "
            "src/generate_openwind_Qr_kappa_gamma_zeta_dataset.py."
        )
    with np.load(path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))
        # n_signals et n_repeats décrivent seulement l'organisation interne
        # du cache. L'optimisation les fusionne ensuite en un batch unique.
        # On valide donc exactement les paramètres physiques/numériques,
        # mais pas la décomposition interne 30 x 10 ou 1 x 300.
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

        requested_total = int(expected["n_signals"]) * int(expected["n_repeats"])
        dataset_total = int(metadata["n_signals"]) * int(metadata["n_repeats"])
        if requested_total != dataset_total:
            raise ValueError(
                "Dataset incompatible pour le nombre total de signaux: "
                f"attendu={requested_total}, trouve={dataset_total} "
                f"({metadata['n_repeats']} x {metadata['n_signals']})."
            )
        arrays = {
            name: np.asarray(dataset[name])
            for name in (
                "true_values",
                "pressure_short",
                "pressure_long",
                "times_short",
                "times_long",
            )
        }
        radiation = {
            "alpha": float(dataset["radiation_alpha"]),
            "beta": float(dataset["radiation_beta"]),
        }

    # Les formes sur disque suivent l'organisation réelle du cache.
    expected_prefix = (metadata["n_repeats"], metadata["n_signals"])
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


def parse_multistart_inits(value):
    """Parse gamma:kappa:zeta;gamma:kappa:zeta;..."""
    value = value.strip()
    if not value:
        return np.empty((0, len(PARAM_NAMES)), dtype=float)

    starts = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "," if "," in item else None
        if separator is None:
            raise ValueError(
                f"Initialisation multi-start invalide: {item!r}. "
                "Format attendu gamma:kappa:zeta."
            )
        values = [float(x.strip()) for x in item.split(separator)]
        if len(values) != len(PARAM_NAMES):
            raise ValueError(
                "Chaque départ multi-start doit contenir exactement "
                "gamma, kappa et zeta."
            )
        if any(x <= 0.0 for x in values):
            raise ValueError(
                "Toutes les valeurs multi-start doivent être strictement positives."
            )
        starts.append(values)
    return np.asarray(starts, dtype=float)


def parse_triplets(value, *, allow_negative=True):
    """Parse des triplets ``a:b:c;a:b:c`` sous forme de tableau NumPy."""
    value = value.strip()
    if not value:
        return np.empty((0, len(PARAM_NAMES)), dtype=float)
    triplets = []
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        separator = ":" if ":" in item else "," if "," in item else None
        if separator is None:
            raise ValueError(f"Triplet invalide {item!r}; format attendu a:b:c.")
        vals = [float(x.strip()) for x in item.split(separator)]
        if len(vals) != len(PARAM_NAMES):
            raise ValueError("Chaque entrée doit contenir gamma, kappa et zeta.")
        if not allow_negative and any(v <= 0.0 for v in vals):
            raise ValueError("Les valeurs doivent être strictement positives.")
        triplets.append(vals)
    return np.asarray(triplets, dtype=float)


def parse_single_triplet(value, name):
    arr = parse_triplets(value, allow_negative=True)
    if arr.shape != (1, len(PARAM_NAMES)):
        raise ValueError(f"{name} doit contenir exactement un triplet gamma:kappa:zeta.")
    return arr[0]


def base_parameter_vector_from_params(params):
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    kappa = float(get_nested(params, PARAM_JSON_PATHS["kappa"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    return np.asarray([gamma, kappa, zeta], dtype=float)



def set_parameter_vector_json(params, values):
    gamma, kappa, zeta = map(float, values)
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["kappa"], kappa)
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    return params



def set_parameter_vector_data(data, values, qr_value):
    """
    Injecte les trois paramètres calibrés et la valeur connue de Qr.

    Qr reste différent d'un signal à l'autre, mais il n'est jamais optimisé.
    """
    gamma, kappa, zeta = values
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    data = set_param(data, "Qr", qr_value, GEO_KEYS)
    return data



def set_trainable_parameters(params):
    params = copy.deepcopy(params)

    # Qr doit rester remplaçable signal par signal dans le PyTree,
    # mais il ne fait pas partie de PARAM_NAMES et n'est donc pas optimisé.
    dynamic_names = ("gamma_final", "kappa", "zeta", "Qr")
    for name in params["trainable"]:
        params["trainable"][name] = name in dynamic_names
    return params




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


def make_batched_loss(
    data_init,
    geometry,
    c,
    targets,
    scales,
    qr_values,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(theta_one, scale_one, qr_one, target_one):
        calibrated_values = theta_one * scale_one
        data = set_parameter_vector_data(
            data_init,
            calibrated_values,
            qr_one,
        )
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        pred = pred[: target_one.shape[0]]
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(
        one_loss,
        in_axes=(0, 0, 0, 0),
    )

    def loss(theta):
        return jnp.mean(
            vmapped_loss(theta, scales, qr_values, targets)
        )

    return loss


def make_losses_per_signal(
    data_init,
    geometry,
    c,
    targets,
    scales,
    qr_values,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(theta_one, scale_one, qr_one, target_one):
        calibrated_values = theta_one * scale_one
        data = set_parameter_vector_data(
            data_init,
            calibrated_values,
            qr_one,
        )
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        pred = pred[: target_one.shape[0]]
        return loss_fn_signal(
            pred,
            target_one,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )

    vmapped_loss = jax.vmap(
        one_loss,
        in_axes=(0, 0, 0, 0),
    )

    @jax.jit
    def evaluate(theta):
        return vmapped_loss(
            theta,
            scales,
            qr_values,
            targets,
        )

    return evaluate


def optimize_stage(
    theta, loss_fn, train_mask, lr, n_iter, print_every, final_lr_factor
):
    optimizer, scheduler = make_optimizer(lr, n_iter, final_lr_factor)
    opt_state = optimizer.init(theta)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    final_loss = np.nan
    for iteration in range(n_iter):
        start = time.time()
        loss_value, gradient = value_and_grad(theta)
        updates, opt_state = optimizer.update(gradient, opt_state, theta)
        # Apply the mask after AdamW so that weight decay cannot move a
        # parameter that was requested as fixed.
        updates = updates * train_mask
        theta = optax.apply_updates(theta, updates)
        final_loss = float(loss_value)

        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            print(
                f"iter {iteration:4d} | loss={final_loss:.4e} | "
                f"lr={float(scheduler(iteration)):.3e} | "
                f"t={time.time() - start:.2f}s"
            )

    return theta, final_loss




def optimize_multistart(
    *, starts_physical, scales, loss_fn, losses_per_signal_fn, train_mask,
    lr, n_iter, final_lr_factor, print_every,
):
    """
    Lance plusieurs Adam courts puis conserve, pour chaque signal,
    le départ donnant la loss individuelle la plus faible.
    """
    if n_iter <= 0 or starts_physical.shape[0] == 0:
        raise ValueError(
            "Le multi-start nécessite des départs et --multistart_iter > 0."
        )

    n_signals = int(scales.shape[0])
    n_starts = int(starts_physical.shape[0])
    best_theta = None
    best_losses = None
    best_start_indices = None

    for start_idx in range(n_starts):
        start_values = jnp.asarray(starts_physical[start_idx], dtype=jnp.float64)
        theta_one = start_values / scales[0]
        theta_start = jnp.broadcast_to(
            theta_one[None, :], (n_signals, len(PARAM_NAMES))
        ).copy()

        print(
            f"\n--- Multi-start {start_idx + 1}/{n_starts} | "
            f"gamma={float(start_values[0]):.4g}, "
            f"kappa={float(start_values[1]):.4g}, "
            f"zeta={float(start_values[2]):.4g} ---"
        )

        theta_candidate, mean_loss = optimize_stage(
            theta_start, loss_fn, train_mask, lr, n_iter,
            print_every, final_lr_factor,
        )
        candidate_losses = losses_per_signal_fn(theta_candidate)
        candidate_losses.block_until_ready()

        if best_theta is None:
            best_theta = theta_candidate
            best_losses = candidate_losses
            best_start_indices = jnp.full(
                (n_signals,), start_idx, dtype=jnp.int32
            )
        else:
            improved = candidate_losses < best_losses
            best_theta = jnp.where(improved[:, None], theta_candidate, best_theta)
            best_losses = jnp.where(improved, candidate_losses, best_losses)
            best_start_indices = jnp.where(
                improved, jnp.asarray(start_idx, dtype=jnp.int32),
                best_start_indices,
            )

        print(
            f"Fin départ {start_idx + 1}: loss moyenne={float(mean_loss):.4e}, "
            f"meilleure moyenne cumulée={float(jnp.mean(best_losses)):.4e}"
        )

    best_theta.block_until_ready()
    counts = np.bincount(np.asarray(best_start_indices), minlength=n_starts)
    print("\n=== Sélection multi-start par signal ===")
    for start_idx, count in enumerate(counts):
        values = starts_physical[start_idx]
        print(
            f"Départ {start_idx + 1}: gamma={values[0]:.4g}, "
            f"kappa={values[1]:.4g}, zeta={values[2]:.4g} "
            f"-> {int(count)}/{n_signals} signaux"
        )
    print(
        "Loss moyenne après sélection multi-start : "
        f"{float(jnp.mean(best_losses)):.4e}"
    )
    return best_theta, best_losses, best_start_indices



def optimize_relative_multistart(
    *, centers_physical, offsets_physical, parameter_mins, parameter_maxs,
    scales, loss_fn, losses_per_signal_fn, train_mask,
    lr, n_iter, final_lr_factor, print_every,
):
    """Multi-start relatif avec un centre propre à chaque signal.

    ``centers_physical`` a la forme (n_signaux, 3). Pour chaque perturbation,
    le départ est ``clip(centre + perturbation, mins, maxs)``.
    """
    if n_iter <= 0 or offsets_physical.shape[0] == 0:
        raise ValueError("Le multi-start relatif nécessite des offsets et n_iter > 0.")

    n_signals = int(scales.shape[0])
    n_starts = int(offsets_physical.shape[0])
    mins = jnp.asarray(parameter_mins, dtype=jnp.float64)
    maxs = jnp.asarray(parameter_maxs, dtype=jnp.float64)
    centers = jnp.asarray(centers_physical, dtype=jnp.float64)

    best_theta = None
    best_losses = None
    best_start_indices = None
    best_start_values = None

    for start_idx in range(n_starts):
        offset = jnp.asarray(offsets_physical[start_idx], dtype=jnp.float64)
        start_values = jnp.clip(centers + offset[None, :], mins, maxs)
        theta_start = start_values / scales

        print(
            f"\n--- Départ relatif {start_idx + 1}/{n_starts} | "
            f"Δgamma={float(offset[0]):+.3f}, "
            f"Δkappa={float(offset[1]):+.3f}, "
            f"Δzeta={float(offset[2]):+.3f} ---"
        )
        theta_candidate, mean_loss = optimize_stage(
            theta_start, loss_fn, train_mask, lr, n_iter,
            print_every, final_lr_factor,
        )
        candidate_losses = losses_per_signal_fn(theta_candidate)
        candidate_losses.block_until_ready()

        if best_theta is None:
            best_theta = theta_candidate
            best_losses = candidate_losses
            best_start_indices = jnp.full((n_signals,), start_idx, dtype=jnp.int32)
            best_start_values = start_values
        else:
            improved = candidate_losses < best_losses
            best_theta = jnp.where(improved[:, None], theta_candidate, best_theta)
            best_losses = jnp.where(improved, candidate_losses, best_losses)
            best_start_indices = jnp.where(
                improved, jnp.asarray(start_idx, dtype=jnp.int32), best_start_indices
            )
            best_start_values = jnp.where(improved[:, None], start_values, best_start_values)

        print(
            f"Fin offset {start_idx + 1}: loss moyenne={float(mean_loss):.4e}, "
            f"meilleure moyenne cumulée={float(jnp.mean(best_losses)):.4e}"
        )

    counts = np.bincount(np.asarray(best_start_indices), minlength=n_starts)
    print("\n=== Sélection des offsets relatifs ===")
    for idx, count in enumerate(counts):
        d = offsets_physical[idx]
        print(
            f"Offset {idx + 1}: ({d[0]:+.3f}, {d[1]:+.3f}, {d[2]:+.3f}) "
            f"-> {int(count)}/{n_signals} signaux"
        )
    print(f"Loss moyenne après sélection relative : {float(jnp.mean(best_losses)):.4e}")
    return best_theta, best_losses, best_start_indices, best_start_values


def optimize_hard_signals(
    *,
    theta,
    signal_losses,
    threshold,
    relative_offsets,
    parameter_mins,
    parameter_maxs,
    data_init,
    geometry,
    c,
    targets,
    scales,
    qr_values,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    train_mask,
    multistart_lr,
    multistart_iter,
    multistart_final_lr_factor,
    multistart_print_every,
    refine_lr,
    refine_iter,
    refine_final_lr_factor,
    refine_print_every,
):
    """
    Relance uniquement les signaux dont la loss individuelle dépasse threshold.

    Étapes :
      1. sélection des indices difficiles ;
      2. second multi-start sur ce sous-ensemble ;
      3. raffinement Adam faible LR sur ce même sous-ensemble ;
      4. réinjection des paramètres optimisés dans le batch complet.
    """
    signal_losses_np = np.asarray(signal_losses)
    hard_indices_np = np.flatnonzero(signal_losses_np > threshold)
    n_total = int(theta.shape[0])
    n_hard = int(hard_indices_np.size)

    hard_selected_indices_full = np.full(n_total, -1, dtype=np.int32)
    hard_selected_starts_full = np.full((n_total, len(PARAM_NAMES)), np.nan, dtype=float)
    loss_before_full = signal_losses_np.copy()
    loss_after_full = signal_losses_np.copy()
    hard_mask_full = np.zeros(n_total, dtype=bool)

    if n_hard == 0:
        print(
            f"\n=== Second multi-start ciblé : aucun signal avec "
            f"loss > {threshold:.4g} ==="
        )
        return (
            theta,
            signal_losses,
            hard_mask_full,
            hard_selected_indices_full,
            hard_selected_starts_full,
            loss_before_full,
            loss_after_full,
        )

    if multistart_iter <= 0:
        print(
            f"\n=== Second multi-start ciblé ignoré : "
            f"{n_hard} signaux ont loss > {threshold:.4g}, "
            "mais --hard_multistart_iter=0 ==="
        )
        return (
            theta,
            signal_losses,
            hard_mask_full,
            hard_selected_indices_full,
            hard_selected_starts_full,
            loss_before_full,
            loss_after_full,
        )

    if relative_offsets.shape[0] == 0:
        raise ValueError("Le second multi-start ciblé nécessite au moins un offset.")

    hard_mask_full[hard_indices_np] = True
    hard_indices = jnp.asarray(hard_indices_np, dtype=jnp.int32)

    theta_hard_current = theta[hard_indices]
    targets_hard = targets[hard_indices]
    scales_hard = scales[hard_indices]
    qr_hard = qr_values[hard_indices]

    loss_hard = make_batched_loss(
        data_init,
        geometry,
        c,
        targets_hard,
        scales_hard,
        qr_hard,
        solve_kwargs,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )
    losses_hard_per_signal = make_losses_per_signal(
        data_init,
        geometry,
        c,
        targets_hard,
        scales_hard,
        qr_hard,
        solve_kwargs,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )

    print(
        f"\n=== Second multi-start ciblé : {n_hard}/{n_total} signaux "
        f"avec loss > {threshold:.4g} ==="
    )
    print(
        f"{relative_offsets.shape[0]} offsets relatifs, "
        f"{multistart_iter} itérations Adam par offset"
    )

    centers_physical = theta_hard_current * scales_hard
    theta_hard, _, selected_hard, selected_start_values = optimize_relative_multistart(
        centers_physical=centers_physical,
        offsets_physical=relative_offsets,
        parameter_mins=parameter_mins,
        parameter_maxs=parameter_maxs,
        scales=scales_hard,
        loss_fn=loss_hard,
        losses_per_signal_fn=losses_hard_per_signal,
        train_mask=train_mask,
        lr=multistart_lr,
        n_iter=multistart_iter,
        final_lr_factor=multistart_final_lr_factor,
        print_every=multistart_print_every,
    )

    if refine_iter > 0:
        print(
            f"\n--- Raffinement ciblé faible LR sur {n_hard} signaux ---"
        )
        theta_hard, _ = optimize_stage(
            theta_hard,
            loss_hard,
            train_mask,
            refine_lr,
            refine_iter,
            refine_print_every,
            refine_final_lr_factor,
        )
    else:
        print("\n--- Raffinement ciblé faible LR ignoré ---")

    hard_losses_after = losses_hard_per_signal(theta_hard)
    hard_losses_after.block_until_ready()

    # Garde la solution antérieure si le second passage n'améliore pas un signal.
    current_hard_losses = jnp.asarray(signal_losses)[hard_indices]
    improved = hard_losses_after < current_hard_losses
    theta_hard_kept = jnp.where(
        improved[:, None],
        theta_hard,
        theta_hard_current,
    )
    kept_losses = jnp.where(
        improved,
        hard_losses_after,
        current_hard_losses,
    )

    theta = theta.at[hard_indices].set(theta_hard_kept)
    updated_losses = jnp.asarray(signal_losses).at[hard_indices].set(kept_losses)

    selected_hard_np = np.asarray(selected_hard)
    improved_np = np.asarray(improved)
    hard_selected_indices_full[hard_indices_np] = np.where(
        improved_np, selected_hard_np, -1,
    )
    selected_start_values_np = np.asarray(selected_start_values)
    hard_selected_starts_full[hard_indices_np] = np.where(
        improved_np[:, None], selected_start_values_np, np.nan,
    )
    loss_after_full[hard_indices_np] = np.asarray(kept_losses)

    print(
        f"Signaux améliorés par le second passage : "
        f"{int(np.sum(improved_np))}/{n_hard}"
    )
    print(
        "Loss moyenne des signaux difficiles : "
        f"{float(np.mean(signal_losses_np[hard_indices_np])):.4e} -> "
        f"{float(np.mean(np.asarray(kept_losses))):.4e}"
    )

    return (
        theta,
        updated_losses,
        hard_mask_full,
        hard_selected_indices_full,
        hard_selected_starts_full,
        loss_before_full,
        loss_after_full,
    )



def make_gauss_newton_batch_step(
    data_init, geometry, c, solve_kwargs,
    stft_resolutions, stft_dynamic_db, stft_allow_padding,
    train_mask_one, step_size, ridge, fd_eps,
):
    """Construit une itération GN vectorisée sur un mini-batch."""
    def residual_one(theta_one, scale_one, qr_one, target_one):
        values = theta_one * scale_one
        data = set_parameter_vector_data(data_init, values, qr_one)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        pred = pred[:target_one.shape[0]]
        return multi_resolution_spectral_residuals(
            pred, target_one,
            resolutions=stft_resolutions,
            dynamic_db=stft_dynamic_db,
            allow_padding=stft_allow_padding,
            include_linear=True,
            include_db=True,
        )

    def finite_difference_jacobian(theta_one, scale_one, qr_one, target_one):
        """Jacobien spectral par differences finies centrales."""
        eye = jnp.eye(theta_one.shape[0], dtype=theta_one.dtype)

        def one_column(direction):
            active_direction = direction * train_mask_one
            theta_plus = jnp.maximum(
                theta_one + fd_eps * active_direction,
                1e-8,
            )
            theta_minus = jnp.maximum(
                theta_one - fd_eps * active_direction,
                1e-8,
            )
            residual_plus = residual_one(
                theta_plus, scale_one, qr_one, target_one
            )
            residual_minus = residual_one(
                theta_minus, scale_one, qr_one, target_one
            )
            column = (residual_plus - residual_minus) / (2.0 * fd_eps)
            active = jnp.sum(jnp.abs(active_direction)) > 0.0
            return jnp.where(active, column, jnp.zeros_like(column))

        return jax.vmap(one_column)(eye).T

    def step_one(theta_one, scale_one, qr_one, target_one):
        residual = residual_one(theta_one, scale_one, qr_one, target_one)
        jacobian = finite_difference_jacobian(
            theta_one, scale_one, qr_one, target_one
        )
        normal_matrix = jacobian.T @ jacobian
        normal_rhs = jacobian.T @ residual
        n_parameters = theta_one.shape[0]
        matrix_scale = jnp.maximum(jnp.trace(normal_matrix) / n_parameters, 1e-12)
        A = normal_matrix + ridge * matrix_scale * jnp.eye(n_parameters, dtype=theta_one.dtype)
        delta = jnp.linalg.solve(A, -normal_rhs) * train_mask_one
        candidate = jnp.maximum(theta_one + step_size * delta, 1e-8)
        old_obj = 0.5 * jnp.sum(residual**2)
        candidate_residual = residual_one(candidate, scale_one, qr_one, target_one)
        new_obj = 0.5 * jnp.sum(candidate_residual**2)
        accepted = jnp.isfinite(new_obj) & (new_obj < old_obj)
        theta_new = jnp.where(accepted, candidate, theta_one)
        effective_obj = jnp.where(accepted, new_obj, old_obj)
        return theta_new, old_obj, effective_obj, accepted, delta

    return jax.jit(jax.vmap(step_one, in_axes=(0, 0, 0, 0)))


def optimize_stage_gauss_newton(
    theta, data_init, geometry, c, targets, scales, qr_values,
    solve_kwargs, stft_resolutions, stft_dynamic_db,
    stft_allow_padding, train_mask, n_iter, step_size,
    ridge, fd_eps, batch_size, print_every,
):
    """Raffinement GN spectral par mini-batches indépendants."""
    if n_iter <= 0:
        return theta, np.nan
    if batch_size <= 0:
        raise ValueError("--gauss_newton_batch_size doit etre strictement positif.")

    n_signals = int(theta.shape[0])
    train_mask_one = train_mask[0]
    step_cache = {}
    final_objective = np.nan

    for iteration in range(n_iter):
        start = time.time()
        theta_chunks, old_chunks, new_chunks = [], [], []
        accepted_chunks, delta_chunks = [], []
        for begin in range(0, n_signals, batch_size):
            end = min(begin + batch_size, n_signals)
            size = end - begin
            if size not in step_cache:
                step_cache[size] = make_gauss_newton_batch_step(
                    data_init, geometry, c, solve_kwargs,
                    stft_resolutions, stft_dynamic_db, stft_allow_padding,
                    train_mask_one, step_size, ridge, fd_eps,
                )
            out = step_cache[size](
                theta[begin:end], scales[begin:end],
                qr_values[begin:end], targets[begin:end],
            )
            th, old, new, acc, delta = out
            theta_chunks.append(th)
            old_chunks.append(old)
            new_chunks.append(new)
            accepted_chunks.append(acc)
            delta_chunks.append(jnp.linalg.norm(delta, axis=1))

        theta = jnp.concatenate(theta_chunks, axis=0)
        old_obj = jnp.concatenate(old_chunks)
        new_obj = jnp.concatenate(new_chunks)
        accepted = jnp.concatenate(accepted_chunks)
        delta_norm = jnp.concatenate(delta_chunks)
        theta.block_until_ready()
        final_objective = float(jnp.mean(new_obj))

        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            print(
                f"GN iter {iteration:3d} | objectif spectral="
                f"{float(jnp.mean(old_obj)):.4e}->{final_objective:.4e} | "
                f"accept={100.0*float(jnp.mean(accepted.astype(jnp.float64))):.1f}% | "
                f"|delta|={float(jnp.mean(delta_norm)):.3e} | "
                f"t={time.time()-start:.2f}s"
            )
    return theta, final_objective


def optimize_stage_lbfgs(
    theta,
    loss_fn,
    train_mask,
    n_iter,
    memory_size,
    tolerance,
    max_linesearch_steps,
    print_every,
):
    """
    Raffinement L-BFGS de la loss scalaire sur l'ensemble du batch.

    Les variables optimisees sont logarithmiques : theta = exp(eta). Cette
    parametrisation garantit la positivite des parametres. Les composantes
    absentes de train_mask restent exactement fixees a leur valeur d'entree.
    """
    if n_iter <= 0:
        final_loss = float(loss_fn(theta))
        return theta, final_loss

    mask = jnp.broadcast_to(train_mask, theta.shape)
    theta_fixed = jax.lax.stop_gradient(theta)
    eta = jnp.log(jnp.maximum(theta, 1e-8))

    def theta_from_eta(eta_value):
        theta_positive = jnp.exp(eta_value)
        return mask * theta_positive + (1.0 - mask) * theta_fixed

    def transformed_loss(eta_value):
        return loss_fn(theta_from_eta(eta_value))

    linesearch = optax.scale_by_zoom_linesearch(
        max_linesearch_steps=max_linesearch_steps,
        initial_guess_strategy="one",
    )
    optimizer = optax.lbfgs(
        memory_size=memory_size,
        linesearch=linesearch,
    )
    opt_state = optimizer.init(eta)

    # Reutilise la valeur et le gradient stockes par la recherche lineaire
    # lorsque l'etat Optax le permet, afin d'eviter des evaluations inutiles.
    value_and_grad = optax.value_and_grad_from_state(transformed_loss)

    final_loss = float(transformed_loss(eta))
    previous_loss = final_loss

    for iteration in range(n_iter):
        start = time.time()

        loss_value, gradient = value_and_grad(eta, state=opt_state)
        gradient = gradient * mask
        gradient_norm = jnp.linalg.norm(gradient)

        updates, opt_state = optimizer.update(
            gradient,
            opt_state,
            eta,
            value=loss_value,
            grad=gradient,
            value_fn=transformed_loss,
        )
        updates = updates * mask
        eta_candidate = optax.apply_updates(eta, updates)

        # La recherche lineaire de L-BFGS doit normalement produire une
        # diminution. Cette verification protege toutefois contre une valeur
        # non finie ou une hausse accidentelle de la loss.
        candidate_loss = transformed_loss(eta_candidate)
        candidate_loss.block_until_ready()

        candidate_loss_value = float(candidate_loss)
        old_loss_value = float(loss_value)
        grad_norm_value = float(gradient_norm)
        accepted = (
            np.isfinite(candidate_loss_value)
            and candidate_loss_value <= old_loss_value
        )

        if accepted:
            eta = eta_candidate
            final_loss = candidate_loss_value
        else:
            final_loss = old_loss_value
            print(
                "Arret L-BFGS : le pas propose n'ameliore pas la loss "
                "ou produit une valeur non finie."
            )

        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            print(
                f"LBFGS iter {iteration:3d} | "
                f"loss={old_loss_value:.4e}->{final_loss:.4e} | "
                f"|grad|={grad_norm_value:.3e} | "
                f"accept={'oui' if accepted else 'non'} | "
                f"t={time.time() - start:.2f}s"
            )

        if not accepted:
            break

        if grad_norm_value < tolerance:
            print(
                f"Arret L-BFGS : norme du gradient "
                f"{grad_norm_value:.3e} < {tolerance:.3e}."
            )
            break

        relative_improvement = abs(previous_loss - final_loss) / max(
            abs(previous_loss), 1e-12
        )
        previous_loss = final_loss
        if relative_improvement < 1e-8:
            print(
                "Arret L-BFGS : amelioration relative inferieure a 1e-8."
            )
            break

    theta = theta_from_eta(eta)
    theta.block_until_ready()
    return theta, final_loss


def relative_errors(estimated, true_values):
    denominator = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    errors = jnp.abs((estimated - true_values) / denominator)
    return errors, float(jnp.linalg.norm(errors))


def make_trained_predictions(
    data_init,
    geometry,
    c,
    estimated_values,
    qr_values,
    solve_kwargs,
):
    def one_prediction(params, qr_value):
        data = set_parameter_vector_data(
            data_init,
            params,
            qr_value,
        )
        return forward_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )

    return jax.jit(
        jax.vmap(one_prediction, in_axes=(0, 0))
    )(estimated_values, qr_values)


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
        "mean_relerr_kappa",
        "std_relerr_kappa",
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
        "multistart_index",
        "multistart_init_gamma",
        "multistart_init_kappa",
        "multistart_init_zeta",
        "hard_refined",
        "loss_before_hard_refinement",
        "loss_after_hard_refinement",
        "hard_multistart_index",
        "hard_multistart_init_gamma",
        "hard_multistart_init_kappa",
        "hard_multistart_init_zeta",
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
        tick_labels=[
            r"$\gamma$",
            r"$\kappa$",
            r"$\zeta$",
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
        r"$\kappa$",
        r"$\zeta$",
    )

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4))

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
    targets_short,
    targets_long,
    generation_T,
    calibration_T,
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

    all_true_values = jnp.asarray(
        true_values_np,
        dtype=jnp.float64,
    )
    if all_true_values.shape[1] < 4:
        raise ValueError(
            "Le dataset doit contenir "
            "[gamma, kappa, zeta, Qr]."
        )

    # Les vraies valeurs servent uniquement aux métriques finales.
    true_values = all_true_values[:, :3]

    # Qr est connu et fixé signal par signal.
    known_qr = all_true_values[:, 3]

    # Échelles fixes, indépendantes des vraies valeurs.
    scales_one = jnp.asarray(
        [args.scale_gamma, args.scale_kappa, args.scale_zeta],
        dtype=jnp.float64,
    )
    scales = jnp.broadcast_to(
        scales_one[None, :],
        (all_true_values.shape[0], len(PARAM_NAMES)),
    )

    # Initialisation physique constante, identique pour tous les signaux.
    init_values_one = jnp.asarray(
        [args.init_gamma, args.init_kappa, args.init_zeta],
        dtype=jnp.float64,
    )
    theta_one = init_values_one / scales_one
    theta = jnp.broadcast_to(
        theta_one[None, :],
        (all_true_values.shape[0], len(PARAM_NAMES)),
    ).copy()

    train_mask = jnp.asarray(
        [name in args.train_params for name in PARAM_NAMES],
        dtype=jnp.float64,
    )[None, :]

    loss_long = make_batched_loss(
        data_ref,
        geometry,
        c,
        targets_long,
        scales,
        known_qr,
        solve_kwargs_long,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )
    loss_short = make_batched_loss(
        data_ref,
        geometry,
        c,
        targets_short,
        scales,
        known_qr,
        solve_kwargs_short,
        stft_resolutions,
        stft_dynamic_db,
        stft_allow_padding,
    )

    losses_long_per_signal = make_losses_per_signal(
        data_ref, geometry, c, targets_long, scales, known_qr,
        solve_kwargs_long, stft_resolutions, stft_dynamic_db,
        stft_allow_padding,
    )

    multistart_values = parse_multistart_inits(args.multistart_inits)
    if args.multistart_iter > 0 and multistart_values.shape[0] > 0:
        print(
            f"\n=== Phase multi-start : {multistart_values.shape[0]} départs, "
            f"{args.multistart_iter} itérations Adam par départ ==="
        )
        theta, multistart_losses, multistart_indices = optimize_multistart(
            starts_physical=multistart_values,
            scales=scales,
            loss_fn=loss_long,
            losses_per_signal_fn=losses_long_per_signal,
            train_mask=train_mask,
            lr=args.multistart_lr,
            n_iter=args.multistart_iter,
            final_lr_factor=args.multistart_lr_final_factor,
            print_every=args.multistart_print_every,
        )
    else:
        multistart_losses = jnp.full(
            (all_true_values.shape[0],), jnp.nan, dtype=jnp.float64
        )
        multistart_indices = jnp.full(
            (all_true_values.shape[0],), -1, dtype=jnp.int32
        )

    print(
        f"\n--- Stage 1 : Adam conjoint gamma-kappa-zeta | "
        f"T={calibration_T:.4f} s (fenetre longue) ---"
    )
    theta, loss_stage1 = optimize_stage(
        theta,
        loss_long,
        train_mask,
        args.stage1_lr,
        args.stage1_iter,
        args.print_every,
        args.stage1_lr_final_factor,
    )

    # Évaluation individuelle après le Stage 1, puis second multi-start
    # uniquement sur les signaux encore mal calibrés.
    signal_losses_after_stage1 = losses_long_per_signal(theta)
    signal_losses_after_stage1.block_until_ready()

    hard_relative_offsets = parse_triplets(
        args.hard_relative_offsets, allow_negative=True
    )
    hard_parameter_mins = parse_single_triplet(
        args.hard_param_mins, "--hard_param_mins"
    )
    hard_parameter_maxs = parse_single_triplet(
        args.hard_param_maxs, "--hard_param_maxs"
    )

    (
        theta,
        signal_losses_after_hard,
        hard_refined_mask,
        hard_multistart_indices,
        hard_multistart_start_values,
        loss_before_hard_refinement,
        loss_after_hard_refinement,
    ) = optimize_hard_signals(
        theta=theta,
        signal_losses=signal_losses_after_stage1,
        threshold=args.hard_loss_threshold,
        relative_offsets=hard_relative_offsets,
        parameter_mins=hard_parameter_mins,
        parameter_maxs=hard_parameter_maxs,
        data_init=data_ref,
        geometry=geometry,
        c=c,
        targets=targets_long,
        scales=scales,
        qr_values=known_qr,
        solve_kwargs=solve_kwargs_long,
        stft_resolutions=stft_resolutions,
        stft_dynamic_db=stft_dynamic_db,
        stft_allow_padding=stft_allow_padding,
        train_mask=train_mask,
        multistart_lr=args.hard_multistart_lr,
        multistart_iter=args.hard_multistart_iter,
        multistart_final_lr_factor=args.hard_multistart_lr_final_factor,
        multistart_print_every=args.hard_multistart_print_every,
        refine_lr=args.hard_refine_lr,
        refine_iter=args.hard_refine_iter,
        refine_final_lr_factor=args.hard_refine_lr_final_factor,
        refine_print_every=args.print_every,
    )

    if args.skip_stage2 or args.stage2_iter <= 0:
        loss_stage2 = float(loss_short(theta))
        print("\n--- Stage 2 conjoint court ignore ---")
    else:
        print(
            "\n--- Stage 2 : Adam conjoint gamma-kappa-zeta "
            "| T=0.01 s (fenetre courte) ---"
        )
        theta, loss_stage2 = optimize_stage(
            theta,
            loss_short,
            train_mask,
            args.stage2_lr,
            args.stage2_iter,
            args.print_every,
            args.stage2_lr_final_factor,
        )

    if (
        args.skip_final_joint
        or args.final_joint_iter <= 0
    ):
        loss_final_joint = float(loss_long(theta))
        print("\n--- Stage 3 conjoint final ignore ---")
    else:
        print(
            f"\n--- Stage 3 : Adam conjoint faible LR | "
            f"T={calibration_T:.4f} s ---"
        )
        theta, loss_final_joint = optimize_stage(
            theta,
            loss_long,
            train_mask,
            args.final_joint_lr,
            args.final_joint_iter,
            args.print_every,
            args.final_joint_lr_final_factor,
        )

    if args.skip_gauss_newton or args.gauss_newton_iter <= 0:
        loss_gauss_newton = np.nan
        print("\n--- Raffinement Gauss-Newton ignore ---")
    else:
        print(f"\n--- Raffinement Gauss-Newton spectral | T={calibration_T:.4f} s ---")
        theta, loss_gauss_newton = optimize_stage_gauss_newton(
            theta=theta, data_init=data_ref, geometry=geometry, c=c,
            targets=targets_long, scales=scales, qr_values=known_qr,
            solve_kwargs=solve_kwargs_long, stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db, stft_allow_padding=stft_allow_padding,
            train_mask=train_mask, n_iter=args.gauss_newton_iter,
            step_size=args.gauss_newton_step, ridge=args.gauss_newton_ridge,
            batch_size=args.gauss_newton_batch_size,
            print_every=args.gauss_newton_print_every,
            fd_eps=args.gauss_newton_fd_eps
        )

    if args.skip_lbfgs or args.lbfgs_iter <= 0:
        loss_lbfgs = np.nan
        print("\n--- Raffinement L-BFGS ignore ---")
    else:
        print(
            f"\n--- Raffinement L-BFGS conjoint gamma-kappa-zeta | "
            f"T={calibration_T:.4f} s ---"
        )
        theta, loss_lbfgs = optimize_stage_lbfgs(
            theta=theta,
            loss_fn=loss_long,
            train_mask=train_mask,
            n_iter=args.lbfgs_iter,
            memory_size=args.lbfgs_memory,
            tolerance=args.lbfgs_tol,
            max_linesearch_steps=args.lbfgs_max_linesearch_steps,
            print_every=args.lbfgs_print_every,
        )

    estimated = theta * scales
    errors, global_error = relative_errors(
        estimated,
        true_values,
    )

    signal_losses = np.asarray(losses_long_per_signal(theta))
    final_loss = float(np.mean(signal_losses))
    elapsed = time.time() - start

    true_np = np.asarray(true_values)
    known_qr_np = np.asarray(known_qr)
    estimated_np = np.asarray(estimated)
    errors_np = np.asarray(errors)
    batch_size = true_np.shape[0]
    multistart_indices_np = np.asarray(multistart_indices)
    hard_refined_mask_np = np.asarray(hard_refined_mask)
    hard_multistart_indices_np = np.asarray(hard_multistart_indices)
    loss_before_hard_np = np.asarray(loss_before_hard_refinement)
    loss_after_hard_np = np.asarray(loss_after_hard_refinement)
    hard_multistart_start_values_np = np.asarray(hard_multistart_start_values)

    signal_rows = []
    for i in range(batch_size):
        signal_rows.append(
            {
                "repeat_idx": repeat_idx,
                "seed": seed,
                "signal_idx": i,
                "true_gamma": float(true_np[i, 0]),
                "true_kappa": float(true_np[i, 1]),
                "true_zeta": float(true_np[i, 2]),
                "known_Qr": float(known_qr_np[i]),
                "estimated_gamma": float(estimated_np[i, 0]),
                "estimated_kappa": float(estimated_np[i, 1]),
                "estimated_zeta": float(estimated_np[i, 2]),
                "relerr_gamma": float(errors_np[i, 0]),
                "relerr_kappa": float(errors_np[i, 1]),
                "relerr_zeta": float(errors_np[i, 2]),
                "signal_loss": float(signal_losses[i]),
                "multistart_index": int(multistart_indices_np[i]),
                "multistart_init_gamma": (
                    float(multistart_values[multistart_indices_np[i], 0])
                    if multistart_indices_np[i] >= 0 else np.nan
                ),
                "multistart_init_kappa": (
                    float(multistart_values[multistart_indices_np[i], 1])
                    if multistart_indices_np[i] >= 0 else np.nan
                ),
                "multistart_init_zeta": (
                    float(multistart_values[multistart_indices_np[i], 2])
                    if multistart_indices_np[i] >= 0 else np.nan
                ),
                "hard_refined": bool(hard_refined_mask_np[i]),
                "loss_before_hard_refinement": float(loss_before_hard_np[i]),
                "loss_after_hard_refinement": float(loss_after_hard_np[i]),
                "hard_multistart_index": int(hard_multistart_indices_np[i]),
                "hard_multistart_init_gamma": (
                    float(hard_multistart_start_values_np[i, 0])
                    if hard_multistart_indices_np[i] >= 0 else np.nan
                ),
                "hard_multistart_init_kappa": (
                    float(hard_multistart_start_values_np[i, 1])
                    if hard_multistart_indices_np[i] >= 0 else np.nan
                ),
                "hard_multistart_init_zeta": (
                    float(hard_multistart_start_values_np[i, 2])
                    if hard_multistart_indices_np[i] >= 0 else np.nan
                ),
                "elapsed_seconds_repeat": elapsed,
            }
        )

    repeat_row = {
        "repeat_idx": repeat_idx,
        "seed": seed,
        "n_signals": batch_size,
        "mean_relerr_gamma": float(np.mean(errors_np[:, 0])),
        "std_relerr_gamma": float(np.std(errors_np[:, 0])),
        "mean_relerr_kappa": float(np.mean(errors_np[:, 1])),
        "std_relerr_kappa": float(np.std(errors_np[:, 1])),
        "mean_relerr_zeta": float(np.mean(errors_np[:, 2])),
        "std_relerr_zeta": float(np.std(errors_np[:, 2])),
        "global_relerr": global_error,
        "final_loss": final_loss,
        "elapsed_seconds": elapsed,
    }

    run_dir = output_dir / (
        f"repeat_{repeat_idx:03d}_seed_{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        run_dir / "signals.csv",
        signal_rows,
        signal_fieldnames(),
    )

    if args.save_detailed_plots:
        predictions = make_trained_predictions(
            data_ref,
            geometry,
            c,
            estimated,
            known_qr,
            solve_kwargs_long,
        )[:, : targets_long.shape[1]]
        plot_signal_comparison(
            run_dir / "signals_openwind_vs_dg.png",
            np.asarray(snapshot_times_long),
            np.asarray(targets_long),
            np.asarray(predictions),
        )

    print(
        f"\nResume tirage {repeat_idx + 1}: "
        f"gamma={100 * repeat_row['mean_relerr_gamma']:.3f}% | "
        f"kappa={100 * repeat_row['mean_relerr_kappa']:.3f}% | "
        f"zeta={100 * repeat_row['mean_relerr_zeta']:.3f}% | "
        f"Qr=connu/fixe | "
        f"loss={final_loss:.3e} | "
        f"temps={elapsed / 60:.2f} min"
    )
    print(
        f"Loss stage 1 conjoint long={loss_stage1:.3e}, "
        f"stage 2 conjoint court={loss_stage2:.3e}, "
        f"stage 3 conjoint={loss_final_joint:.3e}, "
        f"Gauss-Newton={loss_gauss_newton:.3e}, "
        f"L-BFGS={loss_lbfgs:.3e}"
    )
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
    for name, value in (
        ("stage1_iter", args.stage1_iter),
        ("stage2_iter", args.stage2_iter),
        ("final_joint_iter", args.final_joint_iter),
    ):
        if value < 0:
            raise ValueError(
                f"--{name} doit etre positif ou nul."
            )
    if args.multistart_iter < 0:
        raise ValueError("--multistart_iter doit être positif ou nul.")
    if args.multistart_lr <= 0.0:
        raise ValueError("--multistart_lr doit être strictement positif.")
    if not 0.0 <= args.multistart_lr_final_factor <= 1.0:
        raise ValueError(
            "--multistart_lr_final_factor doit être compris entre 0 et 1."
        )
    if args.multistart_print_every <= 0:
        raise ValueError(
            "--multistart_print_every doit être strictement positif."
        )
    multistart_values_check = parse_multistart_inits(args.multistart_inits)
    if args.multistart_iter > 0 and multistart_values_check.shape[0] == 0:
        raise ValueError(
            "--multistart_iter est positif mais --multistart_inits est vide."
        )

    if args.hard_loss_threshold < 0.0:
        raise ValueError("--hard_loss_threshold doit être positif ou nul.")
    if args.hard_multistart_iter < 0:
        raise ValueError("--hard_multistart_iter doit être positif ou nul.")
    if args.hard_multistart_lr <= 0.0:
        raise ValueError("--hard_multistart_lr doit être strictement positif.")
    if not 0.0 <= args.hard_multistart_lr_final_factor <= 1.0:
        raise ValueError(
            "--hard_multistart_lr_final_factor doit être compris entre 0 et 1."
        )
    if args.hard_multistart_print_every <= 0:
        raise ValueError(
            "--hard_multistart_print_every doit être strictement positif."
        )
    if args.hard_refine_iter < 0:
        raise ValueError("--hard_refine_iter doit être positif ou nul.")
    if args.hard_refine_lr <= 0.0:
        raise ValueError("--hard_refine_lr doit être strictement positif.")
    if not 0.0 <= args.hard_refine_lr_final_factor <= 1.0:
        raise ValueError(
            "--hard_refine_lr_final_factor doit être compris entre 0 et 1."
        )

    hard_offsets_check = parse_triplets(
        args.hard_relative_offsets, allow_negative=True
    )
    hard_mins_check = parse_single_triplet(args.hard_param_mins, "--hard_param_mins")
    hard_maxs_check = parse_single_triplet(args.hard_param_maxs, "--hard_param_maxs")
    if args.hard_multistart_iter > 0 and hard_offsets_check.shape[0] == 0:
        raise ValueError("Le second multi-start est actif mais aucun offset relatif n'est défini.")
    if np.any(hard_mins_check >= hard_maxs_check):
        raise ValueError("Chaque borne minimale doit être inférieure à la borne maximale.")

    if args.gauss_newton_iter < 0:
        raise ValueError("--gauss_newton_iter doit etre positif ou nul.")
    if args.gauss_newton_step <= 0.0:
        raise ValueError("--gauss_newton_step doit etre strictement positif.")
    if args.gauss_newton_ridge < 0.0:
        raise ValueError("--gauss_newton_ridge doit etre positif ou nul.")
    if args.gauss_newton_fd_eps <= 0.0:
        raise ValueError(
            "--gauss_newton_fd_eps doit etre strictement positif."
        )
    if args.gauss_newton_batch_size <= 0:
        raise ValueError("--gauss_newton_batch_size doit etre strictement positif.")

    if args.lbfgs_iter < 0:
        raise ValueError("--lbfgs_iter doit etre positif ou nul.")
    if args.lbfgs_memory <= 0:
        raise ValueError("--lbfgs_memory doit etre strictement positif.")
    if args.lbfgs_tol < 0.0:
        raise ValueError("--lbfgs_tol doit etre positif ou nul.")
    if args.lbfgs_max_linesearch_steps <= 0:
        raise ValueError(
            "--lbfgs_max_linesearch_steps doit etre strictement positif."
        )
    if args.random_factor_min <= 0 or args.random_factor_min >= args.random_factor_max:
        raise ValueError("Bornes aleatoires invalides.")

    for name, value in (
        ("init_gamma", args.init_gamma),
        ("init_kappa", args.init_kappa),
        ("init_zeta", args.init_zeta),
        ("scale_gamma", args.scale_gamma),
        ("scale_kappa", args.scale_kappa),
        ("scale_zeta", args.scale_zeta),
    ):
        if value <= 0.0:
            raise ValueError(f"--{name} doit être strictement positif.")
    for name, value in (
        ("stage1_lr", args.stage1_lr),
        ("stage2_lr", args.stage2_lr),
        ("final_joint_lr", args.final_joint_lr),
    ):
        if value <= 0.0:
            raise ValueError(
                f"--{name} doit etre strictement positif."
            )
    for name, factor in (
        ("stage1_lr_final_factor", args.stage1_lr_final_factor),
        ("stage2_lr_final_factor", args.stage2_lr_final_factor),
        (
            "final_joint_lr_final_factor",
            args.final_joint_lr_final_factor,
        ),
    ):
        if not 0.0 <= factor <= 1.0:
            raise ValueError(
                f"--{name} doit etre compris entre 0 et 1."
            )

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
        # N_snapshot est lu dans le dataset.
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

    dataset_n_snapshot = int(dataset_metadata["N_snapshot"])
    pressure_long_count = int(dataset_arrays["pressure_long"].shape[-1])
    pressure_short_count = int(dataset_arrays["pressure_short"].shape[-1])

    if pressure_long_count != dataset_n_snapshot:
        raise ValueError(
            "Incohérence dans le cache OpenWind : "
            f"metadata N_snapshot={dataset_n_snapshot}, "
            f"pressure_long contient {pressure_long_count} snapshots."
        )

    print(
        "Snapshots utilisés pour la calibration : "
        f"{dataset_n_snapshot} (valeur lue dans le dataset)"
    )

    long_keep = int(np.searchsorted(
        dataset_arrays["times_long"],
        calibration_T,
        side="right",
    ))
    long_keep = max(1, long_keep)

    length = data_ref.section.L_tube + data_ref.section.L_bell
    solve_short = make_solver_data(
        0.01,
        train_params["cfl"],
        train_params["Nx"],
        pressure_short_count,
        length, c, bc, phi0, y0, z0,
    )
    solve_long = make_solver_data(
        train_params["T_max"],
        train_params["cfl"],
        train_params["Nx"],
        dataset_n_snapshot,
        length, c, bc, phi0, y0, z0,
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

    total_signals = int(dataset_metadata["simulation_count"])

    print("\n" + "=" * 72)
    print("CALIBRATION STATISTIQUE")
    print("=" * 72)
    print(
        "Organisation du cache : "
        f"{dataset_metadata['n_repeats']} x {dataset_metadata['n_signals']}"
    )
    print(f"Nombre total de signaux : {total_signals}")
    print("\nParametres calibres :")
    for name in args.train_params:
        print(f"  - {name}")
    print("\nParametre connu et fixe :")
    print("  - Qr, avec sa valeur OpenWind propre a chaque signal")

    print("\nInitialisation constante :")
    print(f"  - gamma = {args.init_gamma:.6g}")
    print(f"  - kappa = {args.init_kappa:.6g}")
    print(f"  - zeta  = {args.init_zeta:.6g}")

    print("\nEchelles fixes :")
    print(f"  - gamma = {args.scale_gamma:.6g}")
    print(f"  - kappa = {args.scale_kappa:.6g}")
    print(f"  - zeta  = {args.scale_zeta:.6g}")

    parsed_multistart = parse_multistart_inits(args.multistart_inits)
    print("\nMulti-start :")
    if args.multistart_iter > 0 and parsed_multistart.shape[0] > 0:
        print(f"  - {parsed_multistart.shape[0]} départs")
        print(f"  - {args.multistart_iter} itérations Adam par départ")
        print(
            f"  - LR={args.multistart_lr:.3e} -> "
            f"{args.multistart_lr * args.multistart_lr_final_factor:.3e}"
        )
        for idx, values in enumerate(parsed_multistart, start=1):
            print(
                f"  - départ {idx}: gamma={values[0]:.4g}, "
                f"kappa={values[1]:.4g}, zeta={values[2]:.4g}"
            )
    else:
        print("  - désactivé")

    parsed_hard_offsets = parse_triplets(
        args.hard_relative_offsets, allow_negative=True
    )
    parsed_hard_mins = parse_single_triplet(args.hard_param_mins, "--hard_param_mins")
    parsed_hard_maxs = parse_single_triplet(args.hard_param_maxs, "--hard_param_maxs")
    print("\nSecond multi-start ciblé relatif :")
    if args.hard_multistart_iter > 0 and parsed_hard_offsets.shape[0] > 0:
        print(f"  - seuil de loss : {args.hard_loss_threshold:.4g}")
        print(f"  - {parsed_hard_offsets.shape[0]} offsets autour de chaque estimation")
        print(f"  - {args.hard_multistart_iter} itérations Adam par offset")
        print(
            f"  - LR={args.hard_multistart_lr:.3e} -> "
            f"{args.hard_multistart_lr * args.hard_multistart_lr_final_factor:.3e}"
        )
        print(
            "  - bornes : "
            f"gamma=[{parsed_hard_mins[0]:.3g},{parsed_hard_maxs[0]:.3g}], "
            f"kappa=[{parsed_hard_mins[1]:.3g},{parsed_hard_maxs[1]:.3g}], "
            f"zeta=[{parsed_hard_mins[2]:.3g},{parsed_hard_maxs[2]:.3g}]"
        )
        for idx, values in enumerate(parsed_hard_offsets, start=1):
            print(
                f"  - offset {idx}: dgamma={values[0]:+.3g}, "
                f"dkappa={values[1]:+.3g}, dzeta={values[2]:+.3g}"
            )
    else:
        print("  - désactivé")

    print("\nObservation :")
    print("  - pression au pavillon p(L,t)")
    print("\nProtocole par defaut :")
    print(
        f"  1. Multi-start initial puis Adam conjoint sur la fenetre longue : "
        f"{args.stage1_iter} iterations, "
        f"LR={args.stage1_lr:.3e} -> "
        f"{args.stage1_lr * args.stage1_lr_final_factor:.3e}"
    )
    if args.skip_stage2 or args.stage2_iter <= 0:
        print("  2. Etape fenetre courte : ignoree")
    else:
        print(
            f"  2. Adam conjoint sur la fenetre courte : "
            f"{args.stage2_iter} iterations, "
            f"LR={args.stage2_lr:.3e} -> "
            f"{args.stage2_lr * args.stage2_lr_final_factor:.3e}"
        )
    if args.skip_final_joint or args.final_joint_iter <= 0:
        print("  3. Raffinement conjoint faible LR : ignore par defaut")
    else:
        print(
            f"  3. Raffinement conjoint faible LR : "
            f"{args.final_joint_iter} iterations, "
            f"LR={args.final_joint_lr:.3e} -> "
            f"{args.final_joint_lr * args.final_joint_lr_final_factor:.3e}"
        )
    if args.skip_gauss_newton or args.gauss_newton_iter <= 0:
        print("  4. Raffinement Gauss-Newton : ignore par defaut")
    else:
        print(
            f"  4. Raffinement Gauss-Newton spectral : "
            f"{args.gauss_newton_iter} iterations, "
            f"pas={args.gauss_newton_step:.3g}, "
            f"ridge={args.gauss_newton_ridge:.3e}, "
            f"fd_eps={args.gauss_newton_fd_eps:.3e}, "
            f"batch={args.gauss_newton_batch_size}"
        )

    if args.skip_lbfgs or args.lbfgs_iter <= 0:
        print("  5. Raffinement L-BFGS : ignore par defaut")
    else:
        print(
            f"  5. Raffinement L-BFGS conjoint : "
            f"{args.lbfgs_iter} iterations max, "
            f"memoire={args.lbfgs_memory}, "
            f"tol={args.lbfgs_tol:.3e}"
        )
    print("=" * 72)
    print(f"STFT               : {stft_resolutions}")
    print(
        "Gauss-Newton       : "
        f"iter={args.gauss_newton_iter}, "
        f"step={args.gauss_newton_step:.3g}, "
        f"ridge={args.gauss_newton_ridge:.3e}, "
        f"batch={args.gauss_newton_batch_size}"
    )
    print(
        "L-BFGS             : "
        f"iter={args.lbfgs_iter}, "
        f"memory={args.lbfgs_memory}, "
        f"tol={args.lbfgs_tol:.3e}, "
        f"line-search={args.lbfgs_max_linesearch_steps}, "
        "mode=joint gamma-kappa-zeta"
    )
    print(
        f"LR stage 1          : {args.stage1_lr:.3e} -> "
        f"{args.stage1_lr * args.stage1_lr_final_factor:.3e}"
    )
    print(
        f"LR stage 2          : {args.stage2_lr:.3e} -> "
        f"{args.stage2_lr * args.stage2_lr_final_factor:.3e}"
    )
    print(
        f"Conjoint final      : iter={args.final_joint_iter}, "
        f"LR={args.final_joint_lr:.3e} -> "
        f"{args.final_joint_lr * args.final_joint_lr_final_factor:.3e}"
    )
    print(f"Dataset OpenWind    : {dataset_path}")
    print(f"Simulations cachees : {dataset_metadata['simulation_count']}")
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
    true_values = jnp.asarray(
        dataset_arrays["true_values"].reshape(-1, 4)
    )
    targets_short = jnp.asarray(
        dataset_arrays["pressure_short"].reshape(total_signals, -1)
    )
    targets_long = jnp.asarray(
        dataset_arrays["pressure_long"].reshape(total_signals, -1)[:, :long_keep]
    )

    repeat_row, signal_rows = run_experiment(
        0, args.seed, true_values, targets_short, targets_long,
        generation_T, calibration_T,
        data_matched, geometry, c, solve_short, solve_long, times_long[:long_keep],
        stft_resolutions, args.stft_dynamic_db, stft_allow_padding,
        args, output_dir,
    )
    repeat_rows = [repeat_row]
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