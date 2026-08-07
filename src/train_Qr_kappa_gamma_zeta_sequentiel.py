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
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)

MIN_SCALE = 1e-8
DEFAULT_INIT_FACTOR = 0.8
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
PARAM_NAMES = ("gamma", "kappa", "zeta", "Qr")

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
            "Protocole sequentiel robuste : optimisation conjointe sur fenetre longue, "
            "optimisation conjointe sur fenetre courte, raffinement de Qr seul "
            "sur fenetre longue, puis raffinement conjoint optionnel."
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
            "Parametres a entrainer parmi gamma kappa zeta Qr. "
            "Les autres sont fixes a leur valeur OpenWind pour chaque signal."
        ),
    )

    parser.add_argument("--stage1_iter", type=int, default=300)
    parser.add_argument("--stage2_iter", type=int, default=100)
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage2_lr", type=float, default=5e-3)
    parser.add_argument("--stage1_lr_final_factor", type=float, default=0.8)
    parser.add_argument("--stage2_lr_final_factor", type=float, default=0.8)

    # Stage 3 : raffinement conjoint de kappa et Qr.
    parser.add_argument(
        "--kappa_qr_iter",
        type=int,
        default=150,
        help=(
            "Nombre d'iterations Adam pour le raffinement conjoint "
            "de kappa et Qr sur la fenetre longue."
        ),
    )
    parser.add_argument(
        "--kappa_qr_lr",
        type=float,
        default=1e-3,
        help="Learning rate initial du raffinement conjoint kappa-Qr.",
    )
    parser.add_argument(
        "--kappa_qr_lr_final_factor",
        type=float,
        default=0.5,
        help=(
            "Rapport entre le learning rate final et initial "
            "du raffinement kappa-Qr."
        ),
    )
    parser.add_argument(
        "--skip_kappa_qr",
        action="store_true",
        help="Desactive le raffinement conjoint de kappa et Qr.",
    )

    # Stage 4 : petit raffinement conjoint facultatif.
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
        default="64:16,128:32,256:64",
        help="Liste n_fft:hop, par exemple 64:16,128:32,256:64.",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument(
        "--no_stft_padding",
        action="store_true",
        help="Ignore une resolution STFT trop grande au lieu de zero-padder.",
    )

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
        "--lbfgs_mode",
        choices=("qr", "joint"),
        default="qr",
        help=(
            "Parametres raffines par L-BFGS : 'qr' conserve gamma, kappa et "
            "zeta fixes ; 'joint' raffine tous les parametres demandes."
        ),
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
        default="experiments/gradient/results/Qr_kappa_gamma_zeta_pressure_only",
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


def base_parameter_vector_from_params(params):
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    kappa = float(get_nested(params, PARAM_JSON_PATHS["kappa"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    Qr = float(get_nested(params, PARAM_JSON_PATHS["Qr"]))
    return np.asarray([gamma, kappa, zeta, Qr], dtype=float)


def set_parameter_vector_json(params, values):
    gamma, kappa, zeta, Qr = map(float, values)
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["kappa"], kappa)
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    set_nested(params, PARAM_JSON_PATHS["Qr"], Qr)
    return params


def set_parameter_vector_data(data, values):
    gamma, kappa, zeta, Qr = values
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    data = set_param(data, "Qr", Qr, GEO_KEYS)
    return data


def set_trainable_parameters(params):
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name in ("gamma_final", "kappa", "zeta", "Qr")
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
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(theta_one, scale_one, target_one):
        data = set_parameter_vector_data(data_init, theta_one * scale_one)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        pred = pred[: target_one.shape[0]]
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
        data = set_parameter_vector_data(data_init, theta_one * scale_one)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        pred = pred[: target_one.shape[0]]
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


def make_trained_predictions(data_init, geometry, c, estimated_values, solve_kwargs):
    def one_prediction(params):
        data = set_parameter_vector_data(data_init, params)
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
        "mean_relerr_kappa",
        "std_relerr_kappa",
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
        "true_kappa",
        "true_zeta",
        "true_Qr",
        "estimated_gamma",
        "estimated_kappa",
        "estimated_zeta",
        "estimated_Qr",
        "relerr_gamma",
        "relerr_kappa",
        "relerr_zeta",
        "relerr_Qr",
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
        r"$\gamma$": 100.0 * np.asarray(
            [r["relerr_gamma"] for r in signal_rows]
        ),
        r"$\kappa$": 100.0 * np.asarray(
            [r["relerr_kappa"] for r in signal_rows]
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
        100.0 * np.asarray([r["relerr_kappa"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_zeta"] for r in signal_rows]),
        100.0 * np.asarray([r["relerr_Qr"] for r in signal_rows]),
    ]

    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    ax.boxplot(
        values,
        tick_labels=[
            r"$\gamma$",
            r"$\kappa$",
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
        r"$\kappa$",
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
    true_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    scales = jnp.maximum(jnp.abs(true_values), MIN_SCALE)
    train_mask = jnp.asarray(
        [name in args.train_params for name in PARAM_NAMES],
        dtype=jnp.float64,
    )[None, :]
    normalized_true = true_values / scales
    theta = normalized_true * (
        train_mask * DEFAULT_INIT_FACTOR + (1.0 - train_mask)
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

    # Masque conjoint : tous les parametres demandes par --train_params.
    joint_mask = train_mask

    # Masque du bloc kappa-Qr. Un parametre ne peut etre actif dans
    # ce bloc que s'il a aussi ete demande dans --train_params.
    kappa_requested = "kappa" in args.train_params
    qr_requested = "Qr" in args.train_params
    kappa_qr_requested = kappa_requested and qr_requested
    kappa_qr_mask = jnp.asarray(
        [[0.0, 1.0, 0.0, 1.0]],
        dtype=jnp.float64,
    ) * joint_mask

    print(
        f"\n--- Stage 1 : Adam conjoint | "
        f"T={calibration_T:.4f} s (fenetre longue) ---"
    )
    theta, loss_stage1 = optimize_stage(
        theta,
        loss_long,
        joint_mask,
        args.stage1_lr,
        args.stage1_iter,
        args.print_every,
        args.stage1_lr_final_factor,
    )

    if args.skip_stage2 or args.stage2_iter <= 0:
        loss_stage2 = float(loss_short(theta))
        print("\n--- Stage 2 conjoint court ignore ---")
    else:
        print("\n--- Stage 2 : Adam conjoint | T=0.01 s (fenetre courte) ---")
        theta, loss_stage2 = optimize_stage(
            theta,
            loss_short,
            joint_mask,
            args.stage2_lr,
            args.stage2_iter,
            args.print_every,
            args.stage2_lr_final_factor,
        )

    if (
        args.skip_kappa_qr
        or args.kappa_qr_iter <= 0
        or not kappa_qr_requested
    ):
        loss_kappa_qr = float(loss_long(theta))
        print(
            "\n--- Stage 3 kappa-Qr ignore "
            "(--skip_kappa_qr, 0 iteration, ou kappa/Qr non entraines) ---"
        )
    else:
        print(
            f"\n--- Stage 3 : Adam kappa et Qr | "
            f"gamma et zeta fixes | T={calibration_T:.4f} s ---"
        )
        theta, loss_kappa_qr = optimize_stage(
            theta,
            loss_long,
            kappa_qr_mask,
            args.kappa_qr_lr,
            args.kappa_qr_iter,
            args.print_every,
            args.kappa_qr_lr_final_factor,
        )

    if (
        args.skip_final_joint
        or args.final_joint_iter <= 0
    ):
        loss_final_joint = float(loss_long(theta))
        print("\n--- Stage 4 conjoint final ignore ---")
    else:
        print(
            f"\n--- Stage 4 : Adam conjoint faible LR | "
            f"T={calibration_T:.4f} s ---"
        )
        theta, loss_final_joint = optimize_stage(
            theta,
            loss_long,
            joint_mask,
            args.final_joint_lr,
            args.final_joint_iter,
            args.print_every,
            args.final_joint_lr_final_factor,
        )

    if args.skip_lbfgs or args.lbfgs_iter <= 0:
        loss_lbfgs = np.nan
        print("\n--- Raffinement L-BFGS ignore ---")
    else:
        lbfgs_mask = (
            kappa_qr_mask
            if args.lbfgs_mode == "kappa_qr"
            else joint_mask
        )
        if args.lbfgs_mode == "kappa_qr" and not kappa_qr_requested:
            loss_lbfgs = np.nan
            print(
                "\n--- L-BFGS kappa-Qr ignore : "
                "kappa et Qr doivent etre entraines ---"
            )
        else:
            print(
                f"\n--- Raffinement L-BFGS ({args.lbfgs_mode}) | "
                f"T={calibration_T:.4f} s ---"
            )
            theta, loss_lbfgs = optimize_stage_lbfgs(
                theta=theta,
                loss_fn=loss_long,
                train_mask=lbfgs_mask,
                n_iter=args.lbfgs_iter,
                memory_size=args.lbfgs_memory,
                tolerance=args.lbfgs_tol,
                max_linesearch_steps=args.lbfgs_max_linesearch_steps,
                print_every=args.lbfgs_print_every,
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
    batch_size = true_np.shape[0]

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
                "true_Qr": float(true_np[i, 3]),
                "estimated_gamma": float(estimated_np[i, 0]),
                "estimated_kappa": float(estimated_np[i, 1]),
                "estimated_zeta": float(estimated_np[i, 2]),
                "estimated_Qr": float(estimated_np[i, 3]),
                "relerr_gamma": float(errors_np[i, 0]),
                "relerr_kappa": float(errors_np[i, 1]),
                "relerr_zeta": float(errors_np[i, 2]),
                "relerr_Qr": float(errors_np[i, 3]),
                "signal_loss": float(signal_losses[i]),
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
        predictions = make_trained_predictions(
            data_ref, geometry, c, estimated, solve_kwargs_long
        )[:, : targets_long.shape[1]]
        plot_signal_comparison(
            run_dir / "signals_openwind_vs_dg.png",
            np.asarray(snapshot_times_long),
            np.asarray(targets_long),
            np.asarray(predictions),
        )

    print(
        f"\nResume tirage {repeat_idx + 1}: "
        f"gamma={100*repeat_row['mean_relerr_gamma']:.3f}% | "
        f"kappa={100*repeat_row['mean_relerr_kappa']:.3f}% | "
        f"zeta={100*repeat_row['mean_relerr_zeta']:.3f}% | "
        f"Qr={100*repeat_row['mean_relerr_Qr']:.3f}% | "
        f"loss={final_loss:.3e} | temps={elapsed/60:.2f} min"
    )
    print(
        f"Loss stage 1 conjoint long={loss_stage1:.3e}, "
        f"stage 2 conjoint court={loss_stage2:.3e}, "
        f"stage 3 kappa-Qr={loss_kappa_qr:.3e}, "
        f"stage 4 conjoint={loss_final_joint:.3e}, "
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
        ("kappa_qr_iter", args.kappa_qr_iter),
        ("final_joint_iter", args.final_joint_iter),
    ):
        if value < 0:
            raise ValueError(f"--{name} doit etre positif ou nul.")
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
        ("stage1_lr", args.stage1_lr),
        ("stage2_lr", args.stage2_lr),
        ("kappa_qr_lr", args.kappa_qr_lr),
        ("final_joint_lr", args.final_joint_lr),
    ):
        if value <= 0.0:
            raise ValueError(f"--{name} doit etre strictement positif.")
    for name, factor in (
        ("stage1_lr_final_factor", args.stage1_lr_final_factor),
        ("stage2_lr_final_factor", args.stage2_lr_final_factor),
        ("kappa_qr_lr_final_factor", args.kappa_qr_lr_final_factor),
        ("final_joint_lr_final_factor", args.final_joint_lr_final_factor),
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
    total_signals = int(dataset_metadata["simulation_count"])
    print(
        "Organisation du cache : "
        f"{dataset_metadata['n_repeats']} x {dataset_metadata['n_signals']}"
    )
    print(f"Signaux dans le batch unique : {total_signals}")
    print(f"Parametres calibres : {', '.join(args.train_params)}")
    print(
        "Protocole          : conjoint long -> conjoint court -> "
        "Qr seul -> conjoint optionnel"
    )
    print("Observation         : p(L,t) uniquement")
    print(f"STFT               : {stft_resolutions}")
    print(
        "L-BFGS             : "
        f"iter={args.lbfgs_iter}, "
        f"memory={args.lbfgs_memory}, "
        f"tol={args.lbfgs_tol:.3e}, "
        f"line-search={args.lbfgs_max_linesearch_steps}, "
        f"mode={args.lbfgs_mode}"
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
        f"Raffinement kappa-Qr: iter={args.kappa_qr_iter}, "
        f"LR={args.kappa_qr_lr:.3e} -> "
        f"{args.kappa_qr_lr * args.kappa_qr_lr_final_factor:.3e}"
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
    true_values = jnp.asarray(dataset_arrays["true_values"].reshape(-1, 4))
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