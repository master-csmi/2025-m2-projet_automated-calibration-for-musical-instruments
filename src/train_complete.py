"""Joint parameter training using only the pressure MSTS loss."""

import copy
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import train_Qr_wr_gamma_zeta_scaling_complete as base
from inverse.l_func_nn import LFuncNN
from utils.param_func import set_param

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")


class NormalizedLFuncNN(eqx.Module):
    network: LFuncNN

    def __init__(self, key):
        self.network = LFuncNN(
            [1, 8, 8, 1], activation=jax.nn.tanh, key=key
        )

    def __call__(self, y):
        # Removes the exact scale ambiguity between zeta and l(y).
        return self.network(y) / (self.network(jnp.asarray(1.0)) + 1e-12)


def replace_l(data, ell_nn):
    return eqx.tree_at(lambda current: current.l, data, ell_nn)


def set_parameter_vector_data(data, values, qr_value):
    """Injecte gamma, kappa, zeta et Qr dans les données physiques."""
    gamma, kappa, zeta = values

    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    data = set_param(data, "Qr", qr_value, GEO_KEYS)

    return data


def set_joint_trainable_parameters(params):
    """
    Rend remplaçables dans le PyTree les paramètres utilisés par ce script.

    gamma_final, kappa et zeta sont optimisés.
    Qr est également dynamique car sa valeur est injectée signal par signal.
    """
    params = copy.deepcopy(params)

    dynamic_names = {"gamma_final", "kappa", "zeta", "Qr"}

    for name in params["trainable"]:
        params["trainable"][name] = name in dynamic_names

    return params


def optimize_joint_stage(
    state, loss_fn, scalar_lr, law_lr, n_iter, print_every,
    scalar_final_factor, law_final_factor,
):
    raw_theta, ell_nn = state
    scalar_opt, scalar_schedule = base.make_optimizer(
        scalar_lr, n_iter, scalar_final_factor
    )
    law_opt, law_schedule = base.make_optimizer(
        law_lr, n_iter, law_final_factor
    )
    scalar_state = scalar_opt.init(raw_theta)
    law_state = law_opt.init(ell_nn)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))
    final_loss = np.nan

    for iteration in range(n_iter):
        start = time.time()
        loss_value, (grad_theta, grad_law) = value_and_grad((raw_theta, ell_nn))
        scalar_updates, scalar_state = scalar_opt.update(
            grad_theta, scalar_state, raw_theta
        )
        law_updates, law_state = law_opt.update(grad_law, law_state, ell_nn)
        raw_theta = base.optax.apply_updates(raw_theta, scalar_updates)
        ell_nn = base.optax.apply_updates(ell_nn, law_updates)
        final_loss = float(loss_value)
        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            print(
                f"iter {iteration:4d} | loss={final_loss:.4e} | "
                f"lr_params={float(scalar_schedule(iteration)):.3e} | "
                f"lr_l={float(law_schedule(iteration)):.3e} | "
                f"t={time.time() - start:.2f}s"
            )
    return (raw_theta, ell_nn), final_loss


def make_joint_loss(
    data_init, geometry, c, pressure_targets, reed_targets, scales,
    solve_kwargs, args, stft_resolutions, stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(raw_one, scale_one, target_p, target_y, ell_nn):
        values = base.positive_parameters(raw_one, scale_one)
        data = set_parameter_vector_data(
            data_init,
            values[:3],
            values[3],
        )
        data = replace_l(data, ell_nn)
        pred_p, pred_y = base.forward_complete_snapshots(
            data, geometry, c, **solve_kwargs
        )
        total = base.loss_fn_signal(
            pred_p,
            target_p,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )
        return total

    vmapped = jax.vmap(one_loss, in_axes=(0, 0, 0, 0, None))

    def loss(state):
        raw_theta, ell_nn = state
        return jnp.mean(
            vmapped(
                raw_theta,
                scales,
                pressure_targets,
                reed_targets,
                ell_nn,
            )
        )

    return loss


def make_joint_losses_per_signal(
    data_init, geometry, c, pressure_targets, reed_targets, scales,
    solve_kwargs, args, stft_resolutions, stft_dynamic_db,
    stft_allow_padding,
):
    def one_loss(raw_one, scale_one, target_p, target_y, ell_nn):
        values = base.positive_parameters(raw_one, scale_one)
        data = set_parameter_vector_data(
            data_init,
            values[:3],
            values[3],
        )
        data = replace_l(data, ell_nn)
        pred_p, pred_y = base.forward_complete_snapshots(
            data, geometry, c, **solve_kwargs
        )
        pressure_loss = base.loss_fn_signal(
            pred_p,
            target_p,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )
        # The reed loss is kept only as a diagnostic in the CSV output.
        reed_loss = base.normalized_mse(pred_y, target_y)
        return pressure_loss, pressure_loss, reed_loss

    vmapped = jax.vmap(one_loss, in_axes=(0, 0, 0, 0, None))

    @jax.jit
    def evaluate(state):
        raw_theta, ell_nn = state
        return vmapped(
            raw_theta,
            scales,
            pressure_targets,
            reed_targets,
            ell_nn,
        )

    return evaluate


def make_joint_predictions(
    data_init,
    geometry,
    c,
    values,
    ell_nn,
    solve_kwargs,
):
    def one_prediction(parameters):
        data = set_parameter_vector_data(
            data_init,
            parameters[:3],
            parameters[3],
        )
        data = replace_l(data, ell_nn)
        return base.forward_complete_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )

    return jax.jit(jax.vmap(one_prediction))(values)


def save_l_plot(path, ell_nn, reference_l):
    import matplotlib.pyplot as plt

    y = jnp.linspace(0.0, 1.5, 400)
    learned = jax.vmap(ell_nn)(y)
    reference = jax.vmap(reference_l)(y)
    relative_l2 = jnp.linalg.norm(learned - reference) / (
        jnp.linalg.norm(reference) + 1e-12
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y, reference, label="l(y) DG initial")
    ax.plot(y, learned, "--", label="l(y) appris")
    ax.set_xlabel("y")
    ax.set_ylabel("l(y)")
    ax.set_title(
        f"Erreur relative $L^2$ sur $l(y)$ : "
        f"{100.0 * float(relative_l2):.3f} %"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def relative_l2_error_law(ell_nn, reference_l):
    """Relative L2 error of l(y), evaluated on the plotting interval."""
    y = jnp.linspace(0.0, 1.5, 400)
    learned = jax.vmap(ell_nn)(y)
    reference = jax.vmap(reference_l)(y)
    return jnp.linalg.norm(learned - reference) / (
        jnp.linalg.norm(reference) + 1e-12
    )


def run_joint_experiment(
    repeat_idx, seed, true_values_np,
    pressure_targets_short, reed_targets_short,
    pressure_targets_long, reed_targets_long,
    data_ref, geometry, c, solve_kwargs_short, solve_kwargs_long,
    snapshot_times_long, stft_resolutions, stft_dynamic_db,
    stft_allow_padding, args, output_dir,
):
    start = time.time()
    all_values = jnp.asarray(true_values_np, dtype=jnp.float64)
    if all_values.shape[1] < 4:
        raise ValueError(
            "Le dataset doit contenir [gamma, kappa, zeta, Qr]."
        )

    true_values = all_values[:, :3]
    true_qr = all_values[:, 3]

    scales = jnp.maximum(jnp.abs(true_values), base.MIN_SCALE)
    qr_scales = jnp.maximum(jnp.abs(true_qr), base.MIN_SCALE)
    scales_full = jnp.concatenate((scales, qr_scales[:, None]), axis=1)
    initial_values = jnp.concatenate(
        (true_values, true_qr[:, None]),
        axis=1,
    )
    raw_theta = base.inverse_softplus(
        base.DEFAULT_INIT_FACTOR * initial_values / scales_full
    )
    ell_nn = NormalizedLFuncNN(jax.random.PRNGKey(seed))
    state = (raw_theta, ell_nn)

    loss_short = make_joint_loss(
        data_ref, geometry, c, pressure_targets_short, reed_targets_short,
        scales_full, solve_kwargs_short, args, stft_resolutions,
        stft_dynamic_db, stft_allow_padding,
    )
    loss_long = make_joint_loss(
        data_ref, geometry, c, pressure_targets_long, reed_targets_long,
        scales_full, solve_kwargs_long, args, stft_resolutions,
        stft_dynamic_db, stft_allow_padding,
    )

    print("\n--- Stage 1 conjoint : T=T_max ---")
    state, loss_stage1 = optimize_joint_stage(
        state, loss_long, args.stage1_lr, args.l_lr, args.stage1_iter,
        args.print_every, args.stage1_lr_final_factor,
        args.l_lr_final_factor,
    )
    if args.skip_stage2:
        loss_stage2 = float(loss_short(state))
        print("\n--- Stage 2 ignore (--skip_stage2) ---")
    else:
        print("\n--- Stage 2 conjoint : T=0.01 s ---")
        state, loss_stage2 = optimize_joint_stage(
            state, loss_short, args.stage2_lr, args.l_lr, args.stage2_iter,
            args.print_every, args.stage2_lr_final_factor,
            args.l_lr_final_factor,
        )

    raw_theta, ell_nn = state
    estimated_values = base.positive_parameters(raw_theta, scales_full)
    estimated = estimated_values[:, :3]
    estimated_qr = estimated_values[:, 3]
    errors, global_error = base.relative_errors(estimated, true_values)
    ell_relative_l2 = float(relative_l2_error_law(ell_nn, data_ref.l))
    losses_fn = make_joint_losses_per_signal(
        data_ref, geometry, c, pressure_targets_long, reed_targets_long,
        scales_full, solve_kwargs_long, args, stft_resolutions,
        stft_dynamic_db, stft_allow_padding,
    )
    total_loss, pressure_loss, reed_loss = losses_fn(state)
    total_loss = np.asarray(total_loss)
    pressure_loss = np.asarray(pressure_loss)
    reed_loss = np.asarray(reed_loss)
    true_np = np.asarray(true_values)
    estimated_np = np.asarray(estimated)
    errors_np = np.asarray(errors)
    elapsed = time.time() - start

    rows = []
    for i in range(true_np.shape[0]):
        rows.append({
            "repeat_idx": repeat_idx, "seed": seed, "signal_idx": i,
            "true_gamma": float(true_np[i, 0]),
            "true_kappa": float(true_np[i, 1]),
            "true_zeta": float(true_np[i, 2]),
            "true_Qr": float(true_qr[i]),
            "estimated_Qr": float(estimated_qr[i]),
            "relerr_Qr": float(np.abs((estimated_qr[i] - true_qr[i]) / max(abs(true_qr[i]), base.MIN_SCALE))),
            "estimated_gamma": float(estimated_np[i, 0]),
            "estimated_kappa": float(estimated_np[i, 1]),
            "estimated_zeta": float(estimated_np[i, 2]),
            "relerr_gamma": float(errors_np[i, 0]),
            "relerr_kappa": float(errors_np[i, 1]),
            "relerr_zeta": float(errors_np[i, 2]),
            "signal_loss": float(total_loss[i]),
            "pressure_loss": float(pressure_loss[i]),
            "reed_loss": float(reed_loss[i]),
            "elapsed_seconds_repeat": elapsed,
        })

    summary = {
        "repeat_idx": repeat_idx, "seed": seed,
        "n_signals": true_np.shape[0],
        "mean_relerr_gamma": float(np.mean(errors_np[:, 0])),
        "std_relerr_gamma": float(np.std(errors_np[:, 0])),
        "mean_relerr_kappa": float(np.mean(errors_np[:, 1])),
        "std_relerr_kappa": float(np.std(errors_np[:, 1])),
        "mean_relerr_zeta": float(np.mean(errors_np[:, 2])),
        "std_relerr_zeta": float(np.std(errors_np[:, 2])),
        "mean_relerr_Qr": float(np.mean(np.abs((estimated_qr - true_qr) / jnp.maximum(jnp.abs(true_qr), base.MIN_SCALE)))),
        "std_relerr_Qr": float(np.std(np.abs((estimated_qr - true_qr) / jnp.maximum(jnp.abs(true_qr), base.MIN_SCALE)))),
        "global_relerr": global_error,
        "ell_relative_l2": ell_relative_l2,
        "final_loss": float(np.mean(total_loss)),
        "elapsed_seconds": elapsed,
    }

    run_dir = output_dir / f"joint_seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    base.write_csv(run_dir / "signals.csv", rows, signal_fieldnames())
    eqx.tree_serialise_leaves(run_dir / "ell_nn.eqx", ell_nn)
    save_l_plot(run_dir / "ell_nn.png", ell_nn, data_ref.l)
    if args.save_detailed_plots:
        pred_p, pred_y = make_joint_predictions(
            data_ref,
            geometry,
            c,
            estimated,
            ell_nn,
            solve_kwargs_long,
        )
        base.plot_signal_comparison(
            run_dir / "pressure_openwind_vs_dg.png",
            np.asarray(snapshot_times_long), np.asarray(pressure_targets_long),
            np.asarray(pred_p),
        )
        base.plot_reed_comparison(
            run_dir / "reed_openwind_vs_dg.png",
            np.asarray(snapshot_times_long), np.asarray(reed_targets_long),
            np.asarray(pred_y),
        )

    print(
        f"\nEntrainement conjoint termine | loss={summary['final_loss']:.3e} "
        f"| stage1={loss_stage1:.3e} | stage2={loss_stage2:.3e}"
    )
    print(
        f"Erreur relative L2 sur l(y), y dans [0, 1.5] : "
        f"{100.0 * ell_relative_l2:.3f} %"
    )
    print(f"Reseau sauvegarde : {run_dir / 'ell_nn.eqx'}")
    return summary, rows


def signal_fieldnames():
    return [
        "repeat_idx",
        "seed",
        "signal_idx",
        "true_gamma",
        "true_kappa",
        "true_zeta",
        "true_Qr",
        "estimated_Qr",
        "estimated_gamma",
        "estimated_kappa",
        "estimated_zeta",
        "relerr_Qr",
        "relerr_gamma",
        "relerr_kappa",
        "relerr_zeta",
        "signal_loss",
        "pressure_loss",
        "reed_loss",
        "elapsed_seconds_repeat",
    ]


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
        "ell_relative_l2",
        "final_loss",
        "elapsed_seconds",
    ]


def main():
    original_parse_args = base.parse_args

    def parse_joint_args():
        args = original_parse_args()
        # train_complete deliberately optimizes only the bell-pressure MSTS.
        args.pressure_weight = 1.0
        args.reed_weight = 0.0
        default_output = (
            "experiments/gradient/results/Qr_wr_gamma_zeta_pressure_and_y"
        )
        if args.output_dir == default_output:
            args.output_dir = "experiments/gradient/results/train_complete_gamma_kappa_zeta_joint"
        return args

    base.parse_args = parse_joint_args
    base.set_trainable_parameters = set_joint_trainable_parameters
    base.signal_fieldnames = signal_fieldnames
    base.repeat_fieldnames = repeat_fieldnames
    base.run_experiment = run_joint_experiment
    print("Entrainement conjoint: l(y), gamma, kappa, zeta et Qr")
    base.main()


if __name__ == "__main__":
    main()