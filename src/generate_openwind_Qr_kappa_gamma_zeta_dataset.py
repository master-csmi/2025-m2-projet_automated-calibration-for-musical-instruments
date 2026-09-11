"""Pregenerate the OpenWind dataset for gamma, kappa, zeta and Qr."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

from physics.bc import BC
from train_Qr_wr_gamma_zeta_scaling_complete import (
    P_CLOSED,
    REED_OPENING,
    build_physical_data,
    make_solver_data,
    run_openwind_reference,
)


PARAM_JSON_PATHS = {
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "kappa": ("left_bc_params", "kappa"),
    "zeta": ("left_bc_params", "zeta"),
    "Qr": ("left_bc_params", "Qr"),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Genere une fois les simulations OpenWind pour "
            "gamma, kappa, zeta et Qr."
        )
    )
    parser.add_argument("--type_S", default="const")
    parser.add_argument("--n_signals", type=int, default=10)
    parser.add_argument("--n_repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random_factor_min", type=float, default=0.75)
    parser.add_argument("--random_factor_max", type=float, default=1.25)
    parser.add_argument("--ow_order", type=int, default=4)
    parser.add_argument("--ow_theta", type=float, default=0.5)
    parser.add_argument("--ow_l_ele", type=float, default=5.0e-4)
    parser.add_argument(
        "--output",
        default=(
            "experiments/gradient/datasets/"
            "openwind_Qr_kappa_gamma_zeta_300.npz"
        ),
    )
    return parser.parse_args()


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


def base_parameter_vector_from_params(params):
    """Return parameters in the dataset order [gamma, kappa, zeta, Qr]."""
    gamma = float(get_nested(params, PARAM_JSON_PATHS["gamma_final"]))
    kappa = float(get_nested(params, PARAM_JSON_PATHS["kappa"]))
    zeta = float(get_nested(params, PARAM_JSON_PATHS["zeta"]))
    Qr = float(get_nested(params, PARAM_JSON_PATHS["Qr"]))
    return np.asarray([gamma, kappa, zeta, Qr], dtype=float)


def set_parameter_vector_json(params, values):
    """Inject [gamma, kappa, zeta, Qr] into a parameter dictionary."""
    gamma, kappa, zeta, Qr = map(float, values)
    set_nested(params, PARAM_JSON_PATHS["gamma_final"], gamma)
    set_nested(params, PARAM_JSON_PATHS["kappa"], kappa)
    set_nested(params, PARAM_JSON_PATHS["zeta"], zeta)
    set_nested(params, PARAM_JSON_PATHS["Qr"], Qr)
    return params


def sample_true_parameters(
    base_values,
    n_signals,
    factor_min,
    factor_max,
    seed,
):
    rng = np.random.default_rng(seed)
    factors = rng.uniform(
        factor_min,
        factor_max,
        size=(n_signals, 4),
    )
    return np.asarray(base_values[None, :] * factors, dtype=float)


def main():
    args = parse_args()

    if args.n_signals <= 0 or args.n_repeats <= 0:
        raise ValueError("n_signals et n_repeats doivent etre positifs.")

    if (
        args.random_factor_min <= 0.0
        or args.random_factor_min >= args.random_factor_max
    ):
        raise ValueError("Bornes aleatoires invalides.")

    root = Path(__file__).resolve().parents[1]

    with open(root / "experiments/gradient/config/simu.json") as file:
        train_params = json.load(file)["solver_params"]["train"]

    with open(root / "experiments/gradient/config/param.json") as file:
        params = json.load(file)

    c = float(params["physics"]["c"])
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    data_ref = build_physical_data(params, args.type_S)
    length = data_ref.section.L_tube + data_ref.section.L_bell
    bc = BC(type="full")

    solve_short = make_solver_data(
        0.01,
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        length,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    solve_long = make_solver_data(
        train_params["T_max"],
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        length,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    times_short = np.asarray(
        (solve_short["n_snaps"] + 1) * solve_short["dt"],
        dtype=float,
    )
    times_long = np.asarray(
        (solve_long["n_snaps"] + 1) * solve_long["dt"],
        dtype=float,
    )

    metadata = {
        "format_version": 1,
        "parameter_names": ["gamma", "kappa", "zeta", "Qr"],
        "type_S": args.type_S,
        "n_signals": args.n_signals,
        "n_repeats": args.n_repeats,
        "seed": args.seed,
        "random_factor_min": args.random_factor_min,
        "random_factor_max": args.random_factor_max,
        "T_short": 0.01,
        "T_long": float(train_params["T_max"]),
        "Nx": int(train_params["Nx"]),
        "N_snapshot": int(train_params["N_snapshot"]),
        "cfl": float(train_params["cfl"]),
        "c": c,
        "ow_order": args.ow_order,
        "ow_theta": args.ow_theta,
        "ow_l_ele": args.ow_l_ele,
        "simulation_count": args.n_repeats * args.n_signals,
    }

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = output.with_suffix(".partial.npz")

    shape = (
        args.n_repeats,
        args.n_signals,
        train_params["N_snapshot"],
    )

    true_values_all = np.empty(
        (args.n_repeats, args.n_signals, 4),
        dtype=float,
    )
    pressure_short = np.empty(shape, dtype=float)
    reed_short = np.empty(shape, dtype=float)
    pressure_long = np.empty(shape, dtype=float)
    reed_long = np.empty(shape, dtype=float)

    start_repeat = 0
    radiation = None

    if checkpoint.exists():
        with np.load(checkpoint, allow_pickle=False) as saved:
            saved_metadata = json.loads(
                str(saved["metadata_json"].item())
            )

            if saved_metadata != metadata:
                raise ValueError(
                    f"Checkpoint incompatible: {checkpoint}. "
                    "Supprime-le ou reutilise exactement les memes arguments."
                )

            start_repeat = int(saved["completed_repeats"])
            true_values_all[:] = saved["true_values"]
            pressure_short[:] = saved["pressure_short"]
            reed_short[:] = saved["reed_short"]
            pressure_long[:] = saved["pressure_long"]
            reed_long[:] = saved["reed_long"]

            radiation = {
                "alpha": float(saved["radiation_alpha"]),
                "beta": float(saved["radiation_beta"]),
            }

        print(
            f"Reprise du checkpoint apres {start_repeat} repetitions: "
            f"{checkpoint}"
        )

    base_values = base_parameter_vector_from_params(params)

    print("\n=== Parametres OpenWind de reference ===")
    print(f"gamma = {base_values[0]:.8g}")
    print(f"kappa = {base_values[1]:.8g}")
    print(f"zeta  = {base_values[2]:.8g}")
    print(f"Qr    = {base_values[3]:.8g}")
    print("Ordre du dataset : [gamma, kappa, zeta, Qr]")

    total = args.n_repeats * args.n_signals
    completed = start_repeat * args.n_signals
    start = time.time()

    for repeat_idx in range(start_repeat, args.n_repeats):
        values = sample_true_parameters(
            base_values,
            args.n_signals,
            args.random_factor_min,
            args.random_factor_max,
            args.seed + repeat_idx,
        )
        true_values_all[repeat_idx] = values

        for signal_idx, parameter_vector in enumerate(values):
            params_true = set_parameter_vector_json(
                copy.deepcopy(params),
                parameter_vector,
            )

            t_ow, _, p_right, y_ow, _, ow_params = run_openwind_reference(
                param_json=params_true,
                T_max=float(train_params["T_max"]),
                type_S=args.type_S,
                theta=args.ow_theta,
                l_ele=args.ow_l_ele,
                order=args.ow_order,
            )

            t_ow = np.asarray(t_ow, dtype=float)
            p_right = np.asarray(p_right, dtype=float) / P_CLOSED
            y_ow = np.asarray(y_ow, dtype=float) / REED_OPENING

            pressure_short[repeat_idx, signal_idx] = np.interp(
                times_short,
                t_ow,
                p_right,
            )
            reed_short[repeat_idx, signal_idx] = np.interp(
                times_short,
                t_ow,
                y_ow,
            )
            pressure_long[repeat_idx, signal_idx] = np.interp(
                times_long,
                t_ow,
                p_right,
            )
            reed_long[repeat_idx, signal_idx] = np.interp(
                times_long,
                t_ow,
                y_ow,
            )

            if radiation is None:
                radiation = ow_params

            completed += 1
            elapsed = time.time() - start
            remaining = (
                elapsed / max(completed, 1) * (total - completed)
            )

            print(
                f"[{completed:3d}/{total}] "
                f"repeat={repeat_idx:02d} "
                f"signal={signal_idx:02d} | "
                f"elapsed={elapsed / 60:.1f} min | "
                f"reste~{remaining / 60:.1f} min"
            )

        np.savez_compressed(
            checkpoint,
            metadata_json=np.asarray(
                json.dumps(metadata, sort_keys=True)
            ),
            completed_repeats=np.asarray(repeat_idx + 1),
            true_values=true_values_all,
            pressure_short=pressure_short,
            reed_short=reed_short,
            pressure_long=pressure_long,
            reed_long=reed_long,
            radiation_alpha=np.asarray(float(radiation["alpha"])),
            radiation_beta=np.asarray(float(radiation["beta"])),
        )

        print(
            f"Checkpoint sauvegarde: "
            f"{repeat_idx + 1}/{args.n_repeats}"
        )

    np.savez_compressed(
        output,
        metadata_json=np.asarray(
            json.dumps(metadata, sort_keys=True)
        ),
        true_values=true_values_all,
        pressure_short=pressure_short,
        reed_short=reed_short,
        pressure_long=pressure_long,
        reed_long=reed_long,
        times_short=times_short,
        times_long=times_long,
        radiation_alpha=np.asarray(float(radiation["alpha"])),
        radiation_beta=np.asarray(float(radiation["beta"])),
    )

    if checkpoint.exists():
        checkpoint.unlink()

    print(f"\nDataset sauvegarde: {output}")
    print(f"Simulations OpenWind effectuees: {total}")


if __name__ == "__main__":
    main()
