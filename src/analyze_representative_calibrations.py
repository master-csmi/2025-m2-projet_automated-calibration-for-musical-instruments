#!/usr/bin/env python3
"""
Analyse acoustique de trois calibrations représentatives parmi les 300 signaux.

Sélection automatique :
    - "good"      : cas le plus proche du 10e percentile de l'erreur paramétrique globale
    - "median"    : cas le plus proche du 50e percentile
    - "difficult" : cas le plus proche du 95e percentile

Erreur paramétrique globale :
    e_global = sqrt((e_gamma^2 + e_kappa^2 + e_zeta^2) / 3)

Pour chaque cas :
    1. la cible OpenWind exacte est lue dans le cache .npz utilisé à l'entraînement ;
    2. le signal DG est recalculé avec les paramètres calibrés du CSV ;
    3. les spectrogrammes OpenWind / DG sont tracés avec une normalisation commune ;
    4. plusieurs métriques acoustiques sont calculées et sauvegardées dans un CSV.

À lancer depuis le projet, par exemple :
python3 src/analyze_representative_calibrations.py \
    --results_csv experiments/gradient/results/.../all_signals.csv \
    --dataset_path experiments/gradient/datasets/openwind_Qr_kappa_gamma_zeta_300.npz \
    --calibration_t 0.05
"""

import argparse
import copy
import csv
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from numerics.dg.mesh import (
    cell_edges_from_nodes,
    create_uniform_nodes_with_ghosts,
)
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

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
            "Sélectionne trois calibrations représentatives et compare "
            "leurs spectrogrammes OpenWind et DG."
        )
    )
    parser.add_argument(
        "--results_csv",
        type=str,
        required=True,
        help="CSV all_signals.csv contenant les paramètres vrais et estimés.",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=(
            "experiments/gradient/datasets/"
            "openwind_Qr_kappa_gamma_zeta_300.npz"
        ),
        help="Cache OpenWind exact utilisé pour les calibrations.",
    )
    parser.add_argument("--type_S", type=str, default="const")
    parser.add_argument(
        "--calibration_t",
        type=float,
        default=0.05,
        help="Durée analysée dans les signaux, en secondes.",
    )
    parser.add_argument(
        "--n_fft",
        type=int,
        default=64,
        help="Taille de fenêtre STFT utilisée pour la figure diagnostique.",
    )
    parser.add_argument(
        "--hop",
        type=int,
        default=16,
        help="Hop STFT utilisé pour la figure diagnostique.",
    )
    parser.add_argument("--dynamic_db", type=float, default=60.0)
    parser.add_argument("--max_freq_hz", type=float, default=2000.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "experiments/gradient/results/"
            "representative_calibration_spectrograms"
        ),
    )
    return parser.parse_args()


def repo_root():
    # Le script est supposé être dans src/.
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


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def read_results_csv(path):
    rows = []
    with open(path, "r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV vide : {path}")

    required = {
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
    }
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(
            "Colonnes manquantes dans le CSV : "
            + ", ".join(sorted(missing))
        )

    converted = []
    for row in rows:
        item = dict(row)
        item["signal_idx"] = int(row["signal_idx"])
        for name in required - {"signal_idx"}:
            item[name] = float(row[name])

        errors = np.asarray(
            [
                item["relerr_gamma"],
                item["relerr_kappa"],
                item["relerr_zeta"],
            ],
            dtype=float,
        )
        item["global_rms_relerr"] = float(
            np.sqrt(np.mean(errors**2))
        )
        converted.append(item)

    return converted


def choose_representative_cases(rows):
    global_errors = np.asarray(
        [row["global_rms_relerr"] for row in rows],
        dtype=float,
    )

    specifications = (
        ("good", 0.10),
        ("median", 0.50),
        ("difficult", 0.95),
    )

    selected = []
    used_indices = set()

    for label, quantile in specifications:
        target = float(np.quantile(global_errors, quantile))
        order = np.argsort(np.abs(global_errors - target))

        chosen = None
        for idx in order:
            signal_idx = rows[int(idx)]["signal_idx"]
            if signal_idx not in used_indices:
                chosen = rows[int(idx)]
                break

        if chosen is None:
            raise RuntimeError("Impossible de sélectionner trois cas distincts.")

        chosen = dict(chosen)
        chosen["case_label"] = label
        chosen["target_quantile"] = quantile
        chosen["target_global_error"] = target
        selected.append(chosen)
        used_indices.add(chosen["signal_idx"])

    return selected


def load_dataset(path):
    with np.load(path, allow_pickle=False) as dataset:
        metadata = json.loads(str(dataset["metadata_json"].item()))

        pressure_long = np.asarray(dataset["pressure_long"], dtype=float)
        times_long = np.asarray(dataset["times_long"], dtype=float)
        true_values = np.asarray(dataset["true_values"], dtype=float)

        radiation = {
            "alpha": float(dataset["radiation_alpha"]),
            "beta": float(dataset["radiation_beta"]),
        }

    total_signals = int(np.prod(pressure_long.shape[:2]))

    arrays = {
        "pressure_long": pressure_long.reshape(total_signals, -1),
        "true_values": true_values.reshape(total_signals, -1),
        "times_long": times_long,
    }

    return metadata, arrays, radiation


def set_trainable_parameters(params):
    params = copy.deepcopy(params)

    # Ces paramètres doivent rester des feuilles dynamiques du PyTree afin
    # d'être remplacés signal par signal par set_param.
    dynamic_names = ("gamma_final", "kappa", "zeta", "Qr")
    for name in params["trainable"]:
        params["trainable"][name] = name in dynamic_names

    return params


def params_with_openwind_radiation(base_params, radiation, type_S):
    params = copy.deepcopy(base_params)

    set_nested(
        params,
        PARAM_JSON_PATHS["alpha"],
        radiation["alpha"],
    )
    set_nested(
        params,
        PARAM_JSON_PATHS["beta"],
        radiation["beta"],
    )

    data = build_physical_data(params, type_S)
    length = data.section.L_tube + data.section.L_bell

    # Même convention que dans le code de calibration.
    Zt = float(data.section(0.0) / data.section(length))
    set_nested(params, PARAM_JSON_PATHS["Zt"], Zt)

    return params


def make_solver_data(
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
):
    x_nodes, _ = create_uniform_nodes_with_ghosts(
        Nx,
        0.0,
        L_ref,
    )
    x_left, x_right = cell_edges_from_nodes(x_nodes)

    dt = CFL * (x_right[0] - x_left[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
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


def set_calibrated_values(data, row):
    data = set_param(
        data,
        "gamma_final",
        jnp.asarray(row["estimated_gamma"], dtype=jnp.float64),
        GEO_KEYS,
    )
    data = set_param(
        data,
        "kappa",
        jnp.asarray(row["estimated_kappa"], dtype=jnp.float64),
        GEO_KEYS,
    )
    data = set_param(
        data,
        "zeta",
        jnp.asarray(row["estimated_zeta"], dtype=jnp.float64),
        GEO_KEYS,
    )
    data = set_param(
        data,
        "Qr",
        jnp.asarray(row["known_Qr"], dtype=jnp.float64),
        GEO_KEYS,
    )
    return data


def compute_stft_magnitude(signal, times, n_fft, hop):
    """STFT magnitude with centered zero-padding.

    The previous version only used complete windows. On a short analysed
    interval (e.g. 0.05 s), n_fft=128 could leave only one STFT frame.
    pcolormesh then received a single time coordinate and displayed an empty
    (white) image.

    Here the signal is padded by n_fft//2 samples on both sides. This gives
    several centered frames spanning the complete analysed interval, including
    the beginning and the end of the signal.
    """
    signal = np.asarray(signal, dtype=float).reshape(-1)
    times = np.asarray(times, dtype=float).reshape(-1)

    if len(signal) != len(times):
        raise ValueError(
            f"signal et times de tailles différentes : "
            f"{len(signal)} != {len(times)}"
        )

    if len(signal) < 2:
        raise ValueError("Signal trop court.")

    n_fft = min(int(n_fft), max(2, len(signal)))
    hop = max(1, min(int(hop), n_fft))

    dt = float(np.median(np.diff(times)))
    window = np.hanning(n_fft)

    pad = n_fft // 2
    padded = np.pad(signal, (pad, pad), mode="constant")

    starts = np.arange(
        0,
        len(padded) - n_fft + 1,
        hop,
        dtype=int,
    )

    spectra = []
    frame_times = []

    for start in starts:
        frame = padded[start : start + n_fft]
        spectrum = np.fft.rfft(
            frame * window,
            n=n_fft,
        )
        spectra.append(np.abs(spectrum))

        # Center of the frame expressed in original-signal sample coordinates.
        center_original = start + n_fft // 2 - pad
        center_original = int(
            np.clip(center_original, 0, len(times) - 1)
        )
        frame_times.append(times[center_original])

    magnitude = np.asarray(spectra, dtype=float).T
    frequencies = np.fft.rfftfreq(n_fft, d=dt)

    return (
        np.asarray(frame_times, dtype=float),
        frequencies,
        magnitude,
    )


def centers_to_edges(values):
    """Convert regularly/irregularly spaced bin centers to pcolormesh edges."""
    values = np.asarray(values, dtype=float).reshape(-1)

    if len(values) == 1:
        # Fallback: create a small non-zero interval around the unique center.
        delta = max(abs(values[0]) * 1e-3, 1e-6)
        return np.asarray(
            [values[0] - 0.5 * delta, values[0] + 0.5 * delta]
        )

    mids = 0.5 * (values[:-1] + values[1:])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])

    return np.concatenate(([first], mids, [last]))


def magnitude_to_db(
    magnitude,
    reference_max,
    dynamic_db,
):
    reference_max = max(float(reference_max), 1e-30)
    floor = 10.0 ** (-float(dynamic_db) / 20.0)

    normalized = magnitude / reference_max
    db = 20.0 * np.log10(
        np.maximum(normalized, floor)
    )

    return np.maximum(db, -float(dynamic_db))


def spectral_metrics(
    target_mag,
    predicted_mag,
    dynamic_db,
):
    reference = max(float(np.max(target_mag)), 1e-30)

    target_db = magnitude_to_db(
        target_mag,
        reference,
        dynamic_db,
    )
    predicted_db = magnitude_to_db(
        predicted_mag,
        reference,
        dynamic_db,
    )

    difference_db = predicted_db - target_db

    mae_db = float(np.mean(np.abs(difference_db)))
    rmse_db = float(np.sqrt(np.mean(difference_db**2)))

    spectral_convergence = float(
        np.linalg.norm(predicted_mag - target_mag)
        / max(np.linalg.norm(target_mag), 1e-30)
    )

    target_flat = target_db.ravel()
    predicted_flat = predicted_db.ravel()

    if (
        np.std(target_flat) > 0.0
        and np.std(predicted_flat) > 0.0
    ):
        correlation = float(
            np.corrcoef(target_flat, predicted_flat)[0, 1]
        )
    else:
        correlation = np.nan

    return {
        "spectral_mae_db": mae_db,
        "spectral_rmse_db": rmse_db,
        "spectral_convergence": spectral_convergence,
        "spectral_correlation": correlation,
        "target_db": target_db,
        "predicted_db": predicted_db,
    }


def relative_l2(target, predicted):
    target = np.asarray(target, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    return float(
        np.linalg.norm(predicted - target)
        / max(np.linalg.norm(target), 1e-30)
    )


def plot_spectrogram_comparison(
    selected,
    spectrogram_data,
    output_path,
    max_freq_hz,
    dynamic_db,
):
    """
    Plot the OpenWind target and calibrated DG spectrograms for the three
    representative cases.

    A dedicated axis is reserved for the common colorbar so that it never
    overlaps the DG panels or the MAE/SC annotations.
    """
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(10.5, 10.0),
        sharex=False,
        sharey=True,
    )

    image = None

    for row_idx, (case, data) in enumerate(
        zip(selected, spectrogram_data)
    ):
        frame_times = data["frame_times"]
        frequencies = data["frequencies"]

        titles = (
            "OpenWind target",
            "Calibrated DG",
        )

        spectrograms = (
            data["target_db"],
            data["predicted_db"],
        )

        time_edges = centers_to_edges(frame_times)
        frequency_edges = centers_to_edges(frequencies)

        for col_idx, (title, spectrogram) in enumerate(
            zip(titles, spectrograms)
        ):
            ax = axes[row_idx, col_idx]

            image = ax.pcolormesh(
                time_edges,
                frequency_edges,
                spectrogram,
                shading="flat",
                vmin=-float(dynamic_db),
                vmax=0.0,
            )

            ax.set_ylim(
                0.0,
                min(
                    float(max_freq_hz),
                    float(frequencies[-1]),
                ),
            )

            ax.set_xlim(
                float(data["signal_times"][0]),
                float(data["signal_times"][-1]),
            )

            ax.set_xlabel("Time (s)")

            if col_idx == 0:
                ax.set_ylabel("Frequency (Hz)")

            if row_idx == 0:
                ax.set_title(
                    title,
                    fontsize=11,
                    pad=8,
                )

        case_name = {
            "good": "Well-calibrated case",
            "median": "Median case",
            "difficult": "Difficult case",
        }[case["case_label"]]

        # Label describing the selected percentile/case.
        axes[row_idx, 0].text(
            -0.24,
            0.5,
            (
                f"{case_name}\n"
                f"signal {case['signal_idx']}\n"
                f"$e_{{glob}}$="
                f"{100.0 * case['global_rms_relerr']:.2f}%"
            ),
            transform=axes[row_idx, 0].transAxes,
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
        )

        # Acoustic-error annotation in the DG panel.
        axes[row_idx, 1].text(
            0.96,
            0.95,
            (
                f"MAE={data['spectral_mae_db']:.3f} dB\n"
                f"SC={data['spectral_convergence']:.3e}"
            ),
            transform=axes[row_idx, 1].transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="0.4",
                alpha=0.85,
            ),
        )

    fig.suptitle(
        "Representative calibration cases: OpenWind vs calibrated DG",
        fontsize=13,
        y=0.97,
    )

    # Reserve a real margin on the right for the common colorbar.
    fig.subplots_adjust(
        left=0.14,
        right=0.84,
        bottom=0.08,
        top=0.92,
        hspace=0.32,
        wspace=0.12,
    )

    # Dedicated colorbar axis, fully outside the six spectrogram panels.
    cbar_ax = fig.add_axes([
        0.87,   # left
        0.14,   # bottom
        0.020,  # width
        0.72,   # height
    ])

    cbar = fig.colorbar(
        image,
        cax=cbar_ax,
    )

    cbar.set_label(
        "Magnitude (dB, normalized by OpenWind target)",
        labelpad=10,
    )

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_difference_maps(
    selected,
    spectrogram_data,
    output_path,
    max_freq_hz,
):
    """
    Plot absolute differences between the calibrated DG and OpenWind
    log-magnitude spectrograms.

    All cases use the same color scale and a dedicated colorbar axis.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.5, 9.0),
        sharex=False,
        sharey=True,
    )

    max_difference = max(
        float(
            np.max(
                np.abs(
                    data["predicted_db"]
                    - data["target_db"]
                )
            )
        )
        for data in spectrogram_data
    )
    max_difference = max(max_difference, 1e-12)

    image = None

    for ax, case, data in zip(
        axes,
        selected,
        spectrogram_data,
    ):
        difference = np.abs(
            data["predicted_db"]
            - data["target_db"]
        )

        time_edges = centers_to_edges(
            data["frame_times"]
        )
        frequency_edges = centers_to_edges(
            data["frequencies"]
        )

        image = ax.pcolormesh(
            time_edges,
            frequency_edges,
            difference,
            shading="flat",
            vmin=0.0,
            vmax=max_difference,
        )

        ax.set_ylim(
            0.0,
            min(
                float(max_freq_hz),
                float(data["frequencies"][-1]),
            ),
        )

        ax.set_xlim(
            float(data["signal_times"][0]),
            float(data["signal_times"][-1]),
        )

        ax.set_ylabel("Frequency (Hz)")
        ax.set_xlabel("Time (s)")

        label = {
            "good": "Well-calibrated",
            "median": "Median",
            "difficult": "Difficult",
        }[case["case_label"]]

        ax.set_title(
            f"{label} case — signal {case['signal_idx']} "
            f"— spectral MAE={data['spectral_mae_db']:.3f} dB",
            fontsize=10,
            pad=6,
        )

    fig.suptitle(
        "Absolute spectrogram differences",
        fontsize=13,
        y=0.97,
    )

    # Reserve a full margin on the right.
    fig.subplots_adjust(
        left=0.11,
        right=0.84,
        bottom=0.07,
        top=0.92,
        hspace=0.36,
    )

    # Dedicated colorbar axis, outside the three panels.
    cbar_ax = fig.add_axes([
        0.87,
        0.14,
        0.020,
        0.72,
    ])

    cbar = fig.colorbar(
        image,
        cax=cbar_ax,
    )

    cbar.set_label(
        r"$|D_{\mathrm{DG}}-D_{\mathrm{OW}}|$ (dB)",
        labelpad=10,
    )

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def save_metrics_csv(selected, spectrogram_data, output_path):
    fieldnames = [
        "case",
        "quantile",
        "signal_idx",
        "global_rms_relerr",
        "relerr_gamma",
        "relerr_kappa",
        "relerr_zeta",
        "signal_loss",
        "true_gamma",
        "estimated_gamma",
        "true_kappa",
        "estimated_kappa",
        "true_zeta",
        "estimated_zeta",
        "known_Qr",
        "relative_l2_pressure",
        "spectral_mae_db",
        "spectral_rmse_db",
        "spectral_convergence",
        "spectral_correlation",
    ]

    with open(output_path, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for case, data in zip(
            selected,
            spectrogram_data,
        ):
            writer.writerow(
                {
                    "case": case["case_label"],
                    "quantile": case["target_quantile"],
                    "signal_idx": case["signal_idx"],
                    "global_rms_relerr": case[
                        "global_rms_relerr"
                    ],
                    "relerr_gamma": case["relerr_gamma"],
                    "relerr_kappa": case["relerr_kappa"],
                    "relerr_zeta": case["relerr_zeta"],
                    "signal_loss": case["signal_loss"],
                    "true_gamma": case["true_gamma"],
                    "estimated_gamma": case[
                        "estimated_gamma"
                    ],
                    "true_kappa": case["true_kappa"],
                    "estimated_kappa": case[
                        "estimated_kappa"
                    ],
                    "true_zeta": case["true_zeta"],
                    "estimated_zeta": case[
                        "estimated_zeta"
                    ],
                    "known_Qr": case["known_Qr"],
                    "relative_l2_pressure": data[
                        "relative_l2_pressure"
                    ],
                    "spectral_mae_db": data[
                        "spectral_mae_db"
                    ],
                    "spectral_rmse_db": data[
                        "spectral_rmse_db"
                    ],
                    "spectral_convergence": data[
                        "spectral_convergence"
                    ],
                    "spectral_correlation": data[
                        "spectral_correlation"
                    ],
                }
            )


def main():
    args = parse_args()
    root = repo_root()

    results_csv = resolve_path(
        root,
        args.results_csv,
    )
    dataset_path = resolve_path(
        root,
        args.dataset_path,
    )
    output_dir = resolve_path(
        root,
        args.output_dir,
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = read_results_csv(results_csv)
    selected = choose_representative_cases(rows)

    print("\n=== Cas représentatifs sélectionnés ===")
    for case in selected:
        print(
            f"{case['case_label']:>10} | "
            f"signal={case['signal_idx']:3d} | "
            f"e_global={100.0 * case['global_rms_relerr']:.4f}% | "
            f"e_gamma={100.0 * case['relerr_gamma']:.4f}% | "
            f"e_kappa={100.0 * case['relerr_kappa']:.4f}% | "
            f"e_zeta={100.0 * case['relerr_zeta']:.4f}% | "
            f"loss={case['signal_loss']:.6e}"
        )

    metadata, arrays, radiation = load_dataset(
        dataset_path
    )

    total_signals = arrays["pressure_long"].shape[0]

    for case in selected:
        if not 0 <= case["signal_idx"] < total_signals:
            raise IndexError(
                f"signal_idx={case['signal_idx']} absent du dataset "
                f"({total_signals} signaux)."
            )

    # Vérification que le CSV correspond bien au cache OpenWind.
    true_values_cache = arrays["true_values"]

    for case in selected:
        idx = case["signal_idx"]

        reference = np.asarray(
            [
                case["true_gamma"],
                case["true_kappa"],
                case["true_zeta"],
                case["known_Qr"],
            ],
            dtype=float,
        )
        cached = np.asarray(
            true_values_cache[idx, :4],
            dtype=float,
        )

        if not np.allclose(
            reference,
            cached,
            rtol=1e-9,
            atol=1e-11,
        ):
            raise ValueError(
                f"Le CSV et le dataset ne correspondent pas pour "
                f"signal_idx={idx}.\n"
                f"CSV     : {reference}\n"
                f"Dataset : {cached}"
            )

    with open(
        root / "experiments/gradient/config/simu.json",
        "r",
    ) as file:
        train_params = json.load(file)["solver_params"]["train"]

    with open(
        root / "experiments/gradient/config/param.json",
        "r",
    ) as file:
        params = json.load(file)

    c = float(params["physics"]["c"])
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    params_dynamic = set_trainable_parameters(params)
    params_matched = params_with_openwind_radiation(
        params_dynamic,
        radiation,
        args.type_S,
    )

    data_base = build_physical_data(
        params_matched,
        args.type_S,
    )

    dataset_n_snapshot = int(
        metadata["N_snapshot"]
    )

    generation_T = float(
        metadata.get(
            "T_long",
            train_params["T_max"],
        )
    )

    if args.calibration_t > generation_T:
        raise ValueError(
            f"--calibration_t={args.calibration_t} dépasse "
            f"la durée du dataset {generation_T}."
        )

    times_dataset = np.asarray(
        arrays["times_long"],
        dtype=float,
    )

    long_keep = int(
        np.searchsorted(
            times_dataset,
            args.calibration_t,
            side="right",
        )
    )
    long_keep = max(2, long_keep)

    target_times = times_dataset[:long_keep]

    length = (
        data_base.section.L_tube
        + data_base.section.L_bell
    )

    geometry = build_solver_geometry(
        data_base,
        int(train_params["Nx"]),
        c,
    )
    bc = BC(type="full")

    # Important : on reproduit la grille temporelle du forward utilisé pour
    # la calibration (simulation sur generation_T puis restriction au préfixe).
    solve_long = make_solver_data(
        generation_T,
        float(train_params["cfl"]),
        int(train_params["Nx"]),
        dataset_n_snapshot,
        length,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    solve_times = np.asarray(
        (solve_long["n_snaps"] + 1)
        * solve_long["dt"],
        dtype=float,
    )

    # Les temps du cache et les temps du solveur doivent être cohérents.
    compare_count = min(
        len(times_dataset),
        len(solve_times),
    )
    max_time_diff = float(
        np.max(
            np.abs(
                times_dataset[:compare_count]
                - solve_times[:compare_count]
            )
        )
    )
    print(
        f"\nÉcart max entre grille temporelle OpenWind cache "
        f"et snapshots DG : {max_time_diff:.3e} s"
    )

    spectrogram_data = []

    for case in selected:
        idx = case["signal_idx"]

        target = np.asarray(
            arrays["pressure_long"][idx, :long_keep],
            dtype=float,
        )

        data_case = set_calibrated_values(
            data_base,
            case,
        )

        predicted = forward_snapshots(
            data_case,
            geometry,
            c,
            **solve_long,
        )
        predicted = np.asarray(
            jax.device_get(predicted),
            dtype=float,
        )[:long_keep]

        (
            frame_times,
            frequencies,
            target_mag,
        ) = compute_stft_magnitude(
            target,
            target_times,
            args.n_fft,
            args.hop,
        )

        (
            frame_times_pred,
            frequencies_pred,
            predicted_mag,
        ) = compute_stft_magnitude(
            predicted,
            target_times,
            args.n_fft,
            args.hop,
        )

        if not np.allclose(
            frame_times,
            frame_times_pred,
        ):
            raise RuntimeError(
                "Les trames temporelles STFT ne correspondent pas."
            )

        if not np.allclose(
            frequencies,
            frequencies_pred,
        ):
            raise RuntimeError(
                "Les fréquences STFT ne correspondent pas."
            )

        metrics = spectral_metrics(
            target_mag,
            predicted_mag,
            args.dynamic_db,
        )

        reference_max = max(
            float(np.max(target_mag)),
            1e-30,
        )

        data_result = {
            **metrics,
            "frame_times": frame_times,
            "frequencies": frequencies,
            "signal_times": target_times,
            "relative_l2_pressure": relative_l2(
                target,
                predicted,
            ),
            "target_signal": target,
            "predicted_signal": predicted,
            "reference_max": reference_max,
        }
        spectrogram_data.append(data_result)

        print(
            f"\n{case['case_label'].upper()} "
            f"(signal {idx})"
        )
        print(
            f"  erreur paramètres globale : "
            f"{100.0 * case['global_rms_relerr']:.4f}%"
        )
        print(
            f"  L2 relatif pression       : "
            f"{data_result['relative_l2_pressure']:.6e}"
        )
        print(
            f"  STFT                      : "
            f"n_fft={args.n_fft}, hop={args.hop}, "
            f"{len(frame_times)} trames"
        )
        print(
            f"  spectral MAE              : "
            f"{data_result['spectral_mae_db']:.6f} dB"
        )
        print(
            f"  spectral RMSE             : "
            f"{data_result['spectral_rmse_db']:.6f} dB"
        )
        print(
            f"  spectral convergence      : "
            f"{data_result['spectral_convergence']:.6e}"
        )
        print(
            f"  corrélation spectrale     : "
            f"{data_result['spectral_correlation']:.8f}"
        )

    comparison_path = (
        output_dir
        / "representative_spectrograms_openwind_vs_dg.png"
    )
    plot_spectrogram_comparison(
        selected,
        spectrogram_data,
        comparison_path,
        args.max_freq_hz,
        args.dynamic_db,
    )

    difference_path = (
        output_dir
        / "representative_spectrogram_differences.png"
    )
    plot_difference_maps(
        selected,
        spectrogram_data,
        difference_path,
        args.max_freq_hz,
    )

    metrics_path = (
        output_dir
        / "representative_spectral_metrics.csv"
    )
    save_metrics_csv(
        selected,
        spectrogram_data,
        metrics_path,
    )

    selection_path = (
        output_dir
        / "representative_selection.npz"
    )
    np.savez(
        selection_path,
        case_labels=np.asarray(
            [case["case_label"] for case in selected]
        ),
        signal_indices=np.asarray(
            [case["signal_idx"] for case in selected],
            dtype=int,
        ),
        global_rms_relerrs=np.asarray(
            [
                case["global_rms_relerr"]
                for case in selected
            ],
            dtype=float,
        ),
        target_signals=np.asarray(
            [
                data["target_signal"]
                for data in spectrogram_data
            ],
            dtype=float,
        ),
        predicted_signals=np.asarray(
            [
                data["predicted_signal"]
                for data in spectrogram_data
            ],
            dtype=float,
        ),
        times=target_times,
    )

    print("\n=== Fichiers créés ===")
    print(comparison_path)
    print(difference_path)
    print(metrics_path)
    print(selection_path)


if __name__ == "__main__":
    main()