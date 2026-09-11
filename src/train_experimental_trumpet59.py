#!/usr/bin/env python3
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

try:
    import soundfile as sf
except ImportError as exc:
    raise ImportError(
        "Le paquet 'soundfile' est nécessaire pour lire le .aif. "
        "Installe-le avec: pip install soundfile"
    ) from exc

from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.solve import forward_snapshots

jax.config.update("jax_enable_x64", True)

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
MIN_POSITIVE = 1e-8


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calibration expérimentale de gamma_i, kappa, zeta et du gain "
            "microphone commun à partir d'un fichier .aif."
        )
    )

    parser.add_argument("--audio_path", type=str, required=True)
    parser.add_argument(
        "--sample_intervals",
        type=str,
        default="",
        help="'debut:fin,debut:fin,...'. Vide = détection RMS automatique.",
    )
    parser.add_argument("--analysis_end", type=float, default=130.0)
    parser.add_argument("--rms_frame_ms", type=float, default=20.0)
    parser.add_argument("--rms_threshold_fraction", type=float, default=0.08)
    parser.add_argument("--min_segment_duration", type=float, default=2.0)
    parser.add_argument("--merge_gap_duration", type=float, default=0.5)
    parser.add_argument("--window_duration", type=float, default=0.050)

    parser.add_argument("--type_S", type=str, default="double_cone")
    parser.add_argument("--warmup_duration", type=float, default=0.050)
    parser.add_argument("--n_snapshot", type=int, default=600)

    parser.add_argument("--fixed_fr", type=float, default=None)
    parser.add_argument("--fixed_Qr", type=float, default=None)
    parser.add_argument("--fixed_alpha", type=float, default=None)
    parser.add_argument("--fixed_beta", type=float, default=None)

    parser.add_argument("--gamma_init_min", type=float, default=0.30)
    parser.add_argument("--gamma_init_max", type=float, default=0.50)
    parser.add_argument("--kappa_init", type=float, default=0.70)
    parser.add_argument("--zeta_init", type=float, default=0.40)
    parser.add_argument("--gain_init", type=float, default=0.10)

    parser.add_argument(
        "--multistart_kappa_zeta",
        type=str,
        default=(
            "0.55:0.30;0.55:0.50;"
            "0.70:0.30;0.70:0.40;0.70:0.50;"
            "0.85:0.30;0.85:0.50"
        ),
    )
    parser.add_argument("--multistart_iter", type=int, default=20)
    parser.add_argument("--multistart_lr", type=float, default=1e-2)
    parser.add_argument("--multistart_lr_final_factor", type=float, default=0.8)

    parser.add_argument("--stage1_iter", type=int, default=200)
    parser.add_argument("--stage1_lr", type=float, default=1e-2)
    parser.add_argument("--stage1_lr_final_factor", type=float, default=0.5)

    parser.add_argument("--fine_iter", type=int, default=100)
    parser.add_argument("--fine_lr", type=float, default=1e-3)
    parser.add_argument("--fine_lr_final_factor", type=float, default=0.1)

    parser.add_argument("--print_every", type=int, default=20)
    parser.add_argument("--gamma_monotonic_weight", type=float, default=0.0)

    parser.add_argument(
        "--stft_resolutions",
        type=str,
        default="32:8,64:16,128:32",
    )
    parser.add_argument("--stft_dynamic_db", type=float, default=60.0)
    parser.add_argument("--no_stft_padding", action="store_true")

    parser.add_argument(
        "--param_json",
        type=str,
        default="experiments/data_calib/config/param.json",
    )
    parser.add_argument(
        "--simu_json",
        type=str,
        default="experiments/data_calib/config/simu.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/data_calib/results/experimental_trumpet59",
    )

    return parser.parse_args()


def repo_root():
    return Path(__file__).resolve().parents[1]


def parse_intervals(value):
    value = value.strip()
    if not value:
        return []
    intervals = []
    for token in value.split(","):
        start, end = [float(v.strip()) for v in token.split(":", 1)]
        if not 0.0 <= start < end:
            raise ValueError(f"Intervalle invalide: {token!r}")
        intervals.append((start, end))
    return intervals


def frame_rms(audio, fs, frame_ms):
    frame_size = max(1, int(round(frame_ms * 1e-3 * fs)))
    n_frames = len(audio) // frame_size
    trimmed = audio[: n_frames * frame_size]
    frames = trimmed.reshape(n_frames, frame_size)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    times = (np.arange(n_frames) * frame_size + frame_size / 2) / fs
    return times, rms


def segments_from_boolean(active, times, min_duration, merge_gap):
    if len(active) == 0:
        return []

    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    ends = np.flatnonzero(active & np.r_[~active[1:], True])
    segments = [(times[s], times[e]) for s, e in zip(starts, ends)]

    if not segments:
        return []

    merged = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return [(s, e) for s, e in merged if e - s >= min_duration]


def detect_stationary_segments(
    audio,
    fs,
    analysis_end,
    frame_ms,
    threshold_fraction,
    min_duration,
    merge_gap,
):
    n_keep = min(len(audio), int(round(analysis_end * fs)))
    times, rms = frame_rms(audio[:n_keep], fs, frame_ms)

    noise_floor = float(np.quantile(rms, 0.20))
    high_level = float(np.quantile(rms, 0.95))
    threshold = noise_floor + threshold_fraction * (high_level - noise_floor)

    active = rms > threshold
    segments = segments_from_boolean(active, times, min_duration, merge_gap)
    return segments, times, rms, threshold


def choose_stationary_window(interval, duration):
    start, end = interval
    if end - start < duration:
        raise ValueError(
            f"Segment {interval} plus court que window_duration={duration}."
        )
    center = 0.5 * (start + end)
    return center - 0.5 * duration, center + 0.5 * duration


def extract_audio_targets(audio, fs, intervals, obs_times):
    targets = []
    rows = []

    if len(obs_times) < 2:
        raise ValueError("Il faut au moins deux temps d'observation.")

    dt_obs = float(obs_times[1] - obs_times[0])
    duration = float(obs_times[-1] - obs_times[0] + dt_obs)

    for idx, interval in enumerate(intervals):
        w0, w1 = choose_stationary_window(interval, duration)

        i0 = max(0, int(np.floor(w0 * fs)))
        i1 = min(len(audio), int(np.ceil(w1 * fs)))

        sample = np.asarray(audio[i0:i1], dtype=np.float64)
        sample = sample - np.mean(sample)

        t_sample = np.arange(sample.size) / fs
        relative_obs = np.asarray(obs_times) - obs_times[0]
        target = np.interp(relative_obs, t_sample, sample)

        targets.append(target)
        rows.append({
            "sample_idx": idx,
            "segment_start_s": interval[0],
            "segment_end_s": interval[1],
            "window_start_s": w0,
            "window_end_s": w1,
            "audio_rms": float(np.sqrt(np.mean(sample**2))),
            "audio_max_abs": float(np.max(np.abs(sample))),
        })

    return np.asarray(targets, dtype=np.float64), rows


def save_rms_overview(path, times, rms, threshold, segments):
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.plot(times, rms, linewidth=1.0, label="RMS locale")
    ax.axhline(threshold, linestyle="--", label=f"Seuil={threshold:.3e}")

    for idx, (start, end) in enumerate(segments):
        ax.axvspan(start, end, alpha=0.15)
        ax.text(
            0.5 * (start + end),
            np.max(rms) * 0.92,
            str(idx + 1),
            ha="center",
            va="top",
        )

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("RMS")
    ax.set_title("Détection des sons stationnaires")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def prepare_params(params, args):
    params = copy.deepcopy(params)

    if args.fixed_fr is not None:
        params["left_bc_params"]["fr"] = float(args.fixed_fr)
    if args.fixed_Qr is not None:
        params["left_bc_params"]["Qr"] = float(args.fixed_Qr)
    if args.fixed_alpha is not None:
        params["right_bc_params"]["alpha"] = float(args.fixed_alpha)
    if args.fixed_beta is not None:
        params["right_bc_params"]["beta"] = float(args.fixed_beta)

    dynamic = {"gamma_final", "kappa", "zeta"}
    for name in params["trainable"]:
        params["trainable"][name] = name in dynamic

    data_tmp = build_physical_data(params, args.type_S)
    length = float(data_tmp.section.L_tube + data_tmp.section.L_bell)
    Zt = float(data_tmp.section(0.0) / data_tmp.section(length))
    params["right_bc_params"]["Zt"] = Zt

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


def set_case_data(data, gamma, kappa, zeta):
    data = set_param(data, "gamma_final", gamma, GEO_KEYS)
    data = set_param(data, "kappa", kappa, GEO_KEYS)
    data = set_param(data, "zeta", zeta, GEO_KEYS)
    return data


def parse_stft_resolutions(value):
    out = []
    for token in value.split(","):
        n_fft, hop = [int(v) for v in token.strip().split(":")]
        out.append((n_fft, hop))
    return tuple(out)


def parse_kappa_zeta_starts(value):
    starts = []
    for token in value.split(";"):
        token = token.strip()
        if not token:
            continue
        kappa, zeta = [float(v.strip()) for v in token.split(":")]
        starts.append((kappa, zeta))
    return starts


def pack_initial_state(n_signals, gamma_min, gamma_max, kappa, zeta, gain):
    gammas = jnp.linspace(gamma_min, gamma_max, n_signals)
    return {
        "gammas": gammas,
        "log_kappa": jnp.log(jnp.asarray(max(kappa, MIN_POSITIVE))),
        "log_zeta": jnp.log(jnp.asarray(max(zeta, MIN_POSITIVE))),
        "log_gain": jnp.log(jnp.asarray(max(gain, MIN_POSITIVE))),
    }


def unpack_state(state):
    gammas = jnp.maximum(state["gammas"], MIN_POSITIVE)
    kappa = jnp.exp(state["log_kappa"])
    zeta = jnp.exp(state["log_zeta"])
    gain = jnp.exp(state["log_gain"])
    return gammas, kappa, zeta, gain


def make_experimental_loss(
    data_template,
    geometry,
    c,
    solve_kwargs,
    comparison_indices,
    targets,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    gamma_monotonic_weight,
):
    comparison_indices = jnp.asarray(comparison_indices, dtype=jnp.int32)
    targets = jnp.asarray(targets, dtype=jnp.float64)

    def one_prediction(gamma, kappa, zeta):
        data = set_case_data(data_template, gamma, kappa, zeta)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return pred[comparison_indices]

    vmapped_prediction = jax.vmap(one_prediction, in_axes=(0, None, None))

    def loss_fn(state):
        gammas, kappa, zeta, gain = unpack_state(state)
        preds = vmapped_prediction(gammas, kappa, zeta)
        preds_micro = gain * preds

        losses = jax.vmap(
            lambda pred, target: loss_fn_signal(
                pred,
                target,
                stft_resolutions=stft_resolutions,
                stft_dynamic_db=stft_dynamic_db,
                stft_allow_padding=stft_allow_padding,
            )
        )(preds_micro, targets)

        data_loss = jnp.mean(losses)

        if gamma_monotonic_weight > 0.0 and gammas.shape[0] > 1:
            mono = jnp.mean(jax.nn.relu(gammas[:-1] - gammas[1:]) ** 2)
        else:
            mono = 0.0

        return data_loss + gamma_monotonic_weight * mono

    def losses_per_signal(state):
        gammas, kappa, zeta, gain = unpack_state(state)
        preds = vmapped_prediction(gammas, kappa, zeta)
        preds_micro = gain * preds

        return jax.vmap(
            lambda pred, target: loss_fn_signal(
                pred,
                target,
                stft_resolutions=stft_resolutions,
                stft_dynamic_db=stft_dynamic_db,
                stft_allow_padding=stft_allow_padding,
            )
        )(preds_micro, targets)

    def predictions(state):
        gammas, kappa, zeta, gain = unpack_state(state)
        preds = vmapped_prediction(gammas, kappa, zeta)
        return gain * preds

    return loss_fn, jax.jit(losses_per_signal), jax.jit(predictions)


def make_optimizer(lr, n_iter, final_factor):
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=max(n_iter, 1),
        alpha=final_factor,
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=scheduler),
    )
    return optimizer, scheduler


def optimize_state(state, loss_fn, lr, n_iter, final_factor, print_every, label):
    if n_iter <= 0:
        return state, float(loss_fn(state))

    optimizer, scheduler = make_optimizer(lr, n_iter, final_factor)
    opt_state = optimizer.init(state)
    value_and_grad = jax.jit(jax.value_and_grad(loss_fn))

    last_loss = np.nan

    print(f"\n--- {label} ---")

    for iteration in range(n_iter):
        t0 = time.time()

        value, grads = value_and_grad(state)
        updates, opt_state = optimizer.update(grads, opt_state, state)
        state = optax.apply_updates(state, updates)

        state["gammas"] = jnp.maximum(state["gammas"], MIN_POSITIVE)
        last_loss = float(value)

        if iteration % max(print_every, 1) == 0 or iteration == n_iter - 1:
            gammas, kappa, zeta, gain = unpack_state(state)
            print(
                f"iter {iteration:4d} | loss={last_loss:.4e} | "
                f"lr={float(scheduler(iteration)):.3e} | "
                f"kappa={float(kappa):.4f} | "
                f"zeta={float(zeta):.4f} | "
                f"a={float(gain):.4e} | "
                f"gamma=[{float(jnp.min(gammas)):.3f},"
                f"{float(jnp.max(gammas)):.3f}] | "
                f"t={time.time()-t0:.2f}s"
            )

    return state, last_loss


def run_global_multistart(
    n_signals,
    starts,
    gamma_init_min,
    gamma_init_max,
    gain_init,
    loss_fn,
    lr,
    n_iter,
    final_factor,
    print_every,
):
    best_state = None
    best_loss = np.inf

    print(
        f"\n=== Multi-start global : {len(starts)} départs "
        f"kappa-zeta, {n_iter} itérations par départ ==="
    )

    for idx, (kappa0, zeta0) in enumerate(starts):
        state0 = pack_initial_state(
            n_signals,
            gamma_init_min,
            gamma_init_max,
            kappa0,
            zeta0,
            gain_init,
        )

        candidate, _ = optimize_state(
            state0,
            loss_fn,
            lr,
            n_iter,
            final_factor,
            print_every,
            label=(
                f"Multi-start {idx+1}/{len(starts)} "
                f"(kappa0={kappa0:.3f}, zeta0={zeta0:.3f})"
            ),
        )

        candidate_loss = float(loss_fn(candidate))
        print(f"Loss après départ {idx+1}: {candidate_loss:.4e}")

        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_state = candidate

    print(f"\nMeilleure loss multi-start : {best_loss:.4e}")
    return best_state


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_comparisons(path, obs_times, targets, predictions, gammas):
    n = len(targets)
    n_cols = 2
    n_rows = int(np.ceil(n / n_cols))

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 3.2 * n_rows),
        squeeze=False,
        sharex=True,
    )

    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue

        ax.plot(obs_times, targets[i], label="Micro", linewidth=1.2)
        ax.plot(obs_times, predictions[i], "--", label="DG × gain", linewidth=1.2)
        ax.set_title(rf"Son {i+1} — $\gamma={gammas[i]:.4f}$")
        ax.set_xlabel("Temps relatif (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    start_time = time.time()
    root = repo_root()

    audio_path = Path(args.audio_path).expanduser().resolve()
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    param_path = Path(args.param_json)
    if not param_path.is_absolute():
        param_path = root / param_path

    simu_path = Path(args.simu_json)
    if not simu_path.is_absolute():
        simu_path = root / simu_path

    with open(param_path, "r") as f:
        params = json.load(f)
    with open(simu_path, "r") as f:
        train_params = json.load(f)["solver_params"]["train"]

    audio, fs = sf.read(str(audio_path), dtype="float64")
    if audio.ndim != 1:
        raise ValueError(f"Le script attend un fichier mono. Forme reçue: {audio.shape}.")

    manual_intervals = parse_intervals(args.sample_intervals)

    detected_segments, rms_times, rms, rms_threshold = detect_stationary_segments(
        audio,
        fs,
        analysis_end=args.analysis_end,
        frame_ms=args.rms_frame_ms,
        threshold_fraction=args.rms_threshold_fraction,
        min_duration=args.min_segment_duration,
        merge_gap=args.merge_gap_duration,
    )

    if manual_intervals:
        intervals = manual_intervals
        print(f"Intervalles manuels utilisés : {len(intervals)}")
    else:
        intervals = detected_segments
        print(f"Segments RMS détectés : {len(intervals)}")
        for i, interval in enumerate(intervals):
            print(f"  son {i+1}: {interval[0]:.3f} -> {interval[1]:.3f} s")

    if not intervals:
        raise RuntimeError(
            "Aucun son stationnaire détecté. Utilise --sample_intervals."
        )

    save_rms_overview(
        output_dir / "rms_detection.png",
        rms_times,
        rms,
        rms_threshold,
        intervals,
    )

    params = prepare_params(params, args)

    c = float(params["physics"]["c"])
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    data_template = build_physical_data(params, args.type_S)
    length = float(data_template.section.L_tube + data_template.section.L_bell)

    geometry = build_solver_geometry(
        data_template,
        int(train_params["Nx"]),
        c,
    )

    bc = BC(type="full")
    total_duration = args.warmup_duration + args.window_duration

    solve_kwargs = make_solver_data(
        T_max=total_duration,
        CFL=float(train_params["cfl"]),
        Nx=int(train_params["Nx"]),
        N_snapshot=args.n_snapshot,
        L_ref=length,
        c=c,
        bc=bc,
        phi0=phi0,
        y0=y0,
        z0=z0,
    )

    snapshot_times = (
        np.asarray(solve_kwargs["n_snaps"]) + 1
    ) * float(solve_kwargs["dt"])

    comparison_indices = np.flatnonzero(
        snapshot_times >= args.warmup_duration
    )

    if comparison_indices.size < 8:
        raise ValueError("Pas assez de snapshots après le warm-up.")

    obs_times = snapshot_times[comparison_indices]
    obs_times = obs_times - obs_times[0]

    targets, sample_rows = extract_audio_targets(
        audio,
        fs,
        intervals,
        obs_times,
    )

    order = np.argsort([row["audio_rms"] for row in sample_rows])
    targets = targets[order]
    sample_rows = [sample_rows[i] for i in order]

    for new_idx, row in enumerate(sample_rows):
        row["sample_idx"] = new_idx

    stft_resolutions = parse_stft_resolutions(args.stft_resolutions)
    stft_allow_padding = not args.no_stft_padding

    loss_fn, losses_per_signal_fn, predictions_fn = make_experimental_loss(
        data_template=data_template,
        geometry=geometry,
        c=c,
        solve_kwargs=solve_kwargs,
        comparison_indices=comparison_indices,
        targets=targets,
        stft_resolutions=stft_resolutions,
        stft_dynamic_db=args.stft_dynamic_db,
        stft_allow_padding=stft_allow_padding,
        gamma_monotonic_weight=args.gamma_monotonic_weight,
    )

    starts = parse_kappa_zeta_starts(args.multistart_kappa_zeta)

    if starts and args.multistart_iter > 0:
        state = run_global_multistart(
            n_signals=len(intervals),
            starts=starts,
            gamma_init_min=args.gamma_init_min,
            gamma_init_max=args.gamma_init_max,
            gain_init=args.gain_init,
            loss_fn=loss_fn,
            lr=args.multistart_lr,
            n_iter=args.multistart_iter,
            final_factor=args.multistart_lr_final_factor,
            print_every=args.print_every,
        )
    else:
        state = pack_initial_state(
            len(intervals),
            args.gamma_init_min,
            args.gamma_init_max,
            args.kappa_init,
            args.zeta_init,
            args.gain_init,
        )

    state, _ = optimize_state(
        state,
        loss_fn,
        lr=args.stage1_lr,
        n_iter=args.stage1_iter,
        final_factor=args.stage1_lr_final_factor,
        print_every=args.print_every,
        label="Stage 1 principal",
    )

    state, _ = optimize_state(
        state,
        loss_fn,
        lr=args.fine_lr,
        n_iter=args.fine_iter,
        final_factor=args.fine_lr_final_factor,
        print_every=args.print_every,
        label="Finition faible LR",
    )

    gammas, kappa, zeta, gain = unpack_state(state)
    gammas_np = np.asarray(gammas)
    losses_np = np.asarray(losses_per_signal_fn(state))
    predictions_np = np.asarray(predictions_fn(state))
    final_loss = float(np.mean(losses_np))

    for i, row in enumerate(sample_rows):
        row.update({
            "estimated_gamma": float(gammas_np[i]),
            "signal_loss": float(losses_np[i]),
        })

    write_csv(output_dir / "samples.csv", sample_rows)

    summary = [{
        "n_signals": len(sample_rows),
        "estimated_kappa": float(kappa),
        "estimated_zeta": float(zeta),
        "estimated_gain_a": float(gain),
        "fixed_fr": float(params["left_bc_params"]["fr"]),
        "fixed_Qr": float(params["left_bc_params"]["Qr"]),
        "fixed_alpha": float(params["right_bc_params"]["alpha"]),
        "fixed_beta": float(params["right_bc_params"]["beta"]),
        "Zt_geometry": float(params["right_bc_params"]["Zt"]),
        "final_mean_loss": final_loss,
    }]
    write_csv(output_dir / "summary.csv", summary)

    plot_comparisons(
        output_dir / "micro_vs_dg.png",
        obs_times,
        targets,
        predictions_np,
        gammas_np,
    )

    print("\n" + "=" * 72)
    print("RESULTATS EXPERIMENTAUX")
    print("=" * 72)

    for i, gamma in enumerate(gammas_np):
        print(
            f"son {i+1:02d}: gamma={gamma:.6f} | "
            f"loss={losses_np[i]:.4e} | "
            f"RMS audio={sample_rows[i]['audio_rms']:.4e}"
        )

    print(f"\nkappa commun = {float(kappa):.6f}")
    print(f"zeta commun  = {float(zeta):.6f}")
    print(f"gain a       = {float(gain):.6e}")
    print(f"loss moyenne = {final_loss:.6e}")

    print("\nParamètres fixés :")
    print(f"fr    = {params['left_bc_params']['fr']}")
    print(f"Qr    = {params['left_bc_params']['Qr']}")
    print(f"alpha = {params['right_bc_params']['alpha']}")
    print(f"beta  = {params['right_bc_params']['beta']}")
    print(f"Zt    = {params['right_bc_params']['Zt']}")

    print(f"\nDossier résultats : {output_dir}")
    print(f"Temps total : {(time.time()-start_time)/60:.2f} min")


if __name__ == "__main__":
    main()
