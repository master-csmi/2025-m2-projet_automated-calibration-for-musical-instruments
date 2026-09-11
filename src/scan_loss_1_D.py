import copy
import csv
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.parse_args import parse_args
from utils.res_openwind import run_openwind_reference
from utils.solve import forward_snapshots


jax.config.update("jax_enable_x64", True)


P_CLOSED = 5e3
GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

PARAM_JSON_PATHS = {
    "epsilon": ("left_bc_params", "epsilon"),
    "beta": ("right_bc_params", "beta"),
    "alpha": ("right_bc_params", "alpha"),
    "zeta": ("left_bc_params", "zeta"),
    "kappa": ("left_bc_params", "kappa"),
    "Zt": ("right_bc_params", "Zt"),
    "fr": ("left_bc_params", "fr"),
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "t_attack": ("left_bc_params", "mouth_pressure_params", "t_attack"),
    "L_tube": ("instrument_geometry", "tube", "L_tube"),
    "R_tube": ("instrument_geometry", "tube", "R_tube"),
    "L_bell": ("instrument_geometry", "bell", "L_bell"),
    "k_bell": ("instrument_geometry", "bell", "k_bell"),
    "Qr": ("left_bc_params", "Qr"),
}

SCAN_RANGES = {
    "alpha": (6.0e4, 1.0e5),
    "beta": (0.1, 1.0),
    "Zt": (0.2, 3.0),
    "kappa": (0.1, 2.0),
    "fr": (50.0, 260.0),
    "Qr": (20.0, 250.0),
    "gamma_final": (0.1, 0.9),
    "zeta": (0.1, 0.9),
}


# Spectrogrammes diagnostiques pour interpreter les ruptures des scans 1D.
# Ils sont generes automatiquement uniquement pour les parametres ci-dessous.
SPECTROGRAM_SCAN_PARAMS = ("gamma_final", "zeta")
SPECTROGRAM_DYNAMIC_DB = 60.0
SPECTROGRAM_MAX_FREQ_HZ = 2000.0



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


def value_from_params(params, name):
    return float(get_nested(params, PARAM_JSON_PATHS[name]))


def repo_root():
    return Path(__file__).resolve().parents[1]


def resolve_project_path(root, path):
    path = Path(path)
    if path.is_absolute():
        return path

    parts = path.parts
    if len(parts) >= 2 and parts[0] == ".." and parts[1] == "experiments":
        return root.joinpath(*parts[1:])

    return root / path


def parse_stft_resolutions(value):
    resolutions = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue

        if ":" in item:
            n_fft, hop = item.split(":", maxsplit=1)
        elif "x" in item:
            n_fft, hop = item.split("x", maxsplit=1)
        else:
            raise ValueError(
                "Chaque resolution STFT doit etre au format n_fft:hop, "
                f"recu: {item}"
            )

        n_fft = int(n_fft)
        hop = int(hop)
        if n_fft <= 0 or hop <= 0:
            raise ValueError(f"Resolution STFT invalide: {item}")

        resolutions.append((n_fft, hop))

    if len(resolutions) == 0:
        raise ValueError("La liste --scan_stft_resolutions est vide.")

    return tuple(resolutions)


def parse_float_list(value, name):
    values = []
    for item in value.split(","):
        item = item.strip()
        if item:
            values.append(float(item))

    if len(values) == 0:
        raise ValueError(f"La liste --{name} est vide.")

    return tuple(values)


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
    n_snaps = jnp.round(jnp.linspace(0, nsteps - 1, N_snapshot)).astype(jnp.int32)

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


def params_with_openwind_radiation(base_params, ow_params, type_S):
    """Match the DG radiation parameters used in ``pressure_at_bell``.

    OpenWind's radiation connector supplies ``alpha`` and ``beta``.  The DG
    impedance boundary additionally uses the geometrical area ratio
    ``Zt = S(0) / S(L)``; leaving the JSON value (usually 1) here gives a
    different boundary condition for a flared bore.
    """
    params = copy.deepcopy(base_params)

    if ow_params.get("alpha") is not None:
        set_nested(params, PARAM_JSON_PATHS["alpha"], ow_params["alpha"])
    if ow_params.get("beta") is not None:
        set_nested(params, PARAM_JSON_PATHS["beta"], ow_params["beta"])

    data = build_physical_data(params, type_S)
    length = data.section.L_tube + data.section.L_bell
    zt_geometry = data.section(0.0) / data.section(length)
    set_nested(params, PARAM_JSON_PATHS["Zt"], float(zt_geometry))

    return params


def make_openwind_target(params_true, type_S, T_max, snapshot_times, args):
    # Use the same OpenWind spatial resolution as pressure_at_bell unless the
    # caller explicitly requests another one.
    ow_l_ele = args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4
    t_ow, _, p_right, _, _, ow_params = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=ow_l_ele,
        order=args.ow_order,
    )

    target = np.interp(
        np.asarray(snapshot_times, dtype=float),
        np.asarray(t_ow, dtype=float),
        np.asarray(p_right, dtype=float),
    )

    return jnp.asarray(target / P_CLOSED, dtype=jnp.float64), ow_params


def envelope_rms_loss(pred, target, win=256, hop=64):
    """
    Compare les enveloppes RMS locales de deux signaux.

    Cette perte est utile pour Qr, car Qr agit surtout sur
    l'amortissement, donc sur l'evolution lente de l'amplitude.
    """
    n = pred.shape[0]
    win = min(int(win), int(n))
    hop = min(int(hop), win)
    n_frames = (n - win) // hop + 1

    idx = jnp.arange(win)[None, :] + hop * jnp.arange(n_frames)[:, None]
    pred_frames = pred[idx]
    target_frames = target[idx]

    env_pred = jnp.sqrt(jnp.mean(pred_frames**2, axis=1) + 1e-12)
    env_target = jnp.sqrt(jnp.mean(target_frames**2, axis=1) + 1e-12)

    return jnp.mean((env_pred - env_target) ** 2) / (
        jnp.mean(env_target**2) + 1e-12
    )


def stft_mag_numpy(signal, n_fft, hop_length):
    """STFT magnitude used only for diagnostic figures."""
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 1:
        signal = signal.reshape(-1)

    if signal.size < n_fft:
        signal = np.pad(signal, (0, n_fft - signal.size))

    window = np.hanning(n_fft)
    n_frames = 1 + (signal.size - n_fft) // hop_length
    frames = np.stack(
        [
            signal[i * hop_length : i * hop_length + n_fft] * window
            for i in range(n_frames)
        ],
        axis=0,
    )
    return np.abs(np.fft.rfft(frames, axis=-1))


def choose_visual_stft_resolution(signal_length, stft_resolutions):
    """Choose the largest configured STFT window that fits without padding."""
    valid = [(int(n_fft), int(hop)) for n_fft, hop in stft_resolutions if n_fft <= signal_length]
    if valid:
        return max(valid, key=lambda item: item[0])

    # Fallback for very short signals.
    n_fft = max(8, int(signal_length))
    hop = max(1, n_fft // 4)
    return n_fft, hop


def select_spectrogram_scan_indices(values, losses_stft, true_value):
    """Select representative scan points around the reference and the strongest loss jump.

    The goal is diagnostic: show a low-value case, the reference neighbourhood,
    both sides of the strongest adjacent STFT-loss variation, and a high-value case.
    """
    values = np.asarray(values, dtype=float)
    losses_stft = np.asarray(losses_stft, dtype=float)
    n = values.size
    if n == 0:
        return []

    idx_true = int(np.argmin(np.abs(values - true_value)))
    idx_low = int(round(0.20 * (n - 1)))
    idx_high = int(round(0.80 * (n - 1)))

    if n >= 2:
        # Normalize by the scan increment so this remains meaningful for non-unit ranges.
        dx = np.diff(values)
        dloss = np.diff(losses_stft)
        slope = np.abs(dloss / np.maximum(np.abs(dx), 1e-15))
        idx_jump_left = int(np.argmax(slope))
        idx_jump_right = idx_jump_left + 1
    else:
        idx_jump_left = idx_jump_right = 0

    selected = [idx_low, idx_true, idx_jump_left, idx_jump_right, idx_high]
    # Remove duplicates while keeping the scan order in the final figure.
    return sorted(set(max(0, min(n - 1, idx)) for idx in selected))


def plot_scan_spectrogram_diagnostics(
    *,
    output_dir,
    source,
    param_name,
    T_max,
    values,
    losses_stft,
    true_value,
    target,
    data_base,
    geometry,
    c,
    solve_kwargs,
    snapshot_times,
    stft_resolutions,
):
    """Generate a target + representative DG spectrogram comparison for a 1D scan."""
    selected_indices = select_spectrogram_scan_indices(values, losses_stft, true_value)
    if not selected_indices:
        return None

    selected_values = [float(values[idx]) for idx in selected_indices]
    predictions = []
    for value in selected_values:
        data = set_param(data_base, param_name, value, GEO_KEYS)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        predictions.append(np.asarray(pred, dtype=float))

    target_np = np.asarray(target, dtype=float)
    times = np.asarray(snapshot_times, dtype=float)
    if times.size >= 2:
        dt_snap = float(np.median(np.diff(times)))
    else:
        dt_snap = 1.0

    n_fft, hop = choose_visual_stft_resolution(target_np.size, stft_resolutions)
    target_mag = stft_mag_numpy(target_np, n_fft, hop)
    pred_mags = [stft_mag_numpy(pred, n_fft, hop) for pred in predictions]

    # Common normalization based on the target: amplitude differences remain visible.
    scale = float(np.max(target_mag)) + 1e-12
    eps = 10.0 ** (-SPECTROGRAM_DYNAMIC_DB / 20.0)

    def to_db(mag):
        return 20.0 * np.log10(mag / scale + eps)

    specs_db = [to_db(target_mag)] + [to_db(mag) for mag in pred_mags]
    vmax = max(0.0, max(float(np.nanmax(spec)) for spec in specs_db))
    vmin = -SPECTROGRAM_DYNAMIC_DB

    freqs = np.fft.rfftfreq(n_fft, d=dt_snap)
    n_frames = target_mag.shape[0]
    frame_times = times[0] + np.arange(n_frames) * hop * dt_snap
    t_start = float(frame_times[0]) if frame_times.size else 0.0
    t_end = float(frame_times[-1] + n_fft * dt_snap) if frame_times.size else float(T_max)
    f_max = min(float(freqs[-1]), SPECTROGRAM_MAX_FREQ_HZ)

    n_panels = len(specs_db)
    ncols = 3
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.3 * ncols, 3.4 * nrows),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    titles = ["OpenWind target" if source == "openwind" else "DG target"]
    titles += [f"{param_name} = {value:.4g}" for value in selected_values]

    im = None
    for ax, spec_db, title in zip(axes_flat, specs_db, titles):
        im = ax.imshow(
            spec_db.T,
            origin="lower",
            aspect="auto",
            extent=[t_start, t_end, 0.0, float(freqs[-1])],
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_ylim(0.0, f_max)
        ax.set_title(title)
        ax.set_xlabel("time [s]")
        ax.set_ylabel("frequency [Hz]")

    for ax in axes_flat[n_panels:]:
        ax.axis("off")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes_flat[:n_panels].tolist(), shrink=0.92, pad=0.02)
        cbar.set_label("magnitude [dB, target-referenced]")

    fig.suptitle(
        f"Spectrograms along 1D scan: {source}/DG, {param_name}, T={T_max:.3f}s\n"
        f"STFT: n_fft={n_fft}, hop={hop}; dashed-scan reference={true_value:.4g}",
        y=0.995,
    )
    fig.subplots_adjust(top=0.88, wspace=0.22, hspace=0.30, right=0.92)

    safe_t = f"{T_max:.4f}".replace(".", "p")
    fig_path = os.path.join(
        output_dir,
        f"spectrogram_scan_{source}_{param_name}_T{safe_t}.png",
    )
    fig.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    npz_path = os.path.join(
        output_dir,
        f"spectrogram_scan_{source}_{param_name}_T{safe_t}.npz",
    )
    np.savez(
        npz_path,
        selected_indices=np.asarray(selected_indices, dtype=int),
        selected_values=np.asarray(selected_values, dtype=float),
        true_value=float(true_value),
        target=target_np,
        predictions=np.asarray(predictions, dtype=float),
        snapshot_times=times,
        n_fft=int(n_fft),
        hop=int(hop),
        dynamic_db=float(SPECTROGRAM_DYNAMIC_DB),
    )

    print(f"  spectrogrammes diagnostiques: {fig_path}")
    print(f"  signaux diagnostiques       : {npz_path}")
    return fig_path


def make_scan_losses(
    param_name,
    data_base,
    geometry,
    c,
    target,
    solve_kwargs,
    stft_resolutions,
    stft_dynamic_db,
    stft_allow_padding,
    combo_time_weight,
    combo_spec_weight,
    combo_env_weight,
    env_win,
    env_hop,
):
    def losses(value):
        data = set_param(data_base, param_name, value, GEO_KEYS)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)

        loss_time = loss_fn_signal(
            pred,
            target,
            time_weight=1.0,
            spec_weight=0.0,
        )
        loss_stft = loss_fn_signal(
            pred,
            target,
            time_weight=0.0,
            spec_weight=1.0,
            stft_resolutions=stft_resolutions,
            stft_dynamic_db=stft_dynamic_db,
            stft_allow_padding=stft_allow_padding,
        )
        loss_env = envelope_rms_loss(
            pred,
            target,
            win=env_win,
            hop=env_hop,
        )
        loss_combo = (
            combo_time_weight * loss_time
            + combo_spec_weight * loss_stft
            + combo_env_weight * loss_env
        )

        return loss_time, loss_stft, loss_env, loss_combo

    return jax.jit(losses)


def parse_scan_range(param_name, args):
    if args.scan_range is None:
        if param_name not in SCAN_RANGES:
            raise ValueError(f"Pas de range par defaut pour {param_name}. Utiliser --scan_range min,max.")
        return SCAN_RANGES[param_name]

    parts = [part.strip() for part in args.scan_range.split(",")]
    if len(parts) != 2:
        raise ValueError("--scan_range doit etre au format min,max")
    return float(parts[0]), float(parts[1])


def write_summary_csv(summary, output_dir):
    csv_path = os.path.join(output_dir, "scan_1D_summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    return csv_path


def make_target_for_source(source, params, geometry, c, solve_kwargs, train_params, snapshot_times, args):
    if source == "openwind":
        target, ow_params = make_openwind_target(
            params,
            args.type_S,
            train_params["T_max"],
            snapshot_times,
            args,
        )
        params_dg = params_with_openwind_radiation(params, ow_params, args.type_S)
        print(f"OpenWind alpha={ow_params.get('alpha')}")
        print(f"OpenWind beta={ow_params.get('beta')}")
        return target, params_dg

    data_true = build_physical_data(params, args.type_S)
    target = forward_snapshots(data_true, geometry, c, **solve_kwargs)
    return target, copy.deepcopy(params)


def main():
    args = parse_args()
    root = repo_root()
    output_dir = resolve_project_path(root, args.scan_output_dir)
    os.makedirs(output_dir, exist_ok=True)
    stft_resolutions = parse_stft_resolutions(args.scan_stft_resolutions)
    stft_allow_padding = not args.scan_no_stft_padding
    t_max_values = parse_float_list(args.scan_t_max_values, "scan_t_max_values")

    # Parametres optionnels : pas besoin de modifier parse_args.
    # Si ces arguments n'existent pas, on garde env_weight=0 et le scan
    # se comporte comme avant.
    scan_combo_env_weight = getattr(args, "scan_combo_env_weight", 0.0)
    scan_env_win = getattr(args, "scan_env_win", 256)
    scan_env_hop = getattr(args, "scan_env_hop", 64)

    with open(root / "experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    with open(root / "experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    train_params = solver_params["train"]
    scan_nx = args.scan_nx if args.scan_nx is not None else train_params["Nx"]
    scan_n_snapshot = (
        args.scan_n_snapshot
        if args.scan_n_snapshot is not None
        else train_params["N_snapshot"]
    )
    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    data_ref = build_physical_data(params, args.type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell
    geometry = build_solver_geometry(data_ref, scan_nx, c)
    bc = BC(type="full")

    print("loss: temporelle / STFT / combinaison")
    if args.scan_target_source == "both":
        target_sources = ["dg", "openwind"]
    else:
        target_sources = [args.scan_target_source]

    print(f"target_source: {','.join(target_sources)}")
    print(f"DG discretization: Nx={scan_nx}, N_snapshot={scan_n_snapshot}")
    print(f"STFT resolutions: {stft_resolutions}")
    print(f"STFT dynamic dB : {args.scan_stft_dynamic_db}")
    print(f"STFT padding    : {stft_allow_padding}")
    print(
        "OpenWind l_ele : "
        f"{args.ow_l_ele if args.ow_l_ele is not None else 5.0e-4}"
    )
    print(f"T_max values    : {t_max_values}")
    print(f"Envelope RMS    : win={scan_env_win}, hop={scan_env_hop}")
    print(
        "combo weights  : "
        f"time={args.scan_combo_time_weight}, "
        f"spec={args.scan_combo_spec_weight}, "
        f"env={scan_combo_env_weight}"
    )

    scan_params = [name.strip() for name in args.scan_params.split(",") if name.strip()]
    if not scan_params:
        raise ValueError("La liste --scan_params est vide.")
    summary = []

    for source in target_sources:
        print("\n" + "#" * 80)
        print(f"ETUDE {source.upper()} / DG")
        print("#" * 80)

        source_cases = []
        for T_max in t_max_values:
            dt, nsteps, solve_kwargs = make_solver_data(
                T_max,
                train_params["cfl"],
                scan_nx,
                scan_n_snapshot,
                L_ref,
                c,
                bc,
                phi0,
                y0,
                z0,
            )
            snapshot_times = (solve_kwargs["n_snaps"] + 1) * solve_kwargs["dt"]

            print(
                f"\nPreparation cible {source}/DG: "
                f"T_max={T_max:.4f}, dt={dt:.6e}, nsteps={nsteps}"
            )
            target, params_dg = make_target_for_source(
                source,
                params,
                geometry,
                c,
                solve_kwargs,
                {"T_max": T_max},
                snapshot_times,
                args,
            )

            print("OpenWind alpha =", params_dg["right_bc_params"]["alpha"])
            print("OpenWind beta  =", params_dg["right_bc_params"]["beta"])
            print("OpenWind Zt    =", params_dg["right_bc_params"]["Zt"])

            print("DG alpha =", value_from_params(params_dg, "alpha"))
            print("DG beta  =", value_from_params(params_dg, "beta"))
            print("DG Zt    =", value_from_params(params_dg, "Zt"))

            print("DG gamma =", value_from_params(params_dg, "gamma_final"))
            print("DG zeta  =", value_from_params(params_dg, "zeta"))
            print("DG fr    =", value_from_params(params_dg, "fr"))
            print("DG Qr    =", value_from_params(params_dg, "Qr"))
            source_cases.append(
                {
                    "T_max": T_max,
                    "dt": dt,
                    "nsteps": nsteps,
                    "solve_kwargs": solve_kwargs,
                    "target": target,
                    "params_dg": params_dg,
                    "snapshot_times": snapshot_times,
                }
            )

        for param_name in scan_params:
            if param_name not in PARAM_JSON_PATHS:
                raise ValueError(f"Parametre inconnu: {param_name}")

            vmin, vmax = parse_scan_range(param_name, args)
            values = jnp.linspace(vmin, vmax, args.scan_n)
            values_np = np.asarray(values, dtype=float)

            print("\n" + "=" * 80)
            print(f"SCAN 1D [{source}/DG]: {param_name}")
            print(f"range=[{vmin:.6g}, {vmax:.6g}], n={args.scan_n}")
            print("=" * 80)

            losses_time_all = []
            losses_stft_all = []
            losses_env_all = []
            losses_combo_all = []
            true_values = []
            minima = []

            for case in source_cases:
                T_max = case["T_max"]
                true_value = value_from_params(case["params_dg"], param_name)
                true_values.append(true_value)

                params_scan = copy.deepcopy(case["params_dg"])
                for name in params_scan["trainable"]:
                    params_scan["trainable"][name] = name == param_name
                data_base = build_physical_data(params_scan, args.type_S)
                scan_losses = make_scan_losses(
                    param_name,
                    data_base,
                    geometry,
                    c,
                    case["target"],
                    case["solve_kwargs"],
                    stft_resolutions,
                    args.scan_stft_dynamic_db,
                    stft_allow_padding,
                    0.0,
                    0.5,
                    0.5,
                    scan_env_win,
                    scan_env_hop,
                )

                print(f"\nT_max={T_max:.4f}, true={true_value:.6g}")
                losses_time = []
                losses_stft = []
                losses_env = []
                losses_combo = []
                for idx, value in enumerate(values):
                    loss_time, loss_stft, loss_env, loss_combo = scan_losses(value)
                    loss_time = float(loss_time)
                    loss_stft = float(loss_stft)
                    loss_env = float(loss_env)
                    loss_combo = float(loss_combo)

                    losses_time.append(loss_time)
                    losses_stft.append(loss_stft)
                    losses_env.append(loss_env)
                    losses_combo.append(loss_combo)

                    if idx < 3 or idx % args.scan_print_every == 0 or idx == args.scan_n - 1:
                        print(
                            f"{idx:>4}/{args.scan_n - 1:<4} | "
                            f"{param_name}={float(value):.6g} | "
                            f"time={loss_time:.6e} | "
                            f"stft={loss_stft:.6e} | "
                            f"env={loss_env:.6e} | "
                            f"combo={loss_combo:.6e}"
                        )

                losses_time = np.asarray(losses_time, dtype=float)
                losses_stft = np.asarray(losses_stft, dtype=float)
                losses_env = np.asarray(losses_env, dtype=float)
                losses_combo = np.asarray(losses_combo, dtype=float)

                losses_time_all.append(losses_time)
                losses_stft_all.append(losses_stft)
                losses_env_all.append(losses_env)
                losses_combo_all.append(losses_combo)

                idx_min_time = int(np.argmin(losses_time))
                idx_min_stft = int(np.argmin(losses_stft))
                idx_min_env = int(np.argmin(losses_env))
                idx_min_combo = int(np.argmin(losses_combo))

                value_min_time = float(values_np[idx_min_time])
                value_min_stft = float(values_np[idx_min_stft])
                value_min_env = float(values_np[idx_min_env])
                value_min_combo = float(values_np[idx_min_combo])

                loss_min_time = float(losses_time[idx_min_time])
                loss_min_stft = float(losses_stft[idx_min_stft])
                loss_min_env = float(losses_env[idx_min_env])
                loss_min_combo = float(losses_combo[idx_min_combo])

                rel_error_min_time = abs(value_min_time - true_value) / max(abs(true_value), 1e-12)
                rel_error_min_stft = abs(value_min_stft - true_value) / max(abs(true_value), 1e-12)
                rel_error_min_env = abs(value_min_env - true_value) / max(abs(true_value), 1e-12)
                rel_error_min_combo = abs(value_min_combo - true_value) / max(abs(true_value), 1e-12)

                minima.append(
                    {
                        "time": value_min_time,
                        "stft": value_min_stft,
                        "env": value_min_env,
                        "combo": value_min_combo,
                    }
                )

                summary.append(
                    {
                        "target_source": source,
                        "T_max": T_max,
                        "param": param_name,
                        "true": true_value,
                        "time_min": value_min_time,
                        "time_rel_error_min": rel_error_min_time,
                        "time_loss_min": loss_min_time,
                        "stft_min": value_min_stft,
                        "stft_rel_error_min": rel_error_min_stft,
                        "stft_loss_min": loss_min_stft,
                        "env_min": value_min_env,
                        "env_rel_error_min": rel_error_min_env,
                        "env_loss_min": loss_min_env,
                        "combo_min": value_min_combo,
                        "combo_rel_error_min": rel_error_min_combo,
                        "combo_loss_min": loss_min_combo,
                        "range_min": vmin,
                        "range_max": vmax,
                        "n_scan": args.scan_n,
                        "combo_time_weight": args.scan_combo_time_weight,
                        "combo_spec_weight": args.scan_combo_spec_weight,
                        "combo_env_weight": scan_combo_env_weight,
                        "env_win": scan_env_win,
                        "env_hop": scan_env_hop,
                    }
                )

                # Diagnostic acoustique demande dans le rapport : comparer les
                # spectrogrammes de la cible et de plusieurs points representatifs
                # du scan, notamment de part et d'autre de la rupture la plus forte.
                if param_name in SPECTROGRAM_SCAN_PARAMS:
                    plot_scan_spectrogram_diagnostics(
                        output_dir=output_dir,
                        source=source,
                        param_name=param_name,
                        T_max=T_max,
                        values=values_np,
                        losses_stft=losses_stft,
                        true_value=true_value,
                        target=case["target"],
                        data_base=data_base,
                        geometry=geometry,
                        c=c,
                        solve_kwargs=case["solve_kwargs"],
                        snapshot_times=case["snapshot_times"],
                        stft_resolutions=stft_resolutions,
                    )

            losses_time_all = np.asarray(losses_time_all, dtype=float)
            losses_stft_all = np.asarray(losses_stft_all, dtype=float)
            losses_env_all = np.asarray(losses_env_all, dtype=float)
            losses_combo_all = np.asarray(losses_combo_all, dtype=float)
            true_values = np.asarray(true_values, dtype=float)

            data_path = os.path.join(output_dir, f"scan_{source}_{param_name}.npz")
            np.savez(
                data_path,
                values=values_np,
                t_max_values=np.asarray(t_max_values, dtype=float),
                true_values=true_values,
                losses_time=losses_time_all,
                losses_stft=losses_stft_all,
                losses_env=losses_env_all,
                losses_combo=losses_combo_all,
            )

            fig_path = os.path.join(output_dir, f"scan_{source}_{param_name}.png")
            fig, axes = plt.subplots(4, 1, figsize=(7, 11), sharex=True)
            plot_specs = [
                ("Perte temporelle", losses_time_all, "time"),
                ("Perte STFT", losses_stft_all, "stft"),
                ("Perte enveloppe RMS", losses_env_all, "env"),
                ("Combinaison", losses_combo_all, "combo"),
            ]
            colors = plt.cm.viridis(np.linspace(0.0, 1.0, len(t_max_values)))

            for ax, (title, losses_by_t, min_key) in zip(axes, plot_specs):
                for t_idx, T_max in enumerate(t_max_values):
                    color = colors[t_idx]
                    losses_i = losses_by_t[t_idx]
                    true_value_i = true_values[t_idx]
                    value_min_i = minima[t_idx][min_key]

                    ax.plot(
                        values_np,
                        losses_i,
                        linewidth=1.6,
                        color=color,
                        label=f"T={T_max:.2f}s",
                    )
                    ax.axvline(true_value_i, color=color, linestyle="--", alpha=0.35)
                    ax.axvline(value_min_i, color=color, linestyle=":", alpha=0.85)
                    ax.scatter(
                        [true_value_i],
                        [np.interp(true_value_i, values_np, losses_i)],
                        s=30,
                        color=color,
                    )
                ax.set_ylabel("loss")
                ax.set_title(title)
                ax.legend()
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel(param_name)
            fig.suptitle(f"Scan 1D {source}/DG {param_name}")
            fig.tight_layout()
            fig.savefig(fig_path, dpi=200)
            plt.close(fig)

            print(f"  figure sauvegardee: {fig_path}")
            print(f"  valeurs sauvegardees: {data_path}")

    summary_csv = write_summary_csv(summary, output_dir)

    print("\n=== Resume scans ===")
    for item in summary:
        print(
            f"{item['target_source']:>8}/DG | "
            f"T={item['T_max']:.4f} | "
            f"{item['param']:>12} | true={item['true']:.6g} | "
            f"time_min={item['time_min']:.6g} | "
            f"stft_min={item['stft_min']:.6g} | "
            f"env_min={item['env_min']:.6g} | "
            f"combo_min={item['combo_min']:.6g}"
        )
    print("CSV resume :", summary_csv)


if __name__ == "__main__":
    main()