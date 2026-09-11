#!/usr/bin/env python3
# ======================================================================================
# Performance study of reverse-mode automatic differentiation through the COMPLETE
# differentiable DG forward solver used in the inverse problem.
#
# The benchmark measures, for each mesh size N and for the constant/exponential geometry:
#
#   1. T_forward:
#      compiled evaluation of a scalar MSE objective
#          theta -> p_bell(theta) -> loss
#
#   2. T_backward:
#      isolated reverse-mode pullback obtained with jax.vjp, using the residuals/tape
#      produced by one primal evaluation.
#
#   3. T_value_and_grad:
#      compiled end-to-end evaluation of
#          (loss(theta), grad_theta loss(theta))
#      with jax.value_and_grad. This is the quantity closest to one optimization step
#      before the optimizer update itself.
#
#   4. Ratios:
#          T_backward / T_forward
#          T_value_and_grad / T_forward
#
#   5. Empirical log-log exponents versus the number of DG cells N.
#
# IMPORTANT
# ---------
# - This script benchmarks RK2 because utils.solve.forward_snapshots() is the actual
#   differentiable RK2 forward operator used by the calibration code.
# - The target is generated once with the same DG solver at the reference parameters.
#   It is NOT included in any timed region.
# - A simple time-domain MSE is used by default so that the benchmark measures mainly
#   the cost of differentiating the physical solver rather than the cost of STFTs.
# - The evaluated parameter vector is slightly perturbed from the reference values so
#   that the loss and its gradient are non-zero.
# ======================================================================================

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------------------
# Robust imports
# --------------------------------------------------------------------------------------
try:
    from src.numerics.dg.mesh import (
        create_uniform_nodes_with_ghosts,
        cell_edges_from_nodes,
    )
    from src.physics.bc import BC
    from src.utils.build_physical_data import build_physical_data
    from src.utils.build_solver import build_solver_geometry
    from src.utils.param_func import set_param
    from src.utils.solve import forward_snapshots
except ModuleNotFoundError:
    from numerics.dg.mesh import (
        create_uniform_nodes_with_ghosts,
        cell_edges_from_nodes,
    )
    from physics.bc import BC
    from utils.build_physical_data import build_physical_data
    from utils.build_solver import build_solver_geometry
    from utils.param_func import set_param
    from utils.solve import forward_snapshots


GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")
PARAM_NAMES = ("gamma_final", "kappa", "zeta")


# ======================================================================================
# CLI
# ======================================================================================
def parse_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark forward, reverse-mode backward, and value+gradient costs "
            "for the differentiable complete DG solver."
        )
    )

    parser.add_argument(
        "--simu_config",
        type=str,
        default="experiments/gradient/config/simu.json",
        help="Path to the simulation configuration JSON.",
    )
    parser.add_argument(
        "--param_config",
        type=str,
        default="experiments/gradient/config/param.json",
        help="Path to the physical-parameter JSON.",
    )
    parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 400, 500, 600, 700],
        help="Numbers of DG cells.",
    )
    parser.add_argument(
        "--geometries",
        nargs="+",
        choices=("const", "exp"),
        default=["const", "exp"],
        help="Geometries to benchmark.",
    )
    parser.add_argument(
        "--T_max",
        type=float,
        default=None,
        help="Override the simulation duration from simu.json.",
    )
    parser.add_argument(
        "--CFL",
        type=float,
        default=None,
        help="Override the CFL number from simu.json.",
    )
    parser.add_argument(
        "--N_snapshot",
        type=int,
        default=None,
        help="Override the number of pressure snapshots.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of post-compilation timing repetitions.",
    )
    parser.add_argument(
        "--theta_factors",
        type=float,
        nargs=3,
        default=[0.90, 1.10, 0.90],
        metavar=("F_GAMMA", "F_KAPPA", "F_ZETA"),
        help=(
            "Multiplicative factors applied to the true "
            "[gamma, kappa, zeta] for the benchmark point."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="experiments/performance/results_backward",
        help="Output directory.",
    )

    return parser.parse_args()


# ======================================================================================
# Utilities
# ======================================================================================
def repo_root():
    """
    Infer repository root.

    If this file is placed in src/, parents[1] is the project root.
    Otherwise the current working directory remains a practical fallback.
    """
    path = Path(__file__).resolve()
    if path.parent.name == "src":
        return path.parents[1]
    return Path.cwd().resolve()


def resolve_path(root, value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (root / path).resolve()


def block_tree(tree):
    """Synchronize every JAX array in a PyTree."""
    def block(x):
        if hasattr(x, "block_until_ready"):
            return x.block_until_ready()
        return x

    jax.tree_util.tree_map(block, tree)


def compute_loglog_slope(N, T):
    """Estimate alpha in T ~ C N^alpha."""
    N = np.asarray(N, dtype=float)
    T = np.asarray(T, dtype=float)

    mask = (
        np.isfinite(N)
        & np.isfinite(T)
        & (N > 0.0)
        & (T > 0.0)
    )

    N = N[mask]
    T = T[mask]

    if len(N) < 2:
        return np.nan

    alpha, _ = np.polyfit(
        np.log(N),
        np.log(T),
        1,
    )
    return float(alpha)


def set_trainable_parameters(params):
    """
    gamma, kappa and zeta must be dynamic leaves in PhysicalData so that
    set_param can inject traced JAX values during automatic differentiation.
    """
    params = copy.deepcopy(params)

    dynamic = {
        "gamma_final",
        "kappa",
        "zeta",
    }

    for name in params.get("trainable", {}):
        params["trainable"][name] = name in dynamic

    return params


def extract_solver_params(simu_json):
    """
    Support both configuration layouts encountered in the project:
        solver_params = {...}
    and
        solver_params = {"train": {...}, ...}
    """
    solver_params = simu_json["solver_params"]

    if "train" in solver_params and isinstance(
        solver_params["train"],
        dict,
    ):
        return solver_params["train"]

    return solver_params


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
    """
    Same temporal-grid construction as in the calibration scripts.
    """
    x_nodes, _ = create_uniform_nodes_with_ghosts(
        Nx,
        0.0,
        L_ref,
    )
    x_left, x_right = cell_edges_from_nodes(x_nodes)

    dt = CFL * (x_right[0] - x_left[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = (
        jnp.arange(
            nsteps,
            dtype=jnp.float64,
        )
        * dt
    )

    snapshot_steps = jnp.round(
        jnp.linspace(
            0,
            nsteps - 1,
            N_snapshot,
        )
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


def set_parameter_vector_data(data, theta):
    """Inject [gamma, kappa, zeta] into a PhysicalData PyTree."""
    gamma, kappa, zeta = theta

    data = set_param(
        data,
        "gamma_final",
        gamma,
        GEO_KEYS,
    )
    data = set_param(
        data,
        "kappa",
        kappa,
        GEO_KEYS,
    )
    data = set_param(
        data,
        "zeta",
        zeta,
        GEO_KEYS,
    )

    return data


# ======================================================================================
# Build one benchmark case
# ======================================================================================
def build_case(
    params,
    type_S,
    Nx,
    T_max,
    CFL,
    N_snapshot,
    theta_factors,
):
    params_dynamic = set_trainable_parameters(params)

    # First build a temporary PhysicalData only to evaluate the geometry.
    # Zt is not a trainable leaf in this benchmark, so it must NOT be changed
    # afterwards with set_param()/eqx.tree_at. Instead we compute its
    # geometry-dependent value, write it into the parameter dictionary, and
    # rebuild PhysicalData with the correct static Zt value.
    data_tmp = build_physical_data(
        params_dynamic,
        type_S,
    )

    L = data_tmp.section.L_tube + data_tmp.section.L_bell
    S_star = data_tmp.section(0.0)
    Zt = S_star / data_tmp.section(L)

    params_dynamic = copy.deepcopy(params_dynamic)
    params_dynamic["right_bc_params"]["Zt"] = float(Zt)

    data_base = build_physical_data(
        params_dynamic,
        type_S,
    )

    c = float(params["physics"]["c"])
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    bc = BC(type="full")

    geometry = build_solver_geometry(
        data_base,
        Nx,
        c,
    )

    solve_kwargs = make_solver_data(
        T_max=T_max,
        CFL=CFL,
        Nx=Nx,
        N_snapshot=N_snapshot,
        L_ref=L,
        c=c,
        bc=bc,
        phi0=phi0,
        y0=y0,
        z0=z0,
    )

    theta_true = jnp.asarray(
        [
            data_base.gamma_final,
            data_base.kappa,
            data_base.zeta,
        ],
        dtype=jnp.float64,
    )

    factors = jnp.asarray(
        theta_factors,
        dtype=jnp.float64,
    )

    theta_eval = theta_true * factors

    # ------------------------------------------------------------------
    # Target: generated ONCE and outside all timed regions.
    # ------------------------------------------------------------------
    data_true = set_parameter_vector_data(
        data_base,
        theta_true,
    )

    target = forward_snapshots(
        data_true,
        geometry,
        c,
        **solve_kwargs,
    )
    target = jax.lax.stop_gradient(target)
    block_tree(target)

    # ------------------------------------------------------------------
    # Scalar objective used for the benchmark.
    #
    # A simple MSE deliberately avoids mixing solver AD cost with STFT cost.
    # ------------------------------------------------------------------
    def loss_raw(theta):
        data = set_parameter_vector_data(
            data_base,
            theta,
        )

        pred = forward_snapshots(
            data,
            geometry,
            c,
            **solve_kwargs,
        )

        residual = pred - target

        # Relative normalization makes the scale of the loss less dependent
        # on the absolute pressure magnitude.
        denom = (
            jnp.mean(target**2)
            + jnp.asarray(
                1e-16,
                dtype=jnp.float64,
            )
        )

        return jnp.mean(residual**2) / denom

    return {
        "data_base": data_base,
        "geometry": geometry,
        "solve_kwargs": solve_kwargs,
        "theta_true": theta_true,
        "theta_eval": theta_eval,
        "target": target,
        "loss_raw": loss_raw,
        "dt": float(solve_kwargs["dt"]),
        "nsteps": int(solve_kwargs["nsteps"]),
    }


# ======================================================================================
# Timing helpers
# ======================================================================================
def time_repeated(callable_fn, n_repeats):
    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()

        output = callable_fn()
        block_tree(output)

        times.append(
            time.perf_counter() - start
        )

    return np.asarray(
        times,
        dtype=float,
    )


def stats(values):
    values = np.asarray(
        values,
        dtype=float,
    )

    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values, ddof=0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ======================================================================================
# Benchmark one mesh/geometry
# ======================================================================================
def benchmark_case(
    params,
    type_S,
    Nx,
    T_max,
    CFL,
    N_snapshot,
    theta_factors,
    n_repeats,
):
    case = build_case(
        params=params,
        type_S=type_S,
        Nx=Nx,
        T_max=T_max,
        CFL=CFL,
        N_snapshot=N_snapshot,
        theta_factors=theta_factors,
    )

    theta_eval = case["theta_eval"]
    loss_raw = case["loss_raw"]

    # ------------------------------------------------------------------
    # A) Compiled forward scalar objective
    # ------------------------------------------------------------------
    loss_jit = jax.jit(loss_raw)

    start = time.perf_counter()
    loss_first = loss_jit(theta_eval)
    block_tree(loss_first)
    forward_first_call = time.perf_counter() - start

    forward_times = time_repeated(
        lambda: loss_jit(theta_eval),
        n_repeats,
    )

    # ------------------------------------------------------------------
    # B) Compiled full loss + gradient
    #
    # This is the operational quantity corresponding to what an optimizer
    # needs before applying its parameter update.
    # ------------------------------------------------------------------
    value_and_grad_jit = jax.jit(
        jax.value_and_grad(loss_raw)
    )

    start = time.perf_counter()
    vg_first = value_and_grad_jit(
        theta_eval
    )
    block_tree(vg_first)
    value_grad_first_call = (
        time.perf_counter() - start
    )

    value_grad_times = time_repeated(
        lambda: value_and_grad_jit(
            theta_eval
        ),
        n_repeats,
    )

    # ------------------------------------------------------------------
    # C) Isolated reverse pullback
    #
    # jax.vjp performs one primal evaluation and returns a pullback closure.
    # We keep the residuals produced by that primal pass fixed, JIT the
    # pullback, and then time only the reverse propagation.
    #
    # This is intentionally different from value_and_grad:
    #   - pullback timing = isolated reverse pass on an existing tape;
    #   - value_and_grad = fresh forward + backward, as in optimization.
    # ------------------------------------------------------------------
    primal_value, pullback = jax.vjp(
        loss_raw,
        theta_eval,
    )
    block_tree(primal_value)

    cotangent = jnp.asarray(
        1.0,
        dtype=jnp.float64,
    )

    pullback_jit = jax.jit(
        pullback
    )

    start = time.perf_counter()
    backward_first = pullback_jit(
        cotangent
    )
    block_tree(backward_first)
    backward_first_call = (
        time.perf_counter() - start
    )

    backward_times = time_repeated(
        lambda: pullback_jit(
            cotangent
        ),
        n_repeats,
    )

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    fwd = stats(forward_times)
    bwd = stats(backward_times)
    vg = stats(value_grad_times)

    # A second estimate of incremental backward overhead:
    # combined minus forward. It can fluctuate slightly because the two
    # executables are benchmarked independently, so keep it descriptive.
    backward_increment_median = max(
        vg["median"] - fwd["median"],
        0.0,
    )

    grad = vg_first[1]
    grad_norm = float(
        jnp.linalg.norm(grad)
    )

    loss_value = float(
        vg_first[0]
    )

    return {
        "geometry": type_S,
        "Nx": int(Nx),
        "dt": case["dt"],
        "nsteps": case["nsteps"],
        "N_snapshot": int(N_snapshot),

        "loss_value": loss_value,
        "gradient_norm": grad_norm,

        "forward_first_call_s": forward_first_call,
        "backward_first_call_s": backward_first_call,
        "value_grad_first_call_s": value_grad_first_call,

        "forward_mean_s": fwd["mean"],
        "forward_median_s": fwd["median"],
        "forward_std_s": fwd["std"],

        "backward_mean_s": bwd["mean"],
        "backward_median_s": bwd["median"],
        "backward_std_s": bwd["std"],

        "value_grad_mean_s": vg["mean"],
        "value_grad_median_s": vg["median"],
        "value_grad_std_s": vg["std"],

        "backward_increment_median_s": backward_increment_median,

        "backward_forward_ratio": (
            bwd["median"] / fwd["median"]
        ),
        "value_grad_forward_ratio": (
            vg["median"] / fwd["median"]
        ),

        "forward_per_step_us": (
            1e6 * fwd["median"]
            / case["nsteps"]
        ),
        "backward_per_step_us": (
            1e6 * bwd["median"]
            / case["nsteps"]
        ),
        "value_grad_per_step_us": (
            1e6 * vg["median"]
            / case["nsteps"]
        ),
    }


# ======================================================================================
# CSV
# ======================================================================================
def save_csv(rows, path):
    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


# ======================================================================================
# Plotting
# ======================================================================================
def plot_summary(
    rows,
    output_path,
):
    """
    Three-panel report-oriented figure:
      (a) forward / backward / value+grad times,
      (b) overhead ratios,
      (c) per-time-step costs.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(8.2, 10.5),
        sharex=True,
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

    markers = {
        "const": "o",
        "exp": "s",
    }

    # We intentionally use line styles to distinguish computational phases.
    phase_specs = [
        (
            "forward_median_s",
            "Forward loss",
            "-",
        ),
        (
            "backward_median_s",
            "Backward pullback",
            "--",
        ),
        (
            "value_grad_median_s",
            "Forward + backward",
            ":",
        ),
    ]

    # ------------------------------------------------------------------
    # (a) Wall time
    # ------------------------------------------------------------------
    ax = axes[0]

    for geometry in geometries:
        subset = sorted(
            [
                row
                for row in rows
                if row["geometry"] == geometry
            ],
            key=lambda row: row["Nx"],
        )

        N = np.asarray(
            [row["Nx"] for row in subset],
            dtype=float,
        )

        for key, phase_label, linestyle in phase_specs:
            values = np.asarray(
                [row[key] for row in subset],
                dtype=float,
            )

            alpha = compute_loglog_slope(
                N,
                values,
            )

            ax.plot(
                N,
                values,
                marker=markers[geometry],
                linestyle=linestyle,
                label=(
                    f"{phase_label}, {geometry} "
                    rf"($\alpha={alpha:.2f}$)"
                ),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylabel("Median wall time (s)")
    ax.set_title(
        "(a) Reverse-mode differentiation cost"
    )
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend(
        ncol=2,
        fontsize=8,
    )

    # ------------------------------------------------------------------
    # (b) Ratios
    # ------------------------------------------------------------------
    ax = axes[1]

    for geometry in geometries:
        subset = sorted(
            [
                row
                for row in rows
                if row["geometry"] == geometry
            ],
            key=lambda row: row["Nx"],
        )

        N = np.asarray(
            [row["Nx"] for row in subset],
            dtype=float,
        )

        ratio_backward = np.asarray(
            [
                row["backward_forward_ratio"]
                for row in subset
            ],
            dtype=float,
        )

        ratio_total = np.asarray(
            [
                row["value_grad_forward_ratio"]
                for row in subset
            ],
            dtype=float,
        )

        ax.plot(
            N,
            ratio_backward,
            marker=markers[geometry],
            linestyle="--",
            label=(
                r"$T_B/T_F$, "
                + geometry
            ),
        )

        ax.plot(
            N,
            ratio_total,
            marker=markers[geometry],
            linestyle="-",
            label=(
                r"$T_{F+B}/T_F$, "
                + geometry
            ),
        )

    ax.axhline(
        1.0,
        linestyle=":",
        linewidth=1.0,
    )

    ax.set_xscale("log")
    ax.set_ylabel("Relative computational cost")
    ax.set_title(
        "(b) Reverse-mode overhead relative to the forward pass"
    )
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend(
        ncol=2,
        fontsize=8,
    )

    # ------------------------------------------------------------------
    # (c) Cost normalized by the number of physical time steps
    # ------------------------------------------------------------------
    ax = axes[2]

    per_step_specs = [
        (
            "forward_per_step_us",
            "Forward",
            "-",
        ),
        (
            "backward_per_step_us",
            "Backward",
            "--",
        ),
        (
            "value_grad_per_step_us",
            "Forward + backward",
            ":",
        ),
    ]

    for geometry in geometries:
        subset = sorted(
            [
                row
                for row in rows
                if row["geometry"] == geometry
            ],
            key=lambda row: row["Nx"],
        )

        N = np.asarray(
            [row["Nx"] for row in subset],
            dtype=float,
        )

        for key, label_phase, linestyle in per_step_specs:
            values = np.asarray(
                [row[key] for row in subset],
                dtype=float,
            )

            ax.plot(
                N,
                values,
                marker=markers[geometry],
                linestyle=linestyle,
                label=(
                    f"{label_phase}, "
                    f"{geometry}"
                ),
            )

    ax.set_xscale("log")
    ax.set_xlabel("Number of DG cells $N$")
    ax.set_ylabel(
        r"Median time per physical step ($\mu$s)"
    )
    ax.set_title(
        "(c) Cost normalized by the CFL-imposed time-step count"
    )
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.suptitle(
        "Performance of reverse-mode automatic differentiation",
        fontsize=14,
    )

    fig.tight_layout(
        rect=(0, 0, 1, 0.97)
    )

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_scaling_only(
    rows,
    output_path,
):
    fig, ax = plt.subplots(
        figsize=(8.0, 5.2)
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

    markers = {
        "const": "o",
        "exp": "s",
    }

    phase_specs = [
        (
            "forward_median_s",
            "Forward",
            "-",
        ),
        (
            "backward_median_s",
            "Backward",
            "--",
        ),
        (
            "value_grad_median_s",
            "Forward + backward",
            ":",
        ),
    ]

    for geometry in geometries:
        subset = sorted(
            [
                row
                for row in rows
                if row["geometry"] == geometry
            ],
            key=lambda row: row["Nx"],
        )

        N = np.asarray(
            [row["Nx"] for row in subset],
            dtype=float,
        )

        for key, phase, linestyle in phase_specs:
            values = np.asarray(
                [row[key] for row in subset],
                dtype=float,
            )

            alpha = compute_loglog_slope(
                N,
                values,
            )

            ax.plot(
                N,
                values,
                marker=markers[geometry],
                linestyle=linestyle,
                label=(
                    f"{phase}, {geometry} "
                    rf"($\alpha={alpha:.2f}$)"
                ),
            )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(
        "Number of DG cells $N$"
    )
    ax.set_ylabel(
        "Median wall time (s)"
    )
    ax.set_title(
        "Forward and reverse-mode computational scaling"
    )
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend(
        ncol=2,
        fontsize=8,
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)


# ======================================================================================
# Main
# ======================================================================================
def main():
    args = parse_cli()

    root = repo_root()

    simu_path = resolve_path(
        root,
        args.simu_config,
    )
    param_path = resolve_path(
        root,
        args.param_config,
    )
    output_dir = resolve_path(
        root,
        args.output_dir,
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        simu_path,
        "r",
    ) as file:
        simu_json = json.load(file)

    with open(
        param_path,
        "r",
    ) as file:
        params = json.load(file)

    solver_params = extract_solver_params(
        simu_json
    )

    T_max = (
        float(args.T_max)
        if args.T_max is not None
        else float(solver_params["T_max"])
    )

    CFL = (
        float(args.CFL)
        if args.CFL is not None
        else float(solver_params["cfl"])
    )

    N_snapshot = (
        int(args.N_snapshot)
        if args.N_snapshot is not None
        else int(solver_params["N_snapshot"])
    )

    print("=" * 96)
    print("REVERSE-MODE PERFORMANCE STUDY")
    print("=" * 96)
    print(f"Device             : {jax.devices()[0]}")
    print("Integrator         : RK2 (forward_snapshots)")
    print(f"Geometries         : {args.geometries}")
    print(f"Nx                 : {args.Nx}")
    print(f"T_max              : {T_max}")
    print(f"CFL                : {CFL}")
    print(f"N_snapshot         : {N_snapshot}")
    print(f"Timing repetitions : {args.n_repeats}")
    print(
        "theta factors      : "
        f"{args.theta_factors}"
    )
    print("Loss               : relative time-domain MSE")
    print("=" * 96)

    rows = []

    for geometry in args.geometries:
        print(
            f"\n{'=' * 96}\n"
            f"GEOMETRY: {geometry}\n"
            f"{'=' * 96}"
        )

        for Nx in args.Nx:
            print(
                f"\n--- {geometry} | Nx={Nx} ---"
            )

            result = benchmark_case(
                params=params,
                type_S=geometry,
                Nx=int(Nx),
                T_max=T_max,
                CFL=CFL,
                N_snapshot=N_snapshot,
                theta_factors=args.theta_factors,
                n_repeats=args.n_repeats,
            )

            rows.append(result)

            print(
                f"dt                  = "
                f"{result['dt']:.6e} s"
            )
            print(
                f"nsteps              = "
                f"{result['nsteps']}"
            )
            print(
                f"loss                = "
                f"{result['loss_value']:.6e}"
            )
            print(
                f"||grad||            = "
                f"{result['gradient_norm']:.6e}"
            )
            print(
                f"forward median      = "
                f"{result['forward_median_s']:.6f} s"
            )
            print(
                f"backward median     = "
                f"{result['backward_median_s']:.6f} s"
            )
            print(
                f"value+grad median    = "
                f"{result['value_grad_median_s']:.6f} s"
            )
            print(
                f"T_B / T_F           = "
                f"{result['backward_forward_ratio']:.3f}"
            )
            print(
                f"T_(F+B) / T_F       = "
                f"{result['value_grad_forward_ratio']:.3f}"
            )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    csv_path = (
        output_dir
        / "performance_backward_rk2.csv"
    )
    save_csv(
        rows,
        csv_path,
    )

    summary_path = (
        output_dir
        / "performance_backward_summary_rk2.png"
    )
    plot_summary(
        rows,
        summary_path,
    )

    scaling_path = (
        output_dir
        / "performance_backward_scaling_rk2.png"
    )
    plot_scaling_only(
        rows,
        scaling_path,
    )

    # ------------------------------------------------------------------
    # Print empirical exponents
    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("EMPIRICAL LOG-LOG EXPONENTS")
    print("=" * 96)

    for geometry in args.geometries:
        subset = sorted(
            [
                row
                for row in rows
                if row["geometry"] == geometry
            ],
            key=lambda row: row["Nx"],
        )

        N = np.asarray(
            [row["Nx"] for row in subset],
            dtype=float,
        )

        alpha_f = compute_loglog_slope(
            N,
            [
                row["forward_median_s"]
                for row in subset
            ],
        )

        alpha_b = compute_loglog_slope(
            N,
            [
                row["backward_median_s"]
                for row in subset
            ],
        )

        alpha_vg = compute_loglog_slope(
            N,
            [
                row["value_grad_median_s"]
                for row in subset
            ],
        )

        mean_ratio_b = float(
            np.mean(
                [
                    row["backward_forward_ratio"]
                    for row in subset
                ]
            )
        )

        mean_ratio_vg = float(
            np.mean(
                [
                    row["value_grad_forward_ratio"]
                    for row in subset
                ]
            )
        )

        print(
            f"{geometry:>8s}: "
            f"alpha_F={alpha_f:.3f}, "
            f"alpha_B={alpha_b:.3f}, "
            f"alpha_F+B={alpha_vg:.3f}, "
            f"mean B/F={mean_ratio_b:.3f}, "
            f"mean (F+B)/F={mean_ratio_vg:.3f}"
        )

    print("\nFiles written:")
    print(csv_path)
    print(summary_path)
    print(scaling_path)


if __name__ == "__main__":
    main()