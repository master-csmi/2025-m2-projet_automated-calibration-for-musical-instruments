#!/usr/bin/env python3
# ======================================================================================
# Performance study of the COMPLETE differentiable DG forward solver
#
# Benchmarked model:
#   - variable-section acoustic PDE (DG P1)
#   - nonlinear reed boundary condition
#   - reed oscillator
#   - radiation boundary condition at the bell
#   - mouth-pressure attack
#   - snapshot extraction
#   - reconstruction of p(x,t) at the bell
#
# Geometries:
#   - constant cross-section ("const")
#   - exponential bell ("exp")
#
# The benchmark separates:
#   1) setup/precomputation time,
#   2) first forward call (includes possible JAX compilation),
#   3) compiled forward execution time, averaged over several repetitions.
#
# IMPORTANT:
# OpenWind is NOT run here. This script measures only the DG forward solver.
# ======================================================================================

import os
os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import argparse
import csv
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------------------
# Robust imports: works either from project root (src.*) or from inside src/
# --------------------------------------------------------------------------------------
try:
    from src.numerics.dg.mesh import (
        create_uniform_nodes_with_ghosts,
        cell_edges_from_nodes,
    )
    from src.numerics.dg.mass_matrix import local_mass_inv_system
    from src.numerics.time_integrators.euler import time_integrate_euler
    from src.numerics.time_integrators.rk2 import time_integrate_rk2

    from src.utils.reconstruction import reconstruct_system
    from src.utils.util_func import precompute_S_quad
    from src.utils.param_func import set_param
    from src.utils.build_physical_data import build_physical_data

    from src.physics.bc import BC
    from src.physics.init_func import init_func_const
    from src.physics.mouth_pressure import pressure_at_mouth_alexis

except ModuleNotFoundError:
    from numerics.dg.mesh import (
        create_uniform_nodes_with_ghosts,
        cell_edges_from_nodes,
    )
    from numerics.dg.mass_matrix import local_mass_inv_system
    from numerics.time_integrators.euler import time_integrate_euler
    from numerics.time_integrators.rk2 import time_integrate_rk2

    from utils.reconstruction import reconstruct_system
    from utils.util_func import precompute_S_quad
    from utils.param_func import set_param
    from utils.build_physical_data import build_physical_data

    from physics.bc import BC
    from physics.init_func import init_func_const
    from physics.mouth_pressure import pressure_at_mouth_alexis


GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")


# ======================================================================================
# Command line
# ======================================================================================
def parse_cli():
    parser = argparse.ArgumentParser(
        description=(
            "Performance study of the complete DG + reed + radiation forward solver."
        )
    )

    parser.add_argument(
        "--simu_config",
        type=str,
        default="../experiments/pressure_at_bell/config/simu.json",
        help="Path to simu.json.",
    )
    parser.add_argument(
        "--param_config",
        type=str,
        default="../experiments/pressure_at_bell/config/param.json",
        help="Path to param.json.",
    )
    parser.add_argument(
        "--method",
        choices=("euler", "rk2"),
        default="rk2",
        help="Time integration method.",
    )
    parser.add_argument(
        "--Nx",
        type=int,
        nargs="+",
        default=[50, 100, 150, 200, 300, 400, 500, 600, 800, 1000],
        help="Numbers of DG cells to benchmark.",
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
        help="Override simulation duration from simu.json.",
    )
    parser.add_argument(
        "--CFL",
        type=float,
        default=None,
        help="Override CFL from simu.json.",
    )
    parser.add_argument(
        "--N_snapshot",
        type=int,
        default=None,
        help="Override number of snapshots from simu.json.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=5,
        help="Number of timed compiled forward runs per configuration.",
    )
    parser.add_argument(
        "--nq",
        type=int,
        default=2,
        help="Number of quadrature points used for S(x) precomputation.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../experiments/performance/results",
        help="Directory for CSV and figures.",
    )
    parser.add_argument(
        "--max_reconstruction_points",
        type=int,
        default=1000,
        help=(
            "Number of spatial points used for snapshot reconstruction. "
            "This reproduces the pressure_at_bell forward post-processing."
        ),
    )

    return parser.parse_args()


# ======================================================================================
# Small utilities
# ======================================================================================
def block_tree(tree):
    """Force completion of all JAX computations contained in a PyTree."""
    jax.tree_util.tree_map(
        lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x,
        tree,
    )


def compute_loglog_slope(N, T):
    """
    Estimate alpha in T ~ C N^alpha by least squares in log-log coordinates.
    """
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

    alpha, _ = np.polyfit(np.log(N), np.log(T), 1)
    return float(alpha)


def resolve_path(path_string):
    return Path(path_string).expanduser().resolve()


# ======================================================================================
# Build one complete forward problem
# ======================================================================================
def build_forward_case(
    params,
    type_S,
    Nx,
    T_max,
    CFL,
    N_snapshot,
    nq,
):
    """
    Build exactly the objects needed by the complete forward solver.

    This part is intentionally outside the timed repeated forward execution.
    Its wall time is nevertheless measured separately.
    """
    setup_start = time.perf_counter()

    data = build_physical_data(params, type_S)

    c = float(params["physics"]["c"])
    phi0 = params["physics"]["phi0"]

    initial_conditions_reed = params["init_cond_reed"]
    y0 = initial_conditions_reed["y0"]
    z0 = initial_conditions_reed["y_dot0"]

    L = data.L_tube + data.L_bell

    # ------------------------------------------------------------------
    # Geometry-dependent scaling / radiation impedance scaling
    # ------------------------------------------------------------------
    S_star = data.section(0.0)
    Zt = S_star / data.section(L)

    # Keep alpha and beta from param.json, but use the geometry-consistent Zt.
    data = set_param(
        data,
        "Zt",
        jnp.asarray(Zt, dtype=jnp.float64),
        GEO_KEYS,
    )

    # ------------------------------------------------------------------
    # Full boundary conditions: reed on the left, radiation on the right
    # ------------------------------------------------------------------
    bc = BC(type="full")

    def p0(x):
        return init_func_const(x, L)

    def v0(x):
        return 0.0

    # ------------------------------------------------------------------
    # DG mesh and geometry quantities
    # ------------------------------------------------------------------
    x_nodes, _ = create_uniform_nodes_with_ghosts(
        Nx,
        0.0,
        L,
    )
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    S_quad = precompute_S_quad(
        data.section,
        xLs,
        xRs,
        nq=nq,
    )

    S_ext = S_nodes
    S_bc = (
        S_cells
        .at[0].set(S_nodes[0])
        .at[-1].set(S_nodes[-1])
    )

    Mp_inv, Mv_inv = jax.vmap(
        local_mass_inv_system,
        in_axes=0,
    )(hs)

    # ------------------------------------------------------------------
    # Initial DG state.
    #
    # This is the same initialization actually used in pressure_at_bell.py
    # after the projected state is overwritten there.
    # ------------------------------------------------------------------
    u0 = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.asarray(
                        [
                            S_nodes[i] / (c * S_star) * p0(xLs[i]),
                            S_nodes[i + 1] / (c * S_star) * p0(xRs[i]),
                        ],
                        dtype=jnp.float64,
                    ),
                    jnp.asarray(
                        [
                            S_star / (c * S_nodes[i]) * v0(xLs[i]),
                            S_star / (c * S_nodes[i + 1]) * v0(xRs[i]),
                        ],
                        dtype=jnp.float64,
                    ),
                ]
            )
            for i in range(Nx)
        ],
        axis=0,
    )

    # ------------------------------------------------------------------
    # CFL time step
    # ------------------------------------------------------------------
    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max / dt))

    n_snaps = jnp.round(
        jnp.linspace(
            0,
            nsteps - 1,
            N_snapshot,
        )
    ).astype(jnp.int32)

    t_solver = jnp.arange(nsteps) * dt

    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_solver,
    )

    setup_end = time.perf_counter()

    return {
        "data": data,
        "c": c,
        "L": L,
        "phi0": phi0,
        "y0": y0,
        "z0": z0,
        "bc": bc,
        "x_nodes": x_nodes,
        "xLs": xLs,
        "xRs": xRs,
        "Mp_inv": Mp_inv,
        "Mv_inv": Mv_inv,
        "u0": u0,
        "S_star": S_star,
        "S_bc": S_bc,
        "S_quad": S_quad,
        "S_ext": S_ext,
        "dt": dt,
        "nsteps": nsteps,
        "n_snaps": n_snaps,
        "gamma_t": gamma_t,
        "setup_time": setup_end - setup_start,
    }


# ======================================================================================
# Complete forward
# ======================================================================================
def integrate_complete(case, method):
    """
    Run the complete coupled PDE/reed/radiation time integration.
    """
    common_args = (
        case["u0"],
        case["x_nodes"],
        case["c"],
        case["dt"],
        case["nsteps"],
        case["Mp_inv"],
        case["Mv_inv"],
        case["bc"],
        case["phi0"],
        case["y0"],
        case["z0"],
        case["data"],
    )

    common_kwargs = dict(
        S_cells=case["S_bc"],
        S_star=case["S_star"],
        S_quad=case["S_quad"],
        S_ext=case["S_ext"],
        snapshot_steps=case["n_snaps"],
        gamma_target=case["gamma_t"],
    )

    if method == "euler":
        return time_integrate_euler(
            *common_args,
            **common_kwargs,
        )

    return time_integrate_rk2(
        *common_args,
        **common_kwargs,
    )


def make_reconstruction_function(case, n_x_plot):
    """
    Reconstruct pressure and velocity snapshots exactly as in pressure_at_bell.py.
    """
    x_plot = jnp.linspace(
        0.0,
        case["L"],
        n_x_plot,
    )

    data = case["data"]
    x_nodes = case["x_nodes"]
    c = case["c"]
    S_star = case["S_star"]

    @jax.jit
    def reconstruct_all_snaps(u_tilde_snaps):
        return jax.vmap(
            lambda u_T: reconstruct_system(
                u_T,
                x_nodes,
                x_plot,
                data.section,
                c,
                S_star,
            )
        )(u_tilde_snaps)

    return reconstruct_all_snaps


def complete_forward_output(case, method, reconstruct_all_snaps):
    """
    Run the complete forward and return the observable bell pressure.

    The returned pressure is the same type of quantity ultimately extracted
    in pressure_at_bell.py:
        time integration -> snapshots -> reconstruction -> p(L,t).
    """
    result = integrate_complete(
        case,
        method,
    )

    (
        u_tilde,
        phi,
        y,
        y_dot,
        u_tilde_snaps,
        phi_snaps,
        y_snaps,
        z_snaps,
    ) = result

    p_all, v_all = reconstruct_all_snaps(
        u_tilde_snaps
    )

    p_bell = p_all[:, -1]

    # Keep a few outputs so that JAX cannot discard relevant computations
    # and so that the timing truly corresponds to the coupled forward.
    return (
        p_bell,
        y_snaps,
        phi_snaps,
        u_tilde,
        phi,
        y,
        y_dot,
    )


# ======================================================================================
# Benchmark one configuration
# ======================================================================================
def benchmark_case(
    params,
    type_S,
    Nx,
    T_max,
    CFL,
    N_snapshot,
    nq,
    method,
    n_repeats,
    n_x_plot,
):
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    case = build_forward_case(
        params=params,
        type_S=type_S,
        Nx=Nx,
        T_max=T_max,
        CFL=CFL,
        N_snapshot=N_snapshot,
        nq=nq,
    )

    reconstruct_all_snaps = make_reconstruction_function(
        case,
        n_x_plot,
    )

    # ------------------------------------------------------------------
    # First call:
    # includes any JAX tracing/compilation triggered by this configuration.
    # ------------------------------------------------------------------
    first_start = time.perf_counter()

    first_output = complete_forward_output(
        case,
        method,
        reconstruct_all_snaps,
    )
    block_tree(first_output)

    first_call_time = time.perf_counter() - first_start

    # ------------------------------------------------------------------
    # Compiled execution:
    # repeat the exact same forward with the same shapes.
    # ------------------------------------------------------------------
    execution_times = []

    for _ in range(n_repeats):
        start = time.perf_counter()

        output = complete_forward_output(
            case,
            method,
            reconstruct_all_snaps,
        )
        block_tree(output)

        execution_times.append(
            time.perf_counter() - start
        )

    execution_times = np.asarray(
        execution_times,
        dtype=float,
    )

    execution_mean = float(np.mean(execution_times))
    execution_median = float(np.median(execution_times))
    execution_std = float(np.std(execution_times, ddof=0))
    execution_min = float(np.min(execution_times))
    execution_max = float(np.max(execution_times))

    nsteps = int(case["nsteps"])

    return {
        "geometry": type_S,
        "Nx": int(Nx),
        "dt": float(case["dt"]),
        "nsteps": nsteps,
        "N_snapshot": int(N_snapshot),
        "setup_time_s": float(case["setup_time"]),
        "first_call_time_s": float(first_call_time),
        "execution_mean_s": execution_mean,
        "execution_median_s": execution_median,
        "execution_std_s": execution_std,
        "execution_min_s": execution_min,
        "execution_max_s": execution_max,

        # Average computational cost per physical time step.
        "time_per_step_mean_s": execution_mean / nsteps,
        "time_per_step_median_s": execution_median / nsteps,
        "time_per_step_min_s": execution_min / nsteps,
        "time_per_step_max_s": execution_max / nsteps,
    }


# ======================================================================================
# Results
# ======================================================================================
def save_csv(rows, path):
    fieldnames = [
        "geometry",
        "Nx",
        "dt",
        "nsteps",
        "N_snapshot",
        "setup_time_s",
        "first_call_time_s",
        "execution_mean_s",
        "execution_median_s",
        "execution_std_s",
        "execution_min_s",
        "execution_max_s",
        "time_per_step_mean_s",
        "time_per_step_median_s",
        "time_per_step_min_s",
        "time_per_step_max_s",
    ]

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


def plot_scaling(rows, output_path, method):
    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

    all_N = np.asarray(
        sorted({row["Nx"] for row in rows}),
        dtype=float,
    )

    slope_by_geometry = {}

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
        T = np.asarray(
            [row["execution_median_s"] for row in subset],
            dtype=float,
        )

        # Ignore the smallest point in the fit if enough points are available.
        if len(N) >= 4:
            slope = compute_loglog_slope(
                N[1:],
                T[1:],
            )
        else:
            slope = compute_loglog_slope(
                N,
                T,
            )

        slope_by_geometry[geometry] = slope

        ax.plot(
            N,
            T,
            marker="o",
            label=(
                f"{geometry} "
                rf"($\alpha={slope:.2f}$)"
            ),
        )

    # N^2 reference curve anchored at the median of the largest available N.
    if len(all_N) > 0:
        N_ref = float(all_N[-1])

        ref_times = [
            row["execution_median_s"]
            for row in rows
            if row["Nx"] == int(N_ref)
        ]

        if ref_times:
            T_ref = float(np.mean(ref_times))

            N_line = np.geomspace(
                all_N.min(),
                all_N.max(),
                300,
            )
            T_line = (
                T_ref
                * (N_line / N_ref) ** 2
            )

            ax.plot(
                N_line,
                T_line,
                linestyle="--",
                label=r"$N^2$ reference",
            )

    ax.set_xlabel("Number of DG cells $N$")
    ax.set_ylabel("Compiled forward wall time (s)")
    ax.set_title(
        f"Complete forward-solver scaling ({method.upper()})"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)

    return slope_by_geometry



def plot_time_per_step(rows, output_path, method):
    """
    Plot the median compiled wall time per physical time step.

    Since the CFL condition makes nsteps roughly proportional to N,
    this figure helps distinguish the growth in the number of time
    steps from the computational cost of one individual step.
    """
    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

    slope_by_geometry = {}

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
        T_step = np.asarray(
            [row["time_per_step_median_s"] for row in subset],
            dtype=float,
        )

        if len(N) >= 4:
            slope = compute_loglog_slope(
                N[1:],
                T_step[1:],
            )
        else:
            slope = compute_loglog_slope(
                N,
                T_step,
            )

        slope_by_geometry[geometry] = slope

        ax.plot(
            N,
            1e6 * T_step,
            marker="o",
            label=(
                f"{geometry} "
                rf"($\alpha_{{step}}={slope:.2f}$)"
            ),
        )

    ax.set_xlabel("Number of DG cells $N$")
    ax.set_ylabel(r"Median wall time per time step ($\mu$s)")
    ax.set_title(
        f"Cost per physical time step ({method.upper()})"
    )
    ax.set_xscale("log")
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)

    return slope_by_geometry


def plot_nsteps_scaling(rows, output_path, method):
    """
    Plot the number of time steps versus N.

    With a fixed CFL number and fixed physical duration, one expects
    nsteps to scale linearly with the number of cells.
    """
    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

    all_N = np.asarray(
        sorted({row["Nx"] for row in rows}),
        dtype=float,
    )

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
        nsteps = np.asarray(
            [row["nsteps"] for row in subset],
            dtype=float,
        )

        slope = compute_loglog_slope(
            N,
            nsteps,
        )

        ax.plot(
            N,
            nsteps,
            marker="o",
            label=(
                f"{geometry} "
                rf"($\alpha={slope:.2f}$)"
            ),
        )

    if len(all_N) > 0:
        N_ref = all_N[-1]

        ref_steps = [
            row["nsteps"]
            for row in rows
            if row["Nx"] == int(N_ref)
        ]

        if ref_steps:
            steps_ref = float(np.mean(ref_steps))
            N_line = np.geomspace(
                all_N.min(),
                all_N.max(),
                300,
            )
            step_line = (
                steps_ref
                * (N_line / N_ref)
            )

            ax.plot(
                N_line,
                step_line,
                linestyle="--",
                label=r"$N$ reference",
            )

    ax.set_xlabel("Number of DG cells $N$")
    ax.set_ylabel("Number of physical time steps")
    ax.set_title(
        f"Time-step count imposed by CFL ({method.upper()})"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)


def plot_forward_decomposition(rows, output_path, method):
    """
    Combined diagnostic figure:
      top    : total compiled forward time,
      middle : number of time steps,
      bottom : wall time per step.

    This figure is useful for explaining why the measured total scaling
    may differ from the naive asymptotic N^2 estimate on the tested range.
    """
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.5, 10.0),
        sharex=True,
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

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

        total_time = np.asarray(
            [row["execution_median_s"] for row in subset],
            dtype=float,
        )

        nsteps = np.asarray(
            [row["nsteps"] for row in subset],
            dtype=float,
        )

        per_step_us = 1e6 * np.asarray(
            [row["time_per_step_median_s"] for row in subset],
            dtype=float,
        )

        axes[0].plot(
            N,
            total_time,
            marker="o",
            label=geometry,
        )

        axes[1].plot(
            N,
            nsteps,
            marker="o",
            label=geometry,
        )

        axes[2].plot(
            N,
            per_step_us,
            marker="o",
            label=geometry,
        )

    axes[0].set_ylabel("Forward wall time (s)")
    axes[0].set_yscale("log")
    axes[0].set_title(
        f"Forward performance decomposition ({method.upper()})"
    )

    axes[1].set_ylabel("Number of time steps")
    axes[1].set_yscale("log")

    axes[2].set_ylabel(r"Time per step ($\mu$s)")
    axes[2].set_xlabel("Number of DG cells $N$")

    for ax in axes:
        ax.set_xscale("log")
        ax.grid(
            True,
            which="both",
            alpha=0.3,
        )
        ax.legend()

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)

def plot_first_vs_compiled(rows, output_path, method):
    fig, ax = plt.subplots(
        figsize=(7.0, 5.0)
    )

    geometries = sorted(
        {row["geometry"] for row in rows}
    )

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
        first = np.asarray(
            [row["first_call_time_s"] for row in subset],
            dtype=float,
        )
        compiled = np.asarray(
            [row["execution_median_s"] for row in subset],
            dtype=float,
        )

        ax.plot(
            N,
            first,
            marker="o",
            linestyle="--",
            label=f"{geometry}: first call",
        )

        ax.plot(
            N,
            compiled,
            marker="s",
            linestyle="-",
            label=f"{geometry}: compiled",
        )

    ax.set_xlabel("Number of DG cells $N$")
    ax.set_ylabel("Wall time (s)")
    ax.set_title(
        f"First call vs compiled forward ({method.upper()})"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(
        True,
        which="both",
        alpha=0.3,
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )
    plt.close(fig)


# ======================================================================================
# Main
# ======================================================================================
def main():
    args = parse_cli()

    simu_path = resolve_path(args.simu_config)
    param_path = resolve_path(args.param_config)
    output_dir = resolve_path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        simu_path,
        "r",
    ) as file:
        simu_params = json.load(file)

    with open(
        param_path,
        "r",
    ) as file:
        params = json.load(file)

    solver_params = simu_params["solver_params"]

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

    print("=" * 88)
    print("COMPLETE FORWARD PERFORMANCE STUDY")
    print("=" * 88)
    print(f"Device       : {jax.devices()[0]}")
    print(f"Method       : {args.method}")
    print(f"Geometries   : {args.geometries}")
    print(f"Nx           : {args.Nx}")
    print(f"T_max        : {T_max}")
    print(f"CFL          : {CFL}")
    print(f"N_snapshot   : {N_snapshot}")
    print(f"Repeats      : {args.n_repeats}")
    print("=" * 88)

    rows = []

    for geometry in args.geometries:
        print(
            f"\n{'=' * 88}\n"
            f"GEOMETRY: {geometry}\n"
            f"{'=' * 88}"
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
                nq=args.nq,
                method=args.method,
                n_repeats=args.n_repeats,
                n_x_plot=args.max_reconstruction_points,
            )

            rows.append(result)

            print(
                f"dt                  = {result['dt']:.6e} s"
            )
            print(
                f"nsteps              = {result['nsteps']}"
            )
            print(
                f"setup               = {result['setup_time_s']:.4f} s"
            )
            print(
                f"first forward call  = {result['first_call_time_s']:.4f} s"
            )
            print(
                f"compiled median     = {result['execution_median_s']:.4f} s"
            )
            print(
                f"compiled mean ± std = "
                f"{result['execution_mean_s']:.4f} ± "
                f"{result['execution_std_s']:.4f} s"
            )
            print(
                f"time per step        = "
                f"{1e6 * result['time_per_step_median_s']:.3f} µs"
            )

    csv_path = (
        output_dir
        / f"performance_complete_forward_{args.method}.csv"
    )
    save_csv(
        rows,
        csv_path,
    )

    scaling_path = (
        output_dir
        / f"performance_complete_forward_{args.method}.png"
    )
    slopes = plot_scaling(
        rows,
        scaling_path,
        args.method,
    )

    time_per_step_path = (
        output_dir
        / f"performance_time_per_step_{args.method}.png"
    )
    step_slopes = plot_time_per_step(
        rows,
        time_per_step_path,
        args.method,
    )

    nsteps_path = (
        output_dir
        / f"performance_nsteps_{args.method}.png"
    )
    plot_nsteps_scaling(
        rows,
        nsteps_path,
        args.method,
    )

    decomposition_path = (
        output_dir
        / f"performance_forward_decomposition_{args.method}.png"
    )
    plot_forward_decomposition(
        rows,
        decomposition_path,
        args.method,
    )

    compilation_path = (
        output_dir
        / f"performance_first_vs_compiled_{args.method}.png"
    )
    plot_first_vs_compiled(
        rows,
        compilation_path,
        args.method,
    )

    print("\n" + "=" * 88)
    print("EMPIRICAL LOG-LOG SLOPES")
    print("=" * 88)
    for geometry, slope in slopes.items():
        print(
            f"{geometry:>8s}: total forward alpha = {slope:.4f}"
        )

    print("\nEMPIRICAL PER-STEP LOG-LOG SLOPES")
    print("=" * 88)
    for geometry, slope in step_slopes.items():
        print(
            f"{geometry:>8s}: per-step alpha = {slope:.4f}"
        )

    print("\nFiles written:")
    print(csv_path)
    print(scaling_path)
    print(time_per_step_path)
    print(nsteps_path)
    print(decomposition_path)
    print(compilation_path)


if __name__ == "__main__":
    main()