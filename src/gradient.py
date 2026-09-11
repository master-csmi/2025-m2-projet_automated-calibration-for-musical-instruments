# ======================================================================================
# Gradient test script
# ======================================================================================

import os
import jax
import jax.numpy as jnp
import json
import copy
import time

from utils.parse_args import parse_args

# DG solver
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system

# Time integration
from numerics.time_integrators.rk2 import time_integrate_rk2

# Boundary conditions
from physics.bc import BC

# Utilities
from utils.util_func import precompute_S_quad
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis
from utils.eval import eval_pressure
from utils.util_func import project_L2
from utils.build_physical_data import build_physical_data

jax.config.update("jax_enable_x64", True)

# Mapping : nom FD  (section JSON, clé JSON)
TRAINABLE_MAP = {
    "alpha":       ("right_bc_params", "alpha"),
    "beta":        ("right_bc_params", "beta"),
    "kappa":       ("left_bc_params",  "kappa"),
    "zeta":        ("left_bc_params",  "zeta"),
    "Zt":          ("right_bc_params", "Zt"),
    "fr":          ("left_bc_params",  "fr"),
    "Qr":          ("left_bc_params",  "Qr"),
    # mouth pressure parameters
    "gamma_final": (("left_bc_params", "mouth_pressure_params"), "gamma_final"),
    "t_attack":    (("left_bc_params", "mouth_pressure_params"), "t_attack"),
    # instrument geometry parameters
    "L_tube":      (("instrument_geometry", "tube"), "L_tube"),
    "R_tube":      (("instrument_geometry", "tube"), "R_tube"),
    "L_bell":      (("instrument_geometry", "bell"), "L_bell"),
    "k_bell":      (("instrument_geometry", "bell"), "k_bell"),
}


def main():
    start_time = time.time()

    # --------------------------------------------------------------------------
    # Lecture des configs
    # --------------------------------------------------------------------------
    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params   = json.load(f)["solver_params"]
        T_max           = solver_params["T_max"]
        CFL             = solver_params["cfl"]
        Nx              = solver_params["Nx"]
        N_snapshot_time = solver_params["N_snapshot"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    physical_params         = params["physics"]
    initial_conditions_reed = params["init_cond_reed"]
    c    = physical_params["c"]
    phi0 = physical_params["phi0"]
    y0   = initial_conditions_reed["y0"]
    z0   = initial_conditions_reed["y_dot0"]

    # --------------------------------------------------------------------------
    # Arguments CLI
    # --------------------------------------------------------------------------
    args   = parse_args()
    method = args.method
    type_S = args.type_S

    if method != "rk2":
        print("Test gradient uniquement disponible pour method=rk2, skip.")
        return

    # --------------------------------------------------------------------------
    # Données physiques de référence (pour infos d'affichage uniquement)
    # --------------------------------------------------------------------------
    data_ref = build_physical_data(params, type_S)
    L_ref    = data_ref.L_tube + data_ref.L_bell
    S_star_ref = jnp.pi * (data_ref.R_tube ** 2)
    print(f"S_star = {S_star_ref:.6e}")

    # --------------------------------------------------------------------------
    # Pas de temps (basé sur la géométrie de référence)
    # --------------------------------------------------------------------------
    x_nodes_ref, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs_ref, xRs_ref = cell_edges_from_nodes(x_nodes_ref)
    h_ref  = xRs_ref[0] - xLs_ref[0]
    dt     = CFL * h_ref / c
    nsteps = int(jnp.ceil(T_max / dt))
    print(f"dt={dt:.6e}, nsteps={nsteps}")

    t_solver = jnp.arange(nsteps) * dt

    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot_time)
    ).astype(jnp.int32)

    # --------------------------------------------------------------------------
    # BC + output dir
    # --------------------------------------------------------------------------
    bc = BC(type="full")
    os.makedirs("../experiments/gradient/results", exist_ok=True)

    # --------------------------------------------------------------------------
    # Fonction d'évaluation
    # --------------------------------------------------------------------------
    def eval_fn(data):
        # --- Géométrie ---
        L      = data.section.L_tube + data.section.L_bell
        S_star = jnp.pi * (data.section.R_tube ** 2)

        x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
        xLs, xRs   = cell_edges_from_nodes(x_nodes)
        hs         = xRs - xLs

        S_nodes = data.section(x_nodes)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])
        S_quad  = precompute_S_quad(data.section, xLs, xRs, nq=2)

        Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=(0))(hs)

        # --- Conditions initiales ---
        def p0(x): return init_func_const(x, L)
        def v0(x): return 0.0

        u0 = jnp.stack([
            jnp.stack([
                jnp.array([S_cells[i] / (c * S_star) * p0(xLs[i]),
                            S_cells[i] / (c * S_star) * p0(xRs[i])]),
                jnp.array([S_star / (c * S_cells[i]) * v0(xLs[i]),
                            S_star / (c * S_cells[i]) * v0(xRs[i])])
            ])
            for i in range(Nx)
        ], axis=0)

        # --- Pression à la bouche ---
        gamma_t = pressure_at_mouth_alexis(
            gamma_final=data.gamma_final,
            t_attack=data.t_attack,
            t=t_solver
        )

        return eval_pressure(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0, y0, z0,
            data,
            S_cells=S_cells, S_star=S_star, S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
            x_plot=jnp.linspace(0.0, L_ref, 1000)
        )

    grad_fn = jax.jit(jax.grad(eval_fn))

    # -------------------------------------------------------------------------
    # Gradient AD
    # -------------------------------------------------------------------------
    print("\n=== Calcul gradient AD ===")
    t0 = time.time()
    grad_ad_result = grad_fn(data_ref)
    print(f"  Gradient AD calculé en {time.time() - t0:.2f}s")

    # -------------------------------------------------------------------------
    # Validation par différences finies
    # -------------------------------------------------------------------------
    print("\n=== Validation FD vs AD ===")
    h_fd = 1e-4

    results = {}

    # Construire les données perturbées pour chaque paramètre actif
    names_active  = []
    data_perturbed = []

    for name, (section, key) in TRAINABLE_MAP.items():
        if not params["trainable"].get(key, params["trainable"].get(name, False)):
            continue

        p_plus  = copy.deepcopy(params)
        p_minus = copy.deepcopy(params)
        print(name)

        if isinstance(section, tuple):
            sec, subsec = section
            print(f"Perturbing {sec} -> {subsec} -> {key}")
            p_plus[sec][subsec][key]  += h_fd
            p_minus[sec][subsec][key] -= h_fd
        else:
            p_plus[section][key]  += h_fd
            p_minus[section][key] -= h_fd

        names_active.append((name, key))
        data_perturbed.append(build_physical_data(p_plus,  type_S))
        data_perturbed.append(build_physical_data(p_minus, type_S))

    # Empiler en un seul pytree batché et évaluer tout en une fois
    data_batch = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *data_perturbed
    )
    eval_batch = jax.jit(jax.vmap(eval_fn))(data_batch)

    # Comparer AD vs FD

    GEO_KEYS = {"L_tube", "R_tube", "L_bell", "k_bell"}

    for i, (name, key) in enumerate(names_active):
        eval_plus  = eval_batch[2 * i]
        eval_minus = eval_batch[2 * i + 1]
        grad_fd = float((eval_plus - eval_minus) / (2.0 * h_fd))

        if key in GEO_KEYS:
            grad_ad = float(getattr(grad_ad_result.section, key))
        else:
            grad_ad = float(getattr(grad_ad_result, key))

        err_rel = abs(grad_ad - grad_fd) / (abs(grad_fd) + 1e-30)
    
        results[name] = err_rel

        print(f"{name:12s} | AD={grad_ad:.4e} | FD={grad_fd:.4e} | err={err_rel:.2e}", end="  ")

        if err_rel < 1e-2:
            print("Top (<1%)")
        elif err_rel < 1e-1:
            print("Moyen (>1%)")
        else:
            print("Mauvais (>10%)")

    print(f"\nTemps total: {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()