# ======================================================================================
# Pressure at bell test script
# ======================================================================================

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import json
import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.parse_args import parse_args

from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system

from numerics.time_integrators.euler import time_integrate_euler
from numerics.time_integrators.rk2 import time_integrate_rk2

from utils.reconstruction import reconstruct_system
from utils.util_func import precompute_S_quad, project_L2, compute_v_bc_left
from utils.param_func import set_param
from utils.res_openwind import run_openwind_reference
from utils.build_physical_data import build_physical_data

from physics.bc import BC
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis

from inverse.spectral_loss import spectrogram_db


jax.config.update("jax_enable_x64", True)


GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

def best_time_shift(x, y, dt):
    """
    Cherche le décalage temporel qui maximise la corrélation entre x et y.
    x : signal DG
    y : signal OpenWind
    dt : pas de temps entre snapshots
    """
    x0 = x - jnp.mean(x)
    y0 = y - jnp.mean(y)

    corr_full = jnp.correlate(x0, y0, mode="full")
    lag_index = jnp.argmax(corr_full) - (len(y0) - 1)

    shift_time = lag_index * dt

    corr_max = jnp.max(corr_full) / (
        jnp.linalg.norm(x0) * jnp.linalg.norm(y0) + 1e-12
    )

    return int(lag_index), float(shift_time), float(corr_max)

def compare_dg_openwind_geometry(data, output_dir, type_S, params):
    L = float(data.L_tube + data.L_bell)

    # Géométrie DG échantillonnée finement
    x_dg = jnp.linspace(0.0, L, 2000)
    R_dg = jnp.sqrt(data.section(x_dg) / jnp.pi)

    # Géométrie native réellement donnée à OpenWind.
    geom = params["instrument_geometry"]
    L_tube = float(geom["tube"]["L_tube"])
    R_tube = float(geom["tube"]["R_tube"])
    L_bell = float(geom["bell"]["L_bell"])
    k_bell = float(geom["bell"]["k_bell"])

    if type_S == "const":
        x_ow_raw = jnp.array([0.0, L])
        R_ow_raw = jnp.array([R_tube, R_tube])
        R_ow = jnp.ones_like(x_dg) * R_tube
    elif type_S == "exp":
        R_end = R_tube * jnp.exp(0.5 * k_bell * L_bell)
        x_ow_raw = jnp.array([0.0, L_tube, L])
        R_ow_raw = jnp.array([R_tube, R_tube, R_end])
        R_ow = jnp.where(
            x_dg < L_tube,
            R_tube,
            R_tube * jnp.exp(0.5 * k_bell * (x_dg - L_tube)),
        )
    else:
        from utils.util_func import make_openwind_radius_profile
        x_ow_raw, R_ow_raw = make_openwind_radius_profile(data)
        x_ow_raw = jnp.asarray(x_ow_raw)
        R_ow_raw = jnp.asarray(R_ow_raw)
        R_ow = jnp.interp(x_dg, x_ow_raw, R_ow_raw)

    diff_R = R_dg - R_ow

    # erreur L1 relative
    err_R_l1 = (
        jnp.sum(jnp.abs(diff_R))
        / (jnp.sum(jnp.abs(R_ow)) + 1e-12)
    )

    # erreur L2 relative
    err_R_l2 = (
        jnp.linalg.norm(diff_R)
        / (jnp.linalg.norm(R_ow) + 1e-12)
    )

    # erreur L∞ relative
    err_R_linf = (
        jnp.max(jnp.abs(diff_R))
        / (jnp.max(jnp.abs(R_ow)) + 1e-12)
    )

    print("\n=== Erreur rayon DG/OpenWind ===")
    print("Erreur relative L1   =", float(err_R_l1))
    print("Erreur relative L2   =", float(err_R_l2))
    print("Erreur relative Linf =", float(err_R_linf))
    print("Erreur absolue max   =", float(jnp.max(jnp.abs(diff_R))))

    S_dg = jnp.pi * R_dg**2
    S_ow = jnp.pi * R_ow**2

    diff_S = S_dg - S_ow

    err_S_l2 = (
        jnp.linalg.norm(diff_S)
        / (jnp.linalg.norm(S_ow) + 1e-12)
    )

    err_S_linf = (
        jnp.max(jnp.abs(diff_S))
        / (jnp.max(jnp.abs(S_ow)) + 1e-12)
    )

    print("\n=== Erreur section DG/OpenWind ===")
    print("Erreur relative L2   =", float(err_S_l2))
    print("Erreur relative Linf =", float(err_S_linf))

    plt.figure(figsize=(10, 5))
    plt.plot(x_dg, R_dg, label="Rayon DG : sqrt(S/pi)")
    plt.plot(x_dg, R_ow, "--", label="Rayon OpenWind interpolé")
    plt.scatter(x_ow_raw, R_ow_raw, s=15, label="Points fournis à OpenWind")
    plt.xlabel("x")
    plt.ylabel("Rayon")
    plt.title(f"Comparaison géométrie DG / OpenWind — {type_S}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(output_dir, f"geometry_compare_{type_S}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x_dg, diff_R)
    plt.xlabel("x")
    plt.ylabel("R_DG - R_OpenWind")
    plt.title("Erreur de rayon DG - OpenWind")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    diff_path = os.path.join(output_dir, f"geometry_diff_{type_S}.png")
    plt.savefig(diff_path, dpi=150)
    plt.close()

    print("Figure géométrie sauvegardée :", fig_path)
    print("Figure différence sauvegardée :", diff_path)


def local_time_shift(x, y, t, window_time=0.08, hop_time=0.02):
    dt = float(t[1] - t[0])
    window_size = int(window_time / dt)
    hop_size = int(hop_time / dt)

    centers = []
    shifts = []
    corrs = []

    for start in range(0, len(t) - window_size, hop_size):
        end = start + window_size

        xw = x[start:end]
        yw = y[start:end]
        tw = t[start:end]

        lag, shift, corr = best_time_shift(xw, yw, dt)

        centers.append(float(0.5 * (tw[0] + tw[-1])))
        shifts.append(float(shift))
        corrs.append(float(corr))

    return jnp.array(centers), jnp.array(shifts), jnp.array(corrs)


def main():

    # --------------------------------------------------------------------------
    # Lecture config solveur
    # --------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/simu.json", "r") as f:
        simu_params = json.load(f)

    solver_params = simu_params["solver_params"]

    T_max = solver_params["T_max"]
    CFL = solver_params["cfl"]
    Nx = solver_params["Nx"]
    N_snapshot_time = solver_params["N_snapshot"]

    # --------------------------------------------------------------------------
    # Lecture config physique
    # --------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/param.json", "r") as f:
        params = json.load(f)

    physical_params = params["physics"]
    initial_conditions_reed = params["init_cond_reed"]

    c = physical_params["c"]
    phi0 = physical_params["phi0"]

    y0 = initial_conditions_reed["y0"]
    z0 = initial_conditions_reed["y_dot0"]

    # --------------------------------------------------------------------------
    # Arguments
    # --------------------------------------------------------------------------
    args = parse_args()
    method = args.method
    type_S = args.type_S

    output_dir = "../experiments/pressure_at_bell/results"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Données DG initiales
    # --------------------------------------------------------------------------
    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell

    compare_dg_openwind_geometry(data, output_dir, type_S, params)
    
    S_star = data.section(0.0)
    Zt = S_star / data.section(L)

    print("\n=== Géométrie DG ===")
    print("S_star =", float(S_star))
    print("S(L)  =", float(data.section(L)))
    print("Zt    =", float(Zt))

    # --------------------------------------------------------------------------
    # OpenWind d'abord
    # --------------------------------------------------------------------------
    print("\n=== Simulation OpenWind avant DG ===")

    (
        t_ow,
        p_ow_left,
        p_ow_right,
        y_ow,
        gamma_ow,
        ow_rad_params,
    ) = run_openwind_reference(
        param_json=params,
        T_max=T_max,
        type_S=type_S,
        l_ele = 5e-4
    )

    print("\n=== Paramètres radiation OpenWind ===")
    print(ow_rad_params)

    # --------------------------------------------------------------------------
    # Injection des paramètres OpenWind dans DG
    # --------------------------------------------------------------------------
    data = set_param(data, "Zt", jnp.array(Zt), GEO_KEYS)

    if ow_rad_params["alpha"] is not None:
        data = set_param(
            data,
            "alpha",
            jnp.array(ow_rad_params["alpha"]),
            GEO_KEYS,
        )

    if ow_rad_params["beta"] is not None:
        data = set_param(
            data,
            "beta",
            jnp.array(ow_rad_params["beta"]),
            GEO_KEYS,
        )

    print("\n=== Paramètres DG utilisés ===")
    print("alpha =", float(data.alpha))
    print("beta  =", float(data.beta))
    print("Zt    =", float(data.Zt))
    print("sqrt(alpha)/Zt =", float(jnp.sqrt(data.alpha) / data.Zt))

    start_time = time.time()
    # --------------------------------------------------------------------------
    # Conditions aux limites
    # --------------------------------------------------------------------------
    bc = BC(type="full")

    def p0(x):
        return init_func_const(x, L)

    def v0(x):
        return 0.0

    # --------------------------------------------------------------------------
    # Maillage DG
    # --------------------------------------------------------------------------
    print("\n=== Computing DG solution ===")

    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])
    S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)

    S_ext = S_nodes
    S_bc = S_cells.at[0].set(S_nodes[0]).at[-1].set(S_nodes[-1])
    Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=0)(hs)

    u0 = project_L2(
        xLs,
        xRs,
        p0,
        v0,
        data.section,
        c,
        S_star,
        Mp_inv,
        Mv_inv,
    )

    u0 = jnp.stack(
        [
            jnp.stack(
                [
                    jnp.array(
                        [
                            S_nodes[i] / (c * S_star) * p0(xLs[i]),
                            S_nodes[i + 1] / (c * S_star) * p0(xRs[i]),
                        ]
                    ),
                    jnp.array(
                        [
                            S_star / (c * S_nodes[i]) * v0(xLs[i]),
                            S_star / (c * S_nodes[i + 1]) * v0(xRs[i]),
                        ]
                    ),
                ]
            )
            for i in range(Nx)
        ],
        axis=0,
    )

    print("u0 min/max:", float(u0.min()), float(u0.max()))
    print("u0 shape:", u0.shape)
    print("S_cells min/max:", float(S_cells.min()), float(S_cells.max()))
    print("y0:", y0, "z0:", z0)
    print("phi0:", phi0)
    print("CFL:", CFL)

    # --------------------------------------------------------------------------
    # Temps
    # --------------------------------------------------------------------------
    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max / dt))

    print(f"Time step dt: {dt:.6e} s, number of steps: {nsteps}")

    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot_time)
    ).astype(jnp.int32)

    t_solver = jnp.arange(nsteps) * dt

    gamma_t = pressure_at_mouth_alexis(
        gamma_final=data.gamma_final,
        t_attack=data.t_attack,
        t=t_solver,
    )

    # --------------------------------------------------------------------------
    # Intégration temporelle DG
    # --------------------------------------------------------------------------
    if method == "euler":
        (
            u_tilde,
            phi,
            y,
            y_dot,
            u_tilde_snaps,
            phi_snaps,
            y_snaps,
            z_snaps,
        ) = time_integrate_euler(
            u0,
            x_nodes,
            c,
            dt,
            nsteps,
            Mp_inv,
            Mv_inv,
            bc,
            phi0,
            y0,
            z0,
            data,
            S_cells=S_bc,
            S_star=S_star,
            S_quad=S_quad,
            S_ext=S_ext,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
        )
    else:
        (
            u_tilde,
            phi,
            y,
            y_dot,
            u_tilde_snaps,
            phi_snaps,
            y_snaps,
            z_snaps,
        ) = time_integrate_rk2(
            u0,
            x_nodes,
            c,
            dt,
            nsteps,
            Mp_inv,
            Mv_inv,
            bc,
            phi0,
            y0,
            z0,
            data,
            S_cells=S_bc,
            S_star=S_star,
            S_quad=S_quad,
            S_ext=S_ext,              # ajout
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
        )

    # --------------------------------------------------------------------------
    # Reconstruction
    # --------------------------------------------------------------------------
    x_plot = jnp.linspace(0.0, L, 1000)

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

    p_all, v_all = reconstruct_all_snaps(u_tilde_snaps)

    p_bell = p_all[:, -1]
    p_left_dg = p_all[:, 0]

    t_dg = jnp.asarray(n_snaps * dt)
    p_dg = jnp.asarray(p_bell)

    elapsed_time = time.time() - start_time
    print(f"Elapsed time after DG: {elapsed_time:.2f} seconds")

    # --------------------------------------------------------------------------
    # Interpolation OpenWind sur les temps DG
    # --------------------------------------------------------------------------
    p_ow_left_interp = jnp.interp(t_dg, t_ow, p_ow_left)
    p_ow_right_interp = jnp.interp(t_dg, t_ow, p_ow_right)

    opening = 5e-4
    y_ow_interp = jnp.interp(t_dg, t_ow, y_ow) / opening

    gamma_ow_interp = jnp.interp(t_dg, t_ow, gamma_ow)

    Pclosed = 5e3

    p_dg_scaled = Pclosed * p_dg
    p_left_dg_scaled = Pclosed * p_left_dg


    dt_snap = float(t_dg[1] - t_dg[0])

    lag_index, shift_time, corr_shifted = best_time_shift(
        p_dg_scaled,
        p_ow_right_interp,
        dt_snap,
    )

    print("\n=== Recalage temporel optimal ===")
    print("lag index =", lag_index)
    print("shift time =", shift_time, "s")
    print("corr après recalage =", corr_shifted)

    if lag_index > 0:
        p_dg_shift = p_dg_scaled[lag_index:]
        p_ow_shift = p_ow_right_interp[:-lag_index]
        t_shift = t_dg[lag_index:]

    elif lag_index < 0:
        k = -lag_index
        p_dg_shift = p_dg_scaled[:-k]
        p_ow_shift = p_ow_right_interp[k:]
        t_shift = t_dg[:-k]

    else:
        p_dg_shift = p_dg_scaled
        p_ow_shift = p_ow_right_interp
        t_shift = t_dg

    # --------------------------------------------------------------------------
    # Delta p
    # --------------------------------------------------------------------------
    gamma_dg_snaps = gamma_t[n_snaps]

    delta_p_dg = gamma_dg_snaps - p_left_dg
    delta_p_ow = gamma_ow_interp - p_ow_left_interp / Pclosed

    rel_err_delta_p = (
        jnp.linalg.norm(delta_p_dg - delta_p_ow)
        / (jnp.linalg.norm(delta_p_ow) + 1e-12)
    )

    rel_err = (
        jnp.linalg.norm(p_dg_scaled - p_ow_right_interp)
        / (jnp.linalg.norm(p_ow_right_interp) + 1e-12)
    )

    rel_err_gamma = (
        jnp.linalg.norm(gamma_dg_snaps - gamma_ow_interp)
        / (jnp.linalg.norm(gamma_ow_interp) + 1e-12)
    )

    rel_err_y = (
        jnp.linalg.norm(y_snaps - y_ow_interp)
        / (jnp.linalg.norm(y_ow_interp) + 1e-12)
    )

    rel_err_shifted = jnp.linalg.norm(p_dg_shift - p_ow_shift) / (
    jnp.linalg.norm(p_ow_shift) + 1e-12
    )

    print("Erreur relative après recalage =", float(rel_err_shifted))

    corr = (
        jnp.vdot(
            p_dg_scaled - jnp.mean(p_dg_scaled),
            p_ow_right_interp - jnp.mean(p_ow_right_interp),
        )
        / (
            jnp.linalg.norm(p_dg_scaled - jnp.mean(p_dg_scaled))
            * jnp.linalg.norm(p_ow_right_interp - jnp.mean(p_ow_right_interp))
            + 1e-12
        )
    )

    scale_opt = (
        jnp.vdot(p_dg, p_ow_right_interp)
        / (jnp.vdot(p_dg, p_dg) + 1e-12)
    )

    print("\n=== Comparaison amplitudes ===")
    print("max |p_dg|              =", float(jnp.max(jnp.abs(p_dg))))
    print("max |Pclosed*p_dg|      =", float(jnp.max(jnp.abs(p_dg_scaled))))
    print("max |p_ow_left|         =", float(jnp.max(jnp.abs(p_ow_left_interp))))
    print("max |p_ow_right|        =", float(jnp.max(jnp.abs(p_ow_right_interp))))
    print("Erreur relative sortie  =", float(rel_err))
    print("corr                    =", float(corr))
    print("scale_opt               =", float(scale_opt))
    print("scale_opt / Pclosed     =", float(scale_opt / Pclosed))

    print("\n=== Delta p ===")
    print("DG delta_p min/max =", float(jnp.min(delta_p_dg)), float(jnp.max(delta_p_dg)))
    print("OW delta_p min/max =", float(jnp.min(delta_p_ow)), float(jnp.max(delta_p_ow)))
    print("Erreur relative delta_p =", float(rel_err_delta_p))

    print("\n=== Anche ===")
    print("DG y min/max =", float(jnp.min(y_snaps)), float(jnp.max(y_snaps)))
    print("OW y min/max =", float(jnp.min(y_ow_interp)), float(jnp.max(y_ow_interp)))
    print("Erreur relative y =", float(rel_err_y))

    # --------------------------------------------------------------------------
    # Spectrogrammes
    # --------------------------------------------------------------------------
    dt_spec = float(t_dg[1] - t_dg[0])

    t_spec_dg, f_spec_dg, S_dg = spectrogram_db(
        p_dg_scaled,
        dt_spec,
        n_fft=512,
        hop_length=64,
    )

    t_spec_ow, f_spec_ow, S_ow = spectrogram_db(
        p_ow_right_interp,
        dt_spec,
        n_fft=512,
        hop_length=64,
    )

    S_diff = S_dg - S_ow

    # --------------------------------------------------------------------------
    # Figures
    # --------------------------------------------------------------------------
    fig, axes = plt.subplots(9, 1, figsize=(10, 22))

    ax1, ax2, ax2b, ax3, ax4, ax5, ax6, ax7, ax8 = axes

    ax1.plot(t_dg, p_dg_scaled, label="DG * Pclosed")
    ax1.plot(t_dg, p_ow_right_interp, "--", label="OpenWind sortie")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pressure (Pa)")
    ax1.set_title(
        f"Pressure comparison — rel_err={float(rel_err):.3e}, corr={float(corr):.3f}"
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(t_dg, y_snaps, label="DG y(t)")
    ax2.plot(t_dg, y_ow_interp, "--", label="OpenWind y(t)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Reed displacement y")
    ax2.set_title(f"Reed displacement comparison, rel_err={float(rel_err_y):.3e}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2b.plot(t_dg, delta_p_dg, label="DG")
    ax2b.plot(t_dg, delta_p_ow, "--", label="OpenWind")
    ax2b.set_xlabel("Time")
    ax2b.set_ylabel("Delta p")
    ax2b.set_title(f"Delta p comparison, rel_err={float(rel_err_delta_p):.3e}")
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    x_fine = jnp.linspace(0.0, L, 1000)
    S_fine = data.section(x_fine)
    R_fine = jnp.sqrt(S_fine / jnp.pi)

    ax3.plot(x_fine, R_fine, label="R(x)")
    ax3.plot(x_fine, -R_fine, label="-R(x)")
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Rayon (m)")
    ax3.set_title(f"Profil (type_S={type_S})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4.plot(t_solver, gamma_t, label="Gamma(t) DG")
    ax4.plot(t_dg, gamma_ow_interp, "--", label="Gamma(t) OpenWind")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Gamma(t)")
    ax4.set_title(f"Gamma(t) comparison, rel_err={float(rel_err_gamma):.3e}")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    im = ax5.pcolormesh(
        t_spec_dg,
        f_spec_dg,
        S_diff,
        shading="auto",
    )

    ax5.set_ylim(0, 3000)
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Frequency (Hz)")
    ax5.set_title("Spectrogram difference: DG - OpenWind (dB)")
    fig.colorbar(im, ax=ax5, label="Difference (dB)")

    p_final, v_final = reconstruct_system(
        u_tilde_snaps[-1],
        x_nodes,
        x_plot,
        data.section,
        c,
        S_star,
    )

    ax6.plot(x_plot, p_final, label="Final pressure p(x,T_max)")
    ax6.set_xlabel("Position x")
    ax6.set_ylabel("Final pressure")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7.plot(x_plot, v_final, label="Final velocity v(x,T_max)")
    ax7.set_xlabel("Position x")
    ax7.set_ylabel("Final velocity")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    ax8.plot(t_dg, p_left_dg_scaled, label="DG entrée")
    ax8.plot(t_dg, p_ow_left_interp, "--", label="OpenWind entrée")
    ax8.set_xlabel("Time")
    ax8.set_ylabel("Pressure at left end (Pa)")
    ax8.set_title("Pressure at left end comparison")
    ax8.legend()
    ax8.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = os.path.join(
        output_dir,
        f"pressure_and_reed_{method}_{type_S}.png",
    )

    plt.savefig(fig_path, dpi=150)
    plt.close()

    print("\nFigure sauvegardée :", fig_path)

    # --------------------------------------------------------------------------
    # Diagnostics fréquentiels
    # --------------------------------------------------------------------------
    t_snaps = n_snaps * dt
    dt_snap = float(t_snaps[1] - t_snaps[0])

    df = 1.0 / (len(p_bell) * dt_snap)
    print(f"\nRésolution fréquentielle (signal entier) : {df:.2f} Hz")

    p_signal = p_bell - jnp.mean(p_bell)
    freqs = jnp.fft.fftfreq(len(p_signal), d=dt_snap)
    spectrum = jnp.abs(jnp.fft.fft(p_signal))
    pos_mask = freqs > 0

    f_play = float(freqs[pos_mask][jnp.argmax(spectrum[pos_mask])])
    print(f"Fréquence dominante (signal entier) : {f_play:.4f} Hz")

    i_start = int(jnp.searchsorted(t_snaps, data.t_attack * 3))

    p_steady = p_bell[i_start:] - jnp.mean(p_bell[i_start:])
    y_steady = y_snaps[i_start:]

    df_steady = 1.0 / (len(p_steady) * dt_snap)
    print(f"Résolution fréquentielle (régime établi): {df_steady:.2f} Hz")

    freqs_s = jnp.fft.fftfreq(len(p_steady), d=dt_snap)
    spectrum_s = jnp.abs(jnp.fft.fft(p_steady))
    pos_mask_s = freqs_s > 0

    f_play_steady = float(freqs_s[pos_mask_s][jnp.argmax(spectrum_s[pos_mask_s])])

    print(f"Fréquence de jeu (régime établi) : {f_play_steady:.4f} Hz")
    print(
        "Amplitude (régime établi)        :",
        float(0.5 * (jnp.max(p_steady) - jnp.min(p_steady))),
    )
    print("Ouverture moyenne (régime établi):", float(jnp.mean(y_steady)))

    top_k = 5
    top_indices = jnp.argsort(spectrum_s[pos_mask_s])[-top_k:][::-1]
    freqs_top = freqs_s[pos_mask_s][top_indices]
    amps_top = spectrum_s[pos_mask_s][top_indices]

    print("\nTop 5 fréquences dominantes (régime établi):")
    for f, a in zip(freqs_top, amps_top):
        print(f"  {float(f):8.2f} Hz   amplitude: {float(a):.4e}")

    print(f"\nFréquences théoriques du tube (L={float(L):.3f} m):")
    for n in range(1, 6):
        print(f"  Mode {n}: {n * 340 / (4 * float(L)):.2f} Hz  (tube ouvert-fermé)")
        print(f"  Mode {n}: {n * 340 / (2 * float(L)):.2f} Hz  (tube ouvert-ouvert)")

    # --------------------------------------------------------------------------
    # Debug final
    # --------------------------------------------------------------------------
    v_bc_adim = compute_v_bc_left(
        y_snaps,
        z_snaps,
        p_all[:, 0],
        data.zeta,
        gamma_t[n_snaps],
        data.eps,
        data.kappa,
        data.fr * (2 * jnp.pi),
        data.l,
    )

    print("\n=== Debug final ===")
    print("gamma_t min/max =", float(jnp.min(gamma_t)), float(jnp.max(gamma_t)))
    print("wr =", float(data.fr * (2 * jnp.pi)))
    print("Qr =", float(data.Qr))
    print("zeta =", float(data.zeta))
    print("kappa =", float(data.kappa))
    print("alpha =", float(data.alpha))
    print("beta =", float(data.beta))
    print("Zt =", float(data.Zt))
    print("c =", float(c))
    print("L =", float(L))
    print("sqrt(alpha)/Zt =", float(jnp.sqrt(data.alpha) / data.Zt))
    print("v_bc_adim min/max =", float(jnp.min(v_bc_adim)), float(jnp.max(v_bc_adim)))
    print("v_left_dg min/max =", float(jnp.min(v_all[:, 0])), float(jnp.max(v_all[:, 0])))

    plt.figure(figsize=(10, 4))
    plt.plot(t_shift, p_dg_shift, label="DG recalé")
    plt.plot(t_shift, p_ow_shift, "--", label="OpenWind")
    plt.xlabel("Time")
    plt.ylabel("Pressure (Pa)")
    plt.title(
        f"Recalage temporel — shift={shift_time:.3e}s, "
        f"corr={corr_shifted:.3f}, err={float(rel_err_shifted):.3e}"
    )
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../experiments/pressure_at_bell/results/pressure_shifted.png", dpi=150)
    plt.close()

    # ============================================================
    # Comparaison de forme pure, après recalage temporel
    # ============================================================

    x = p_dg_shift - jnp.mean(p_dg_shift)
    y = p_ow_shift - jnp.mean(p_ow_shift)

    x = x / (jnp.linalg.norm(x) + 1e-12)
    y = y / (jnp.linalg.norm(y) + 1e-12)

    corr_shape = jnp.vdot(x, y)
    err_shape = jnp.linalg.norm(x - y)

    print("\n=== Comparaison de forme après recalage ===")
    print("corr_shape =", float(corr_shape))
    print("err_shape =", float(err_shape))
    print("sqrt(2*(1-corr)) =", float(jnp.sqrt(2.0 * (1.0 - corr_shape))))

    plt.figure(figsize=(10, 4))
    plt.plot(t_shift, x, label="DG centré-normalisé")
    plt.plot(t_shift, y, "--", label="OpenWind centré-normalisé")
    plt.xlabel("Time")
    plt.ylabel("Signal normalisé")
    plt.title(f"Forme après recalage — err={float(err_shape):.3e}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../experiments/pressure_at_bell/results/pressure_shape_shifted.png", dpi=150)
    plt.close()

    # ============================================================
    # Comparaison fréquentielle FFT après recalage
    # ============================================================

    x_fft = p_dg_shift - jnp.mean(p_dg_shift)
    y_fft = p_ow_shift - jnp.mean(p_ow_shift)

    dt_shift = float(t_shift[1] - t_shift[0])

    freqs = jnp.fft.rfftfreq(len(x_fft), d=dt_shift)

    X = jnp.abs(jnp.fft.rfft(x_fft))
    Y = jnp.abs(jnp.fft.rfft(y_fft))

    # Normalisation pour comparer les formes spectrales
    Xn = X / (jnp.max(X) + 1e-12)
    Yn = Y / (jnp.max(Y) + 1e-12)

    diff_fft = jnp.abs(Xn - Yn)

    plt.figure(figsize=(10, 4))
    plt.semilogy(freqs, Xn + 1e-12, label="DG")
    plt.semilogy(freqs, Yn + 1e-12, "--", label="OpenWind")
    plt.xlim(0, 3000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude normalisée")
    plt.title("FFT comparée après recalage")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("../experiments/pressure_at_bell/results/fft_compare_shifted.png", dpi=150)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, diff_fft)
    plt.xlim(0, 3000)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("|FFT_DG - FFT_OW| normalisé")
    plt.title("Différence spectrale après recalage")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../experiments/pressure_at_bell/results/fft_diff_shifted.png", dpi=150)
    plt.close()

    print("\nFigures FFT sauvegardées :")
    print("../experiments/pressure_at_bell/results/fft_compare_shifted.png")
    print("../experiments/pressure_at_bell/results/fft_diff_shifted.png")

    mask = (freqs > 0) & (freqs < 3000)

    freqs_masked = freqs[mask]
    diff_masked = diff_fft[mask]

    top_k = 20
    idx = jnp.argsort(diff_masked)[-top_k:][::-1]

    print("\nTop différences fréquentielles :")
    for i in idx:
        print(
            f"{float(freqs_masked[i]):8.2f} Hz"
            f"   diff = {float(diff_masked[i]):.4e}"
        )


    def dominant_frequency(signal, dt, t, t_min=0.15):
        mask = t >= t_min
        s = signal[mask] - jnp.mean(signal[mask])

        freqs = jnp.fft.rfftfreq(len(s), d=dt)
        spec = jnp.abs(jnp.fft.rfft(s))

        idx = jnp.argmax(spec[1:]) + 1
        return float(freqs[idx])

    f_dg = dominant_frequency(p_dg_scaled, dt_snap, t_dg)
    f_ow = dominant_frequency(p_ow_right_interp, dt_snap, t_dg)

    print("f_DG =", f_dg)
    print("f_OW =", f_ow)
    print("delta f =", f_dg - f_ow)


    centers, shifts, corrs = local_time_shift(
    p_dg_scaled,
    p_ow_right_interp,
    t_dg,
    window_time=0.08,
    hop_time=0.02,
    )

    plt.figure(figsize=(10, 4))
    plt.plot(centers, shifts, "o-")
    plt.xlabel("Time")
    plt.ylabel("Shift local optimal (s)")
    plt.title("Décalage temporel local DG vs OpenWind")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../experiments/pressure_at_bell/results/local_time_shift.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    main()
