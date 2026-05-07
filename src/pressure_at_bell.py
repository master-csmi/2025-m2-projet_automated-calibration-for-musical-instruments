# ======================================================================================
# Pressure at bell test script 
# ======================================================================================

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json

import time

from utils.parse_args import parse_args

# DG solver
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system

# Time integration 
from numerics.time_integrators.euler import (
    time_integrate_euler
)
from numerics.time_integrators.rk2 import (
    time_integrate_rk2
)

from utils.reconstruction import reconstruct_system
from utils.residual import compute_total_residual

# Boundary conditions
from physics.bc import BC

# Utilities
from utils.util_func import precompute_S_quad
from physics.init_func import init_func, init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis

from utils.util_func import project_L2
from utils.build_physical_data import build_physical_data


jax.config.update("jax_enable_x64", True)



def main():
    # launch time 
    start_time = time.time()

    #------------------------------------------------------------------------------
    # Read simulation parameters from json file
    #------------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/simu.json", "r") as f:
        params = json.load(f)

        solver_params = params["solver_params"]

        # Extract solver parameters
        T_max = solver_params["T_max"]
        CFL = solver_params["cfl"]
        Nx = solver_params["Nx"]
        N_snapshot_time = solver_params["N_snapshot"]
    #------------------------------------------------------------------------------
    # Read physical parameters from json file
    #------------------------------------------------------------------------------
    with open("../experiments/pressure_at_bell/config/param.json", "r") as f:
        params = json.load(f)
        physical_params = params["physics"]
        initial_conditions_reed = params["init_cond_reed"]
        instrument_geometry = params["instrument_geometry"]

        # Extract solver parameters
        c = physical_params["c"]

        
        phi0 = physical_params["phi0"]

        L_tube = instrument_geometry["tube"]["L_tube"]

        # Extract initial conditions for reed
        y0 = initial_conditions_reed["y0"]
        z0 = initial_conditions_reed["y_dot0"]

    # ------------------------------------------------------------------------------
    # Parse command-line arguments
    # ------------------------------------------------------------------------------
    args = parse_args()
    method = args.method
    type_S = args.type_S
    

    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell

    S_star = jnp.pi * (data.R_tube**2)  # section de référence pour les variables tilde
    print("Section de la reed", S_star)
    # ------------------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------------------
          # Mesh refinements

    # Reed BC at left + impedance BC at right
    bc = BC(type="full")


    # Initial conditions
    #def p0(x): return init_func(x, L, phi0=1.0)
    def p0(x): return init_func_const(x, L)
    def v0(x): return 0.0
    #y0 = 1.0 + data.eps * (data.gamma_final - p0(0.0))
    

    

    # Output directory
    output_dir = "../experiments/pressure_at_bell/results"
    os.makedirs(output_dir, exist_ok=True)


    # ==============================================================================
    # 1) SOLUTION FIGURES
    # ==============================================================================
    print("\n=== Computing solution plots ===")
   
    plt.figure(figsize=(16, 10))


    # ----------------------------------------------------------------------
    # Mesh
    # ----------------------------------------------------------------------
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    # ----------------------------------------------------------------------
    # Cross-section
    # ----------------------------------------------------------------------
    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)  # (N, nq) sections pré-calculées pour quadrature

    # ----------------------------------------------------------------------
    # Inverse mass matrices
    # ----------------------------------------------------------------------
    Mp_inv, Mv_inv = jax.vmap(
        local_mass_inv_system,
        in_axes=(0)
    )(hs)

    # ----------------------------------------------------------------------
    # Initial DG coefficients
    # ----------------------------------------------------------------------

    u0 = project_L2(
    xLs, xRs,
    p0, v0,
    data.section,
    c, S_star,
    Mp_inv, Mv_inv
    )

    u0 = jnp.stack([
                jnp.stack([

                    # ---- tilde p ----
                    jnp.array([
                        S_cells[i]/(c*S_star) * p0(xLs[i]),
                        S_cells[i]/(c*S_star) * p0(xRs[i])
                    ]),

                    # ---- tilde v ----
                    jnp.array([
                        S_star/(c*S_cells[i]) * v0(xLs[i]),
                        S_star/(c*S_cells[i]) * v0(xRs[i])
                    ])

                ])
                for i in range(Nx)
            ], axis=0)

    print("u0 min/max:", u0.min(), u0.max())
    print("u0 shape:", u0.shape)

    print("S_cells min/max:", S_cells.min(), S_cells.max())
    print("u0 min/max:", u0.min(), u0.max())
    print("y0:", y0, "z0:", z0)
    print("phi0:", phi0)
    print("CFL", CFL)
    # ----------------------------------------------------------------------
    # Time step
    # ----------------------------------------------------------------------
    h = xRs[0] - xLs[0]
    dt = CFL * h / c
    nsteps = int(jnp.ceil(T_max/ dt))
    print(f"Time step dt: {dt:.6e} s, number of steps: {nsteps}")

    n_steps = jnp.arange(0, nsteps, dtype=int)
    # To store snapshots for plot
    n_snaps = jnp.round(
    jnp.linspace(0, nsteps - 1, N_snapshot_time)
    ).astype(jnp.int32)
    print(f"  Snapshot steps: {len(n_snaps)}")
    snapshots = {}

    # Grille temporelle du solveur
    t_solver = jnp.arange(nsteps) * dt

    # Générer gamma(t)
    #gamma_t = pressure_at_mouth(
    #    gamma_final = data.gamma_final,   # valeur plateau depuis JSON
    #    t_attack    = data.t_attack, 
    #    sharpness   = data.sharpness,   
    #    t    = t_solver,
    #    shape       = "linear"
    #)

    gamma_t = pressure_at_mouth_alexis(
        gamma_final = data.gamma_final,   # valeur plateau depuis JSON
        t_attack    = data.t_attack,  
        t    = t_solver
    )


    # ----------------------------------------------------------------------
    # Time integration
    # ----------------------------------------------------------------------


    if method == "euler":
        u_tilde, phi, y, y_dot,u_tilde_snaps,phi_snaps, y_snaps,z_snaps, = time_integrate_euler(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0,
            y0, z0,
            data,
            S_cells=S_cells, S_star=S_star,S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t

        )
    else:
        u_tilde, phi, y, y_dot,u_tilde_snaps,phi_snaps, y_snaps, z_snaps = time_integrate_rk2(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0, 
            y0, z0,
            data,
            S_cells=S_cells, S_star=S_star,S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t
        )

    #residuals = compute_total_residual(
    #u_snaps      = u_tilde_snaps,
    #y_snaps      = y_snaps,
    #z_snaps      = z_snaps,
    #phi_snaps    = phi_snaps,
    #gamma_snaps  = gamma_t,
    #S_cells      = S_cells,
    #S_star       = S_star,
    #c            = c,
    #dt_snap      = float(n_snaps[1] - n_snaps[0]) * dt,
    #h            = h,
    #beta         = data.beta,
    #Z            = data.Zt,
    #alpha        = data.alpha,
    #eps          = data.eps,
    #kappa        = data.kappa,
    #omega_r      = data.wr,
    #Q_r          = data.Qr,
    #zeta         = data.eta,
    #l            = data.l,
#)

    #print("Mean / MaxResidual PDE p:", jnp.mean(residuals["pde_p"]), jnp.max(residuals["pde_p"]))
    #print("Mean / MaxResidual PDE v:", jnp.mean(residuals["pde_v"]), jnp.max(residuals["pde_v"]))
    #print("Mean / MaxResidual ODE:", jnp.mean(residuals["ode"]), jnp.max(residuals["ode"]))
    #print("Mean / MaxResidual BC L:", jnp.mean(residuals["bc_left"]), jnp.max(residuals["bc_left"]))
    #print("Mean / MaxResidual BC R:", jnp.mean(residuals["bc_right"]), jnp.max(residuals["bc_right"]))
    #print("u_tilde shape:", u_tilde.shape)
    #print(u_tilde[-1].shape)
    #print("Before reconstruction, u min/max:", u_tilde_snaps.min(), u_tilde_snaps.max())
    
    
    # ----------------------------------------------------------------------
    # Reconstruction
    # ----------------------------------------------------------------------
    x_plot = jnp.linspace(0.0, L, 1000)
    p_bell=[]
    p = []
    v=[]

    @jax.jit
    def reconstruct_all_snaps(u_tilde_snaps):
        return jax.vmap(
            lambda u_T: reconstruct_system(u_T, x_nodes, x_plot, data.section, c, S_star)
        )(u_tilde_snaps)

    p_all, v_all = reconstruct_all_snaps(u_tilde_snaps)  # (n_snaps, n_plot)
    p_bell = p_all[:, -1]
    #for i, T in enumerate(n_steps * dt):
#
    #    u_T = u_tilde_snaps[i]
    #    p_num, v_num = reconstruct_system(
    #        u_T, x_nodes, x_plot, data.section, c, S_star
    #    )
    #    p.append(p_num)
    #    v.append(v_num) 
    #    p_bell.append(p_num[-1])  # pressure at the bell (x=L)
    
    #stop timer 
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("p shape:", jnp.array(p).shape)

    print("v shape:", jnp.array(v).shape)
   

    # plot results
    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(10, 19))

    # Plot pressure at bell
    ax1.plot(n_snaps * dt, p_bell, label="Pressure at bell (x=L)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Pressure")
    ax1.set_title(f"Pressure at bell over time (method={method}, type_S={type_S})")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot reed displacement
    print("y_snaps shape:", y_snaps.shape)
    ax2.plot(n_snaps * dt, y_snaps, label="Reed displacement y(t)")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Reed displacement y")
    ax2.set_title(f"Reed displacement over time (method={method})")
    ax2.legend()
    ax2.grid(True, alpha=0.3)


    # plot cross-section profile
    x_fine = jnp.linspace(0.0, L, 1000)
    S_fine = data.section(x_fine)
    R_fine = jnp.sqrt(S_fine / jnp.pi)  # rayon équivalent pour visualisation
    ax3.plot(x_fine, R_fine)
    ax3.plot(x_fine, -R_fine)  # symétrique pour visualiser le tube 
    ax3.set_xlabel("Position x")
    ax3.set_ylabel("Rayon (m)" )
    ax3.set_title(f"Profil (type_S={type_S})")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    #plot gamma(t)
    ax4.plot(n_steps * dt, gamma_t, label="Gamma(t) (mouth pressure)")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Gamma(t)")
    ax4.set_title(f"Gamma(t) used in simulation")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    #plot p(x,0), v(x,0) intial conditions
    x_plot = jnp.linspace(0.0, L, 1000)
    p_init, v_init = reconstruct_system(
        u0, x_nodes, x_plot, data.section, c, S_star
    )
    ax5.plot(x_plot, p_init, label="Initial pressure p(x,0)")
    ax5.plot(x_plot, v_init, label="Initial velocity v(x,0)")
    ax5.set_xlabel("Position x")
    ax5.set_ylabel("Initial conditions")
    ax5.set_title(f"Initial conditions (method={method})")
    ax5.legend()
    ax5.grid(True, alpha=0.3) 

    # plot p(x,t_max) et v(x,t_max) at final time
    p_final, v_final = reconstruct_system(
        u_tilde_snaps[-1], x_nodes, x_plot, data.section, c, S_star
    )
    ax6.plot(x_plot, p_final, label=f"Final pressure p(x,T_max)")
    ax6.set_xlabel("Position x")
    ax6.set_ylabel("Final conditions")        
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    ax7.plot(x_plot, v_final, label=f"Final velocity v(x,T_max)")
    ax7.set_xlabel("Position x")
    ax7.set_ylabel("Final conditions")
    ax7.legend()
    ax7.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"pressure_and_reed_{method}_{type_S}.png"),
        dpi=150
    )
    plt.close()


    #print(residuals["pde_p"].shape)
    ## ----------------------------------------------------------------------
    ## Residuals plot
    ## ----------------------------------------------------------------------
    #fig, (ax1,ax2,ax3,ax4,ax5) = plt.subplots(5, 1, figsize=(10, 15))
#
    #ax1.plot(jnp.mean(residuals["pde_p"], axis=1), label="PDE p")
    #ax1.set_yscale("log")
    #ax1.legend()
    #ax1.set_title("Residuals over time")
#
    #ax2.plot(jnp.mean(residuals["pde_v"], axis=1), label="PDE v")
    #ax2.set_yscale("log")
    #ax2.legend()
    #ax2.set_title("Residuals over time")
#
    #ax3.plot(residuals["ode"], label="ODE")
    #ax3.set_yscale("log")
    #ax3.legend()
    #ax3.set_title("Residuals over time")
#
    #ax4.plot(residuals["bc_left"], label="BC left")
    #ax4.set_yscale("log")
    #ax4.legend()
    #ax4.set_title("Residuals over time")
#
    #ax5.plot(residuals["bc_right"], label="BC right")
    #ax5.set_yscale("log")
    #ax5.legend()
    #ax5.set_title("Residuals over time")
#
    #plt.tight_layout()
    #plt.savefig(
    #    os.path.join(output_dir, f"residuals_{method}_{type_S}.png"),
    #    dpi=150
    #)
    #plt.close()

    
   # Fréquence des oscillations
    t_snaps = n_snaps * dt
    dt_snap = float(t_snaps[1] - t_snaps[0])

    # Résolution fréquentielle
    df = 1.0 / (len(p_bell) * dt_snap)
    print(f"Résolution fréquentielle (signal entier) : {df:.2f} Hz")

    # ---- Signal entier ----
    p_signal = jnp.array(p_bell) - jnp.mean(p_bell)
    freqs = jnp.fft.fftfreq(len(p_signal), d=dt_snap)
    spectrum = jnp.abs(jnp.fft.fft(p_signal))
    pos_mask = freqs > 0
    f_play = float(freqs[pos_mask][jnp.argmax(spectrum[pos_mask])])
    print(f"Fréquence dominante (signal entier) : {f_play:.4f} Hz")

    # ---- Régime établi (après 3 * t_attack) ----
    i_start = int(jnp.searchsorted(t_snaps, data.t_attack * 3))
    p_steady = jnp.array(p_bell[i_start:]) - jnp.mean(p_bell[i_start:])
    y_steady = y_snaps[i_start:]

    df_steady = 1.0 / (len(p_steady) * dt_snap)
    print(f"Résolution fréquentielle (régime établi): {df_steady:.2f} Hz")

    freqs_s = jnp.fft.fftfreq(len(p_steady), d=dt_snap)
    spectrum_s = jnp.abs(jnp.fft.fft(p_steady))
    pos_mask_s = freqs_s > 0
    f_play_steady = float(freqs_s[pos_mask_s][jnp.argmax(spectrum_s[pos_mask_s])])

    print(f"Fréquence de jeu (régime établi) : {f_play_steady:.4f} Hz")
    print(f"Amplitude (régime établi)        : {float(0.5*(jnp.max(p_steady)-jnp.min(p_steady))):.4f}")
    print(f"Ouverture moyenne (régime établi): {float(jnp.mean(y_steady)):.4f}")

    # Afficher les 5 fréquences dominantes pour voir le spectre complet
    top_k = 5
    top_indices = jnp.argsort(spectrum_s[pos_mask_s])[-top_k:][::-1]
    freqs_top = freqs_s[pos_mask_s][top_indices]
    amps_top  = spectrum_s[pos_mask_s][top_indices]

    print("\nTop 5 fréquences dominantes (régime établi):")
    for f, a in zip(freqs_top, amps_top):
        print(f"  {float(f):8.2f} Hz   amplitude: {float(a):.4e}")

    # Vérifier aussi les fréquences théoriques
    print(f"\nFréquences théoriques du tube (L={float(L):.3f} m):")
    for n in range(1, 6):
        print(f"  Mode {n}: {n * 340 / (4*L):.2f} Hz  (tube ouvert-fermé)")
        print(f"  Mode {n}: {n * 340 / (2*L):.2f} Hz  (tube ouvert-ouvert)")

    # 1. Regarder le signal temporel pour savoir si on a atteint le régime établi
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    t_snaps_arr = n_snaps * dt

    axes[0].plot(t_snaps_arr, p_bell)
    axes[0].axvline(t_snaps_arr[i_start], color='r', linestyle='--', label='i_start')
    axes[0].set_title("p(L, t) - signal complet")
    axes[0].set_xlabel("t (s)")
    axes[0].legend()

    axes[1].plot(t_snaps_arr[i_start:], p_bell[i_start:])
    axes[1].set_title("p(L, t) - régime établi seulement")
    axes[1].set_xlabel("t (s)")

    axes[2].plot(t_snaps_arr, y_snaps)
    axes[2].axvline(t_snaps_arr[i_start], color='r', linestyle='--', label='i_start')
    axes[2].set_title("Reed y(t)")
    axes[2].set_xlabel("t (s)")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_signal.png"), dpi=150)
    plt.close()

    # 2. Appliquer une fenêtre de Hann pour réduire le leakage
    from jax.numpy import hanning
    window = jnp.hanning(len(p_steady))
    p_windowed = p_steady * window

    freqs_s = jnp.fft.fftfreq(len(p_windowed), d=dt_snap)
    spectrum_s = jnp.abs(jnp.fft.fft(p_windowed))
    pos_mask_s = freqs_s > 0

    # Zoom sur 0-500 Hz pour voir les vrais pics
    fig, ax = plt.subplots(figsize=(10, 5))
    zoom_mask = (freqs_s[pos_mask_s] > 50) & (freqs_s[pos_mask_s] < 500)
    ax.semilogy(freqs_s[pos_mask_s][zoom_mask], spectrum_s[pos_mask_s][zoom_mask])
    ax.axvline(170, color='r', linestyle='--', label='f_r = 170 Hz')
    ax.axvline(340/(4*0.5), color='g', linestyle='--', label='Mode 1 tube (170 Hz)')
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.set_title("Zoom spectre avec fenêtre de Hann")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "debug_spectrum_zoom.png"), dpi=150)
    plt.close()

if __name__ == "__main__":
    main()