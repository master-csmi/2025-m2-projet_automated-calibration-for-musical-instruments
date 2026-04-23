# ======================================================================================
# Gradient test script
# ======================================================================================

import os
import jax
import jax.numpy as jnp
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
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis
from utils.eval import eval_pressure


from utils.util_func import project_L2
from utils.build_physical_data import build_physical_data


jax.config.update("jax_enable_x64", True)

def main():
    # launch time 
    start_time = time.time()

    #------------------------------------------------------------------------------
    # Read simulation parameters from json file
    #------------------------------------------------------------------------------
    with open("../experiments/gradient/config/simu.json", "r") as f:
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
    with open("../experiments/gradient/config/param.json", "r") as f:
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

    # Dirichlet BC at left + impedance BC at right
    bc = BC(type="full")

    # Initial conditions
    #def p0(x): return init_func(x, L, phi0=1.0)
    def p0(x): return init_func_const(x, L)
    def v0(x): return 0.0
    #y0 = 1.0 + data.eps * (data.gamma_final - p0(0.0))
    

    

    # Output directory
    output_dir = "../experiments/gradient/results"
    os.makedirs(output_dir, exist_ok=True)


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

    project_L2_jit = jax.jit(
    project_L2,
    static_argnames=("p0", "v0", "S_fun")
    )

    u0 = project_L2_jit(
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
        u_tilde, phi, y, y_dot,snap_id, y_snaps,z_snaps, u_tilde_snaps= time_integrate_euler(
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
        u_tilde, phi, y, y_dot,snap_id, y_snaps, z_snaps, phi_snaps, u_tilde_snaps = time_integrate_rk2(
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
   

    # ==============================================================================
    #TEST GRADIENT AD vs DIFFERENCES FINIES
    # ==============================================================================
    print("\n=== Test gradient AD ===")
 
    # On ne teste le gradient que pour rk2 car euler ne retourne pas phi_snaps
    if method != "rk2":
        print("Test gradient uniquement disponible pour method=rk2, skip.")
        return
 
    def eval_fn(data):
        return eval_pressure(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0,
            y0, z0,
            data,
            S_cells=S_cells, S_star=S_star,S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t,
            x_plot=x_plot
        )
    # ------------------------------------------------------------------
    # Gradient par différentiation automatique
    # ------------------------------------------------------------------
    alpha_val = jnp.array(data.alpha)

    print(f"  Testing alpha = {float(alpha_val):.4f} ...")

    # jit de la fonction loss
    loss_fn_jit = jax.jit(eval_fn)
    grad_fn = jax.jit(jax.grad(eval_fn))
 
    print(f"Calcul du gradient AD pour alpha = {float(alpha_val):.4f} ... \n")
    
    grad_ad  = grad_fn(data)
    print(f"  Gradient AD  : {grad_ad}")
 
    # ------------------------------------------------------------------
    # Validation par différences finies centrées
    # ------------------------------------------------------------------
    h_fd = 1e-4
    print(f"Calcul des différences finies (h={h_fd}) ... \n")
    #points calculés pour le calcul à la main du gradient
    alpha_val_plus  = alpha_val + h_fd
    alpha_val_minus = alpha_val - h_fd

    print(f"Points de test pour le gradient FD:")

    # On crée deux nouvelles instances de PhysicalData avec alpha légèrement modifié pour calculer les pertes correspondantes
    params_plus = params.copy()
    params_plus["right_bc_params"]["alpha"] = float(alpha_val_plus)

    data_plus = build_physical_data(params_plus, type_S)

    loss_plus  = loss_fn_jit(data_plus)
    print(f"  Loss at alpha+h: {float(loss_plus):.6e}")

    params_minus = params.copy()
    params_minus["right_bc_params"]["alpha"] = float(alpha_val_minus)

    data_minus = build_physical_data(params_minus, type_S)
    loss_minus = loss_fn_jit(data_minus)
    print(f"  Loss at alpha-h: {float(loss_minus):.6e}")


    grad_fd    = (loss_plus - loss_minus) / (2.0 * h_fd)
    print(f"  Gradient FD  : {float(grad_fd):.6e}")

    #print gradient ad qui est un physcial date
    print(f"  Gradient AD (PhysicalData) : {grad_ad.alpha}")

 
    # ------------------------------------------------------------------
    # Comparaison
    # ------------------------------------------------------------------
    err_rel = abs(grad_ad.alpha - float(grad_fd)) / (abs(float(grad_fd)) + 1e-30)
    print(f"  Erreur relative AD/FD : {err_rel:.2e}")
 
    if err_rel < 1e-2:
        print(" Gradient AD validé (erreur < 1%)")
    elif err_rel < 1e-1:
        print(" Gradient AD acceptable (erreur < 10%), vérifier le CFL ou nsteps")
    else:
        print(" Gradient AD suspect (erreur > 10%), vérifier la différentiabilité du pipeline")
 
    print("\n=== Fin du test gradient ===")

if __name__ == "__main__":
    main()