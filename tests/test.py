import jax
import jax.numpy as jnp
import pytest
import json

from src.numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from src.numerics.dg.mass_matrix import local_mass_inv_system
from src.numerics.time_integrators.euler import time_integrate_euler
from src.numerics.time_integrators.rk2 import time_integrate_rk2
from src.utils.reconstruction import reconstruct_system
from src.physics.bc import BC
from src.physics.S_profiles import S_of_x
from src.physics.init_func import init_func
from src.utils.exact_solution import exact_solution_characteristics
from src.utils.build_physical_data import build_physical_data
from src.physics.mouth_pressure import pressure_at_mouth_alexis
from src.utils.util_func import precompute_S_quad

jax.config.update("jax_enable_x64", True)

@pytest.mark.parametrize(
    "method, slope_min, slope_max",
    [
        ("euler", 0.75, 1.25),
        ("rk2",   1.75, 2.25),
    ]
)
def test_convergence_rate(method, slope_min, slope_max):

    # ------------------------------------------------------------------
    # Parameters
    # ------------------------------------------------------------------

    #------------------------------------------------------------------------------
    # Read simulation parameters from json file
    #------------------------------------------------------------------------------
    with open("experiments/convergence/config/simu.json", "r") as f:
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
    with open("experiments/convergence/config/param.json", "r") as f:
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


    T_max = 0.2

    type_S = "const"
    data = build_physical_data(params, type_S)
    L = data.L_tube + data.L_bell

    S_star = jnp.pi * (data.R_tube**2)  # section de référence pour les variables tilde
    Ns = [100,200,400,800]
    
    

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])

    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))

    # free BC at left + impedance BC at right
    type = "right_free"
    assert type in ["full", "right_free"], "Invalid BC type : for test use 'right_free', for full simulation use 'full'"
    bc = BC(type="right_free")


    def p0(x): return init_func(x, L)
    def v0(x): return 0.0

    p_errors = []

    # ------------------------------------------------------------------
    # Convergence loop
    # ------------------------------------------------------------------
    for N in Ns:

        x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs

        S_nodes = data.section(x_nodes)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

        S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)  # (N, nq) sections pré-calculées pour quadrature


        Mp_inv, Mv_inv = jax.vmap(
            local_mass_inv_system,
            in_axes=(0)
        )(hs)

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
                for i in range(N)
            ], axis=0)
        
        h = xRs[0] - xLs[0]

        if method == "euler":
            CFL = 0.05
            dt = CFL * h / c
            nsteps = int(jnp.ceil(T_max / dt))


            # snapshot for plotting
            n_snaps = [nsteps]

            t_solver = jnp.arange(nsteps) * dt

            gamma_t = pressure_at_mouth_alexis(
            gamma_final = data.gamma_final,   
            t_attack    = data.t_attack,  
            t    = t_solver
            )
            

            

            
            u, _, _, _, _,_,_,_,_ = time_integrate_euler(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0,
                y0, z0,
                data,
                S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                snapshot_steps=n_snaps,
                gamma_target=gamma_t

            )

            print(f"u nan: {jnp.any(jnp.isnan(u))}")
            print(f"u min/max: {u.min():.4f} {u.max():.4f}")
        else:
            CFL = 0.2
            dt = CFL * h / c
            nsteps = int(jnp.ceil(T_max / dt))

            n_steps = jnp.arange(0, nsteps, dtype=int)

            # snapshot for plotting
            n_snaps = [nsteps - 1]

            t_solver = jnp.arange(nsteps) * dt

            gamma_t = pressure_at_mouth_alexis(
            gamma_final = data.gamma_final,   
            t_attack    = data.t_attack,  
            t    = t_solver
            )
            
            u, _, _, _, _,_,_,_,_ = time_integrate_rk2(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0,
                y0, z0,
                data,
                S_cells=S_cells, S_star=S_star,S_quad=S_quad,
                snapshot_steps=n_snaps,
                gamma_target=gamma_t
            )

            print(f"u nan: {jnp.any(jnp.isnan(u))}")
            print(f"u min/max: {u.min():.4f} {u.max():.4f}")


        x_plot = jnp.linspace(0.0, L, 3000)
        p_num, _ = reconstruct_system(u, x_nodes, x_plot, data.section, c, S_star)


        p_ex, _ = exact_solution_characteristics(
            x_plot, T_max, p0, c, L,
            data.alpha, data.beta, data.Zt, dt=1e-4
        )

        print(f"p_ex nan: {jnp.any(jnp.isnan(p_ex))}")
        print(f"p_ex min/max: {p_ex.min():.4f} {p_ex.max():.4f}")
        

        dx = x_plot[1] - x_plot[0]
        err = jnp.sqrt(jnp.sum((p_num - p_ex) ** 2) * dx)
        print(f"N={N}, L2 error on pressure = {err}")
        p_errors.append(err)

    # ------------------------------------------------------------------
    # Slope computation
    # ------------------------------------------------------------------
    Ns_arr = jnp.array(Ns)
    errs = jnp.array(p_errors)

    slope = jnp.polyfit(jnp.log10(Ns_arr),
                        jnp.log10(errs), 1)[0]
    slope = -slope  # because error decreases with N

    print(f"[{method}] convergence slope = {slope:.3f}")

    assert slope_min <= slope <= slope_max, (
        f"{method} convergence slope {slope:.2f} "
        f"not in [{slope_min}, {slope_max}]"
    )