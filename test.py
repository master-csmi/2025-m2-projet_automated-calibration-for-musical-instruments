import jax
import jax.numpy as jnp
import pytest
import json

from dg_solver.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from dg_solver.mass_matrix import local_mass_inv_system
from dg_solver.time_integrators import time_integrate_euler, time_integrate_rk2
from dg_solver.reconstruction import reconstruct_system
from bc.bc import BC
from utils.S_profiles import S_of_x
from utils.init_func import init_func
from physics.exact_solution import exact_solution_characteristics


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
    # Read parameters from json file
    #------------------------------------------------------------------------------
    with open("utils/parameters.json", "r") as f:
        params = json.load(f)

        solver_params = params["solver_params"]
        left_bc_parameters = params["left_bc_params"]
        right_bc_parameters = params["right_bc_params"]
        initial_conditions_reed = params["init_cond_reed"]

        # Extract solver parameters
        c = solver_params["c"]
        phi0 = solver_params["phi0"]
        S_star = solver_params["S_star"]

        # Extract initial conditions for reed
        y0 = initial_conditions_reed["y0"]
        z0 = initial_conditions_reed["y_dot0"]

        # Extract parameters for left BC
        gamma = left_bc_parameters["gamma"]
        epsilon = left_bc_parameters["epsilon"]
        kappa = left_bc_parameters["kappa"]
        f_r = left_bc_parameters["f_r"]
        Qr = left_bc_parameters["Qr"]
        zeta = left_bc_parameters["zeta"]

        # Extract parameters for right BC
        beta = right_bc_parameters["beta"]
        Z = right_bc_parameters["Z"] #Z_T
        alpha = right_bc_parameters["alpha"]

    L = 2.0
    T = 0.5
    CFL = 0.05
    Ns = [100, 200, 400, 800]
    type_S = "const"
    

    A = jnp.array([[0.0, 1.0],
                   [1.0, 0.0]])

    smax = c * jnp.max(jnp.abs(jnp.linalg.eigvals(A)))

    bc = BC(type="dirichlet")

    def p0(x): return init_func(x, L, phi0=1.0)
    def v0(x): return 0.0

    p_errors = []

    # ------------------------------------------------------------------
    # Convergence loop
    # ------------------------------------------------------------------
    for N in Ns:

        x_nodes, _ = create_uniform_nodes_with_ghosts(N, 0.0, L)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs

        S_nodes = S_of_x(x_nodes, type_S)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

        Mp_inv, Mv_inv = jax.vmap(
            local_mass_inv_system,
            in_axes=(0)
        )(hs)

        u0 = jnp.stack([
            jnp.stack([
                jnp.array([p0(xLs[i]), p0(xRs[i])]),
                jnp.array([v0(xLs[i]), v0(xRs[i])])
            ])
            for i in range(N)
        ], axis=0)

        h = xRs[0] - xLs[0]
        dt = CFL * h / c
        nsteps = int(jnp.ceil(T / dt))
        

        if method == "euler":
            u, _, _, _, _,_ = time_integrate_euler(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0, beta, Z, alpha,
                y0, z0, gamma, epsilon, kappa, omega_r=f_r*2*jnp.pi, Q_r=Qr, zeta=zeta,
                S_cells=S_cells, S_star=S_star,
                snapshot_steps=[nsteps]
            )
        else:
            u, _, _, _, _,_ = time_integrate_rk2(
                u0, x_nodes, c,
                dt, nsteps, Mp_inv, Mv_inv,
                bc, phi0, beta, Z, alpha,
                y0, z0, gamma, epsilon, kappa, omega_r=f_r*2*jnp.pi, Q_r=Qr, zeta=zeta,
                S_cells=S_cells, S_star=S_star,
                snapshot_steps=[nsteps]
            )

        x_plot = jnp.linspace(0.0, L, 3000)
        p_num, _ = reconstruct_system(u, x_nodes, x_plot, type_S)

        p_ex, _ = exact_solution_characteristics(
            x_plot, T, p0, c, L,
            alpha, beta, Z, dt=1e-4
        )

        dx = x_plot[1] - x_plot[0]
        err = jnp.sqrt(jnp.sum((p_num - p_ex) ** 2) * dx)
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