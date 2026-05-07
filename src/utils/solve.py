import jax.numpy as jnp
import jax
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from utils.util_func import precompute_S_quad
from physics.init_func import init_func_const
from physics.mouth_pressure import pressure_at_mouth_alexis
from numerics.time_integrators.rk2 import time_integrate_rk2_bell
from numerics.dg.mass_matrix import local_mass_inv_system




def forward_snapshots(data, Nx_train,c, dt, nsteps, bc, phi0, y0, z0, t_solver, n_snaps):
        L      = data.section.L_tube + data.section.L_bell
        S_star = jnp.pi * (data.section.R_tube ** 2)

        x_nodes, _ = create_uniform_nodes_with_ghosts(Nx_train, 0.0, L)
        xLs, xRs   = cell_edges_from_nodes(x_nodes)
        hs         = xRs - xLs

        S_nodes = data.section(x_nodes)
        S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])
        S_quad  = precompute_S_quad(data.section, xLs, xRs, nq=2)
        
        Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=(0))(hs)

        def p0(x): return init_func_const(x, L)
        def v0(x): return 0.0

        u0 = jnp.stack([
            jnp.stack([
                jnp.array([S_cells[i] / (c * S_star) * p0(xLs[i]),
                            S_cells[i] / (c * S_star) * p0(xRs[i])]),
                jnp.array([S_star / (c * S_cells[i]) * v0(xLs[i]),
                            S_star / (c * S_cells[i]) * v0(xRs[i])])
            ])
            for i in range(Nx_train)
        ], axis=0)

        gamma_t = pressure_at_mouth_alexis(
            gamma_final=data.gamma_final,
            t_attack=data.t_attack,
            t=t_solver
        )
        

        (_, _, _, _, p_bell_snaps) = time_integrate_rk2_bell(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0, y0, z0,
            data,
            S_cells=S_cells, S_star=S_star, S_quad=S_quad,
            snapshot_steps=n_snaps,
            gamma_target=gamma_t
        )


        return p_bell_snaps  # (N_snapshot,)