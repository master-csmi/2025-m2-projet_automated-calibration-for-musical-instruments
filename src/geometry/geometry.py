import equinox as eqx
import jax
import jax.numpy as jnp
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from numerics.dg.mass_matrix import local_mass_inv_system
from physics.init_func import init_func_const
from utils.util_func import precompute_S_quad


class SolverGeometry(eqx.Module):
    x_nodes: jnp.ndarray
    S_cells: jnp.ndarray
    S_star: jnp.ndarray
    S_quad: jnp.ndarray
    S_ext: jnp.ndarray
    Mp_inv: jnp.ndarray
    Mv_inv: jnp.ndarray
    u0: jnp.ndarray


def make_geometry(data, Nx, c):
    L = data.section.L_tube + data.section.L_bell
    S_star = data.section(0.0)

    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L)
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])
    S_quad = precompute_S_quad(data.section, xLs, xRs, nq=2)
    S_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])

    Mp_inv, Mv_inv = jax.vmap(local_mass_inv_system, in_axes=0)(hs)

    def p0(x):
        return init_func_const(x, L)

    def v0(x):
        return 0.0

    u0 = jnp.stack([
        jnp.stack([
            jnp.array([
                S_cells[i] / (c * S_star) * p0(xLs[i]),
                S_cells[i] / (c * S_star) * p0(xRs[i]),
            ]),
            jnp.array([
                S_star / (c * S_cells[i]) * v0(xLs[i]),
                S_star / (c * S_cells[i]) * v0(xRs[i]),
            ]),
        ])
        for i in range(Nx)
    ], axis=0)

    return SolverGeometry(
        x_nodes=x_nodes,
        S_cells=S_cells,
        S_star=S_star,
        S_quad=S_quad,
        S_ext=S_ext,
        Mp_inv=Mp_inv,
        Mv_inv=Mv_inv,
        u0=u0,
    )