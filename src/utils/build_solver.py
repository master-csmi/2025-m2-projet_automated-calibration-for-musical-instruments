import equinox as eqx
import jax
import jax.numpy as jnp
from src.numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from src.physics.init_func import init_func_const
from utils.util_func import precompute_S_quad
from src.numerics.dg.mass_matrix import local_mass_inv_system


class SolverGeometry(eqx.Module):
    x_nodes: jax.Array
    S_cells: jax.Array
    S_quad: jax.Array
    S_ext: jax.Array
    Mp_inv: jax.Array
    Mv_inv: jax.Array
    u0: jax.Array
    S_star: jax.Array

def build_solver_geometry(data, Nx, c):
    """
    Pré-calcule les données spatiales d'une géométrie fixe.

    Cette fonction doit être rappelée si Nx, c ou la section changent.
    """
    L = data.section.L_tube + data.section.L_bell
    S_star = jnp.pi * data.section.R_tube**2

    x_nodes, _ = create_uniform_nodes_with_ghosts(
        Nx,
        0.0,
        L,
    )

    xLs, xRs = cell_edges_from_nodes(x_nodes)
    hs = xRs - xLs

    # Section aux nœuds et dans les cellules
    S_nodes = data.section(x_nodes)
    S_cells = 0.5 * (S_nodes[:-1] + S_nodes[1:])

    # Section aux points de quadrature
    S_quad = precompute_S_quad(
        data.section,
        xLs,
        xRs,
        nq=2,
    )

    # Section étendue pour les interfaces avec cellules fantômes
    S_ext = jnp.concatenate([
        S_cells[:1],
        S_cells,
        S_cells[-1:],
    ])

    # Matrices de masse
    Mp_inv, Mv_inv = jax.vmap(
        local_mass_inv_system
    )(hs)

    # État initial
    p_left = jax.vmap(
        lambda x: init_func_const(x, L)
    )(xLs)

    p_right = jax.vmap(
        lambda x: init_func_const(x, L)
    )(xRs)

    p0_nodes = jnp.stack(
        [p_left, p_right],
        axis=1,
    )

    u0_p = (
        S_cells[:, None] / (c * S_star)
    ) * p0_nodes

    # La vitesse initiale vaut actuellement zéro
    u0_v = jnp.zeros_like(u0_p)

    u0 = jnp.stack(
        [u0_p, u0_v],
        axis=1,
    )  # (Nx, 2, 2)

    return SolverGeometry(
        x_nodes=x_nodes,
        S_cells=S_cells,
        S_quad=S_quad,
        S_ext=S_ext,
        Mp_inv=Mp_inv,
        Mv_inv=Mv_inv,
        u0=u0,
        S_star=S_star,
    )