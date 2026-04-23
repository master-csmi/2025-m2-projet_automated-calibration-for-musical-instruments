import jax
import jax.numpy as jnp
from src.numerics.dg.mesh import cell_edges_from_nodes
from src.numerics.dg.basis import vphi_at
from src.physics.bc import apply_bc, apply_bc_test
from src.numerics.dg.flux import rusanov_flux


def local_volume_system(u_cell, S_q, h, S_star, c):
    nq = S_q.shape[0]

    # poids trapézoïdaux
    w = jnp.ones(nq) * (h / (nq - 1))
    w = w.at[0].set(h / (2 * (nq - 1)))
    w = w.at[-1].set(h / (2 * (nq - 1)))

    # base
    x_ref = jnp.linspace(0.0, 1.0, nq)
    phi_q = jnp.stack([1 - x_ref, x_ref], axis=1)

    p_q = phi_q @ u_cell[0]
    v_q = phi_q @ u_cell[1]

    # conversion
    p_q_phys = (c * S_star / S_q) * p_q
    v_q_phys = (c * S_q / S_star) * v_q

    Fq = jnp.stack([v_q_phys, p_q_phys], axis=1)

    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    V0 = jnp.sum(w[:, None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:, None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)


def surface_term_system(u_ext, S_ext, j, c, S_star):
    jp = j + 1

    S_left  = 0.5 * (S_ext[jp - 1] + S_ext[jp])
    S_right = 0.5 * (S_ext[jp]     + S_ext[jp + 1])

    UL_left  = u_ext[jp - 1, :, 1]
    UR_left  = u_ext[jp,     :, 0]
    UL_right = u_ext[jp,     :, 1]
    UR_right = u_ext[jp + 1, :, 0]

    f_left  = rusanov_flux(UL_left,  UR_left,  S_left,  c, S_star)
    f_right = rusanov_flux(UL_right, UR_right, S_right, c, S_star)

    S_term = jnp.zeros((2, 2))
    S_term = S_term.at[:, 0].set(-f_left)
    S_term = S_term.at[:, 1].set( f_right)
    return S_term


def dg_rhs_system(u_tilde_cells, x_nodes, c, Mp_inv, Mv_inv,
                  bc, phi, beta, Z, alpha, v_bc_tilde,
                  S_cells, S_star,
                  zeta, gamma, eps, kappa, omega_r, y, z,
                  S_quad):          
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_tilde_cells.shape[0]

    # Ghost cells
    if bc.type == "full":
        u_ext = apply_bc(
            u_tilde_cells, phi, beta, v_bc_tilde, Z, alpha,
            S_cells, c, S_star,zeta, gamma, eps, kappa, omega_r, y, z
        )
    elif bc.type == "right_free":
        u_ext = apply_bc_test(
            u_tilde_cells, phi, beta, v_bc_tilde, Z, alpha,
            S_cells, c, S_star,zeta, gamma, eps, kappa, omega_r, y, z
        )

    S_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])

    # Terme de surface
    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, S_ext, j, c, S_star)
    )(jnp.arange(N))

    # Terme volume — section évaluée exactement ✓
    V_all = jax.vmap(
    lambda Ue, S_q, h: local_volume_system(Ue, S_q, h, S_star, c)
    )(u_tilde_cells, S_quad, xRs - xLs)

    # Assemblage RHS
    def element_rhs(e):
        Vi, Si = V_all[e], S_all[e]
        rhs_p = Mp_inv[e] @ (Vi[0] - Si[0])
        rhs_v = Mv_inv[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_p, rhs_v], axis=0)

    return jax.vmap(element_rhs)(jnp.arange(N))