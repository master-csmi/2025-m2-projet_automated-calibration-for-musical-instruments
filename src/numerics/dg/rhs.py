import jax
import jax.numpy as jnp
from src.numerics.dg.mesh import cell_edges_from_nodes
from src.numerics.dg.basis import vphi_at
from src.physics.bc import apply_bc, apply_bc_test, apply_bc_fixed
from src.numerics.dg.flux import rusanov_flux


def local_volume_system(u_cell, S_q, h, S_star, c):
    nq = S_q.shape[0]

    # poids trapézoïdaux
    w = jnp.ones(nq) * (h / (nq - 1))
    w = w.at[0].set(h / (2 * (nq - 1)))
    w = w.at[-1].set(h / (2 * (nq - 1)))

    p_phys = (c * S_star / S_q) * u_cell[0]
    v_phys = (c * S_q / S_star) * u_cell[1]

    flux_sum = 0.5 * jnp.array([
        v_phys[0] + v_phys[1],
        p_phys[0] + p_phys[1],
    ])

    return jnp.stack([-flux_sum, flux_sum], axis=1)


def surface_terms_system(u_ext, S_ext, c, S_star):
    """
    Calcule les flux des N+1 interfaces une seule fois.

    u_ext : (N+2, 2, 2)
    S_ext : (N+1,) sections aux interfaces, ou ancien format (N+2,)
    retourne S_all : (N, 2, 2)
    """

    # États situés de part et d'autre des N+1 interfaces
    U_left = u_ext[:-1, :, 1]   # (N+1, 2)
    U_right = u_ext[1:, :, 0]   # (N+1, 2)

    if S_ext.shape[0] == U_left.shape[0]:
        S_interfaces = S_ext
    else:
        # Ancien format: sections par cellule et cellules fantomes.
        S_interfaces = 0.5 * (S_ext[:-1] + S_ext[1:])

    # Chaque flux d'interface est calculé exactement une fois
    fluxes = jax.vmap(
        rusanov_flux,
        in_axes=(0, 0, 0, None, None),
    )(
        U_left,
        U_right,
        S_interfaces,
        c,
        S_star,
    )  # (N+1, 2)

    # Pour la cellule j :
    #   colonne 0 = -flux à l'interface gauche
    #   colonne 1 = +flux à l'interface droite
    return jnp.stack(
        [-fluxes[:-1], fluxes[1:]],
        axis=-1,
    )  # (N, 2, 2)


def dg_rhs_system(u_tilde_cells, x_nodes, c, Mp_inv, Mv_inv,
                  bc, phi, beta, Z, alpha, v_bc_tilde,
                  S_cells, S_star,S_ext,
                  zeta, gamma, eps, kappa, omega_r, y, z, opening,
                  S_quad):          
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_tilde_cells.shape[0]
    # Ghost cells
    if bc.type == "full":
        u_ext = apply_bc_fixed(
            u_tilde_cells, phi, beta, Z, alpha,
            S_cells, c, S_star,zeta, gamma, eps, kappa, omega_r, y, z, opening
        )
    elif bc.type == "right_free":
        u_ext = apply_bc_test(
            u_tilde_cells, phi, beta, v_bc_tilde, Z, alpha,
            S_cells, c, S_star,zeta, gamma, eps, kappa, omega_r, y, z
        )



    # Terme de surface
    S_all = surface_terms_system(
        u_ext,
        S_ext,
        c,
        S_star,
    )

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

