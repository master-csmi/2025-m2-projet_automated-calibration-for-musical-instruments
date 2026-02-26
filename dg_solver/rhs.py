import jax
import jax.numpy as jnp
from dg_solver.mesh import cell_edges_from_nodes
from dg_solver.basis import vphi_at
from bc.bc import apply_bc, apply_bc_neumann
from utils.flux import rusanov_flux


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          local volume term for system
# ------------------------------------------------------------------------------------------------------------------------------
def local_volume_system(u_cell, xL, xR, nq=24):
    h = xR - xL
    xq = jnp.linspace(xL, xR, nq)
    w  = jnp.ones(nq) * (h/(nq-1))
    w  = w.at[0].set(h/(2*(nq-1)))
    w  = w.at[-1].set(h/(2*(nq-1)))

    phi_q = vphi_at(xq, xL, xR)
    p_q = phi_q @ u_cell[0]
    v_q = phi_q @ u_cell[1]

    Fq = jnp.stack([v_q, p_q], axis=1)

    # dérivées exactes des fonctions de base
    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    # Intégration
    V0 = jnp.sum(w[:,None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:,None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)  


def surface_term_system(u_ext, j, c=1.0):
    jp = j + 1

    # Left interface
    UL_left  = u_ext[jp-1, :, 1]
    UR_left  = u_ext[jp,   :, 0]


    # Right interface
    UL_right = u_ext[jp,   :, 1]
    UR_right = u_ext[jp+1, :, 0]


    # Flux Rusanov conservatif
    f_left  = rusanov_flux(UL_left, UR_left, c)
    f_right = rusanov_flux(UL_right, UR_right, c)

    S_term = jnp.zeros((2,2))
    S_term = S_term.at[:,0].set(-f_left)
    S_term = S_term.at[:,1].set( f_right)
    return S_term



def dg_rhs_system(u_cells, x_nodes, c, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]

    # add ghost cells according to BC
    if bc.type == "dirichlet":
        u_ext, dy_new, ddy_new = apply_bc(u_cells, phi, beta, Z, alpha, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta)
    elif bc.type == "neumann":
        u_ext = apply_bc_neumann(u_cells)
 

    # Surface term (fluxes)
    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, j, c=c)
    )(jnp.arange(N))

    # Volume term
    N = xLs.shape[0]

    V_all = jax.vmap(
        lambda e: local_volume_system(
            u_cells[e],
            xLs[e],
            xRs[e],
            nq=24
        )
    )(jnp.arange(N))

    # assemble RHS cell by cell
    def element_rhs(e):
        Vi = V_all[e]
        Si = S_all[e]
        rhs_p = Mp_inv[e] @ (Vi[0] - Si[0])
        rhs_v = Mv_inv[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_p, rhs_v], axis=0)

    RHS = jax.vmap(element_rhs)(jnp.arange(N))
    return RHS