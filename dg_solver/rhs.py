import jax
import jax.numpy as jnp
from dg_solver.mesh import cell_edges_from_nodes
from dg_solver.basis import vphi_at
from bc.bc import apply_bc, apply_bc_neumann
from utils.flux import rusanov_flux


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          local volume term for system
# ------------------------------------------------------------------------------------------------------------------------------
def local_volume_system(u_cell, xL, xR, A, nq=24):
    h = xR - xL
    xq = jnp.linspace(xL, xR, nq)
    w  = jnp.ones(nq) * (h/(nq-1))
    w  = w.at[0].set(h/(2*(nq-1)))
    w  = w.at[-1].set(h/(2*(nq-1)))

    # reconstruction U(x)
    phi_q = vphi_at(xq, xL, xR)       
    p_q = phi_q @ u_cell[0]           
    v_q = phi_q @ u_cell[1]           
    Uq = jnp.stack([p_q, v_q], axis=1)

    Fq = Uq @ A.T                    

    # exact derivative of basis functions
    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    # Integration of flux * dphi
    V0 = jnp.sum(w[:,None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:,None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)   


v_local_volume_system = jax.vmap(local_volume_system, in_axes=(0, 0, 0, None, None))

def surface_term_system(u_ext, j, c=1.0):
    """
    Compute DG surface term for cell j with S(x) variable.

    u_ext   : array (N+2,2,2) with ghost cells
    S_ext   : array (N+2,) S at nodes/interfaces
    j       : cell index in original u_cells
    """
    jp = j + 1  # offset due to ghost cell

    # Left interface
    UL_left  = u_ext[jp-1, :, 1]  # right node of left cell
    UR_left  = u_ext[jp,   :, 0]  # left node of current cell
   
    

    # Right interface
    UL_right = u_ext[jp,   :, 1]  # right node of current cell
    UR_right = u_ext[jp+1, :, 0]  # left node of right cell
    

    f_left  = rusanov_flux(UL_left, UR_left, c)
    f_right = rusanov_flux(UL_right, UR_right, c)
    # assemble surface term (2x2)
    S_term = jnp.zeros((2,2))
    S_term = S_term.at[:,0].set(-f_left)
    S_term = S_term.at[:,1].set( f_right)
    return S_term


v_surface_term_system = jax.vmap(surface_term_system, in_axes=(None, 0, None))

<<<<<<< HEAD
def dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi, beta, Z, T, alpha, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt):
=======
def dg_rhs_system(u_cells, x_nodes, S_cells, c, A, Mp_inv, Mv_inv, bc, phi, beta, Z, alpha):
>>>>>>> 0bea72a3f55d5c06fb96d589d2a96e42c9fd6484
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]

    # add ghost cells according to BC
    if bc.type == "dirichlet":
<<<<<<< HEAD
        u_ext, y_new, dy_new = apply_bc(u_cells, bc.left, phi, beta, Z, T, alpha, y, dy, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func, dt)
=======
        u_ext = apply_bc(u_cells, bc.left, phi, beta, Z, alpha)
>>>>>>> 0bea72a3f55d5c06fb96d589d2a96e42c9fd6484
    elif bc.type == "neumann":
        u_ext = apply_bc_neumann(u_cells)

    # S at nodes/interfaces with ghost cells
    S_nodes_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])  

    # Surface term (fluxes)
    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, j, c=c)
    )(jnp.arange(N))

    # Volume term
    V_all = jax.vmap(
        lambda Ue, xL, xR: local_volume_system(Ue, xL, xR, A, 24)
    )(u_cells, xLs, xRs)

    # assemble RHS cell by cell
    def element_rhs(e):
        Vi = V_all[e]
        Si = S_all[e]
        rhs_p = Mp_inv[e] @ (Vi[0] - Si[0])
        rhs_v = Mv_inv[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_p, rhs_v], axis=0)

    RHS = jax.vmap(element_rhs)(jnp.arange(N))
    return RHS