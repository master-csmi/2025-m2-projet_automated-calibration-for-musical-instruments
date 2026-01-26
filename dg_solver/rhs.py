import jax
import jax.numpy as jnp
from dg_solver.mesh import cell_edges_from_nodes
from dg_solver.basis import vphi_at
from bc.bc import apply_bc, apply_bc_neumann
from utils.flux import rusanov_hyperbolic


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          local volume term for system
# ------------------------------------------------------------------------------------------------------------------------------
def local_volume_system(u_cell, xL, xR, S_cell, c=1.0, S_star=1.0, nq=24):
    h = xR - xL
    xq = jnp.linspace(xL, xR, nq)
    w  = jnp.ones(nq) * (h/(nq-1))
    w  = w.at[0].set(h/(2*(nq-1)))
    w  = w.at[-1].set(h/(2*(nq-1)))

    # reconstruction U(x)
    phi_q = vphi_at(xq, xL, xR)       
    p_q = phi_q @ u_cell[0]           
    v_q = phi_q @ u_cell[1]           

    a = S_cell / (c * S_star)
    b = c * S_star / S_cell

    Fq = jnp.stack([
        a * v_q,
        b * p_q
    ], axis=1)                  

    # exact derivative of basis functions
    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    # Integration of flux * dphi
    V0 = jnp.sum(w[:,None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:,None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)   


v_local_volume_system = jax.vmap(local_volume_system, in_axes=(0, 0, 0, None, None, None, None))

def surface_term_system(u_ext, j, S_cells, c=1.0):
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
    

    S_face_L = 0.5 * (S_cells[j-1] + S_cells[j])
    S_face_R = 0.5 * (S_cells[j]   + S_cells[j+1])

    f_left  = rusanov_hyperbolic(UL_left, UR_left, S_face_L, c)
    f_right = rusanov_hyperbolic(UL_right, UR_right, S_face_R, c)
    # assemble surface term (2x2)
    S_term = jnp.zeros((2,2))
    S_term = S_term.at[:,0].set(-f_left)
    S_term = S_term.at[:,1].set( f_right)
    return S_term


v_surface_term_system = jax.vmap(surface_term_system, in_axes=(None, 0, None, None))

def dg_rhs_system(u_cells, x_nodes, S_cells, c, Mp_inv, Mv_inv, bc, phi, beta, Z,T, alpha):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]

    # add ghost cells according to BC
    if bc.type == "dirichlet":
        u_ext = apply_bc(u_cells, bc.left, phi, beta, Z, T, alpha, S_cells, S_star=1.0, c=c)
    elif bc.type == "neumann":
        u_ext = apply_bc_neumann(u_cells)

   

    # Surface term (fluxes)
    S_all = jax.vmap(
        lambda j: surface_term_system(u_ext, j, S_cells, c=c)
    )(jnp.arange(N))

    # Volume term
    V_all = jax.vmap(
        lambda Ue, xL, xR, S: local_volume_system(Ue, xL, xR, S, c, 1.0, 24)
    )(u_cells, xLs, xRs, S_cells)

    # assemble RHS cell by cell
    def element_rhs(e):
        Vi = V_all[e]
        Si = S_all[e]
        rhs_p = Mp_inv[e] @ (Vi[0] - Si[0])
        rhs_v = Mv_inv[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_p, rhs_v], axis=0)

    RHS = jax.vmap(element_rhs)(jnp.arange(N))
    return RHS