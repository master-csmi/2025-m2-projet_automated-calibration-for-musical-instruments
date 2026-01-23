import jax
import jax.numpy as jnp
from dg_solver.basis import phi_at
from dg_solver.mesh import cell_edges_from_nodes

# ------------------------------------------------------------------------------------------------------------------------------
#                                                           reconstruction for plotting
# ------------------------------------------------------------------------------------------------------------------------------
def reconstruct_system(u_cells, x_nodes, x_plot):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    h = xRs[0]-xLs[0]
    Ncells = u_cells.shape[0]
    idx = jnp.clip(jnp.floor(x_plot / h).astype(int), 0, Ncells-1)
    def eval_point(x, j):
        xL = xLs[j]; xR = xRs[j]
        ph = phi_at(x, xL, xR)
        # compute p and v
        p = jnp.dot(ph, u_cells[j,0])
        v = jnp.dot(ph, u_cells[j,1])
        return jnp.stack([p, v])
    UV = jax.vmap(eval_point)(x_plot, idx)  # (len(x_plot),2)
    return UV[:,0], UV[:,1] # p_rec, v_rec