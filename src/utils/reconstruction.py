import jax
import jax.numpy as jnp
from src.numerics.dg.basis import phi_at
from src.numerics.dg.mesh import cell_edges_from_nodes



# ------------------------------------------------------------------------------------------------------------------------------
# reconstruction for plotting (physical variables)
# ------------------------------------------------------------------------------------------------------------------------------
def reconstruct_system(u_cells, x_nodes, x_plot,
                       section,
                       c=1.0,
                       S_star=1.0):

    xLs, xRs = cell_edges_from_nodes(x_nodes)
    h = xRs[0] - xLs[0]
    Ncells = u_cells.shape[0]

    idx = jnp.clip(jnp.floor((x_plot - xLs[0]) / h).astype(int), 0, Ncells-1)

    def eval_point(x, j):
        xL = xLs[j]
        xR = xRs[j]

        ph = phi_at(x, xL, xR)

        # conservative variables
        pt = jnp.dot(ph, u_cells[j,0])  # tilde p
        vt = jnp.dot(ph, u_cells[j,1])  # tilde v

        # section at point
        Sx = section(x)

        # back to physical variables
        p = (c*S_star/Sx) * pt
        v = (c*Sx/S_star) * vt

        return jnp.stack([p, v])

    UV = jax.vmap(eval_point)(x_plot, idx)

    return UV[:,0], UV[:,1]