import jax
import jax.numpy as jnp
from numerics.time_integrators.rk2 import time_integrate_rk2
from utils.reconstruction import reconstruct_system


def eval_pressure(u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0,
            y0, z0,
            physical_data,
            S_cells, S_star, S_quad,
            snapshot_steps,
            gamma_target,
            x_plot):
    
    (_, _, _, _, _, _, _, _,
         u_tilde_snaps_new) = time_integrate_rk2(
            u0, x_nodes, c,
            dt, nsteps, Mp_inv, Mv_inv,
            bc, phi0,
            y0, z0,
            physical_data,
            S_cells=S_cells, S_star=S_star, S_quad=S_quad,
            snapshot_steps=snapshot_steps,
            gamma_target=gamma_target
        )
    
    # Reconstruction de p au pavillon pour tous les snapshots
    p_all_new, _ = jax.vmap(
        lambda u_T: reconstruct_system(u_T, x_nodes, x_plot, physical_data.section, c, S_star)
    )(u_tilde_snaps_new)

    p_bell_new = p_all_new[:, -1]

    # Scalaire : moyenne sur le régime établi (deuxième moitié)
    i_s = p_bell_new.shape[0] // 2
    return jnp.mean(p_bell_new[i_s:])