import jax.numpy as jnp
from utils.util_func import best_time_shift

def compute_metrics(p_dg, p_ow, t_dg):
    dt_snap = float(t_dg[1] - t_dg[0])

    rel_l2 = jnp.linalg.norm(p_dg - p_ow) / (jnp.linalg.norm(p_ow) + 1e-12)

    rel_linf = jnp.max(jnp.abs(p_dg - p_ow)) / (
        jnp.max(jnp.abs(p_ow)) + 1e-12
    )

    lag_index, shift_time, corr = best_time_shift(p_dg, p_ow, dt_snap)

    if lag_index > 0:
        p_dg_s = p_dg[lag_index:]
        p_ow_s = p_ow[:-lag_index]
    elif lag_index < 0:
        k = -lag_index
        p_dg_s = p_dg[:-k]
        p_ow_s = p_ow[k:]
    else:
        p_dg_s = p_dg
        p_ow_s = p_ow

    rel_l2_shift = jnp.linalg.norm(p_dg_s - p_ow_s) / (
        jnp.linalg.norm(p_ow_s) + 1e-12
    )

    return {
        "rel_l2": float(rel_l2),
        "rel_linf": float(rel_linf),
        "rel_l2_shift": float(rel_l2_shift),
        "shift_time": float(shift_time),
        "corr_shifted": float(corr),
    }

