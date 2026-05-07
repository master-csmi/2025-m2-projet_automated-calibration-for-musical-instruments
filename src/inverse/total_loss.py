# --------------------------------------------------------------------------
# Loss de base : L2 temporel + FFT fréquentiel
# --------------------------------------------------------------------------
    
from utils.param_func import set_param
import jax.numpy as jnp
from src.inverse.spectral_loss import multi_resolution_spectral_loss
from utils.solve import forward_snapshots
    
def loss_fn_signal(pred, target):
    loss_time = jnp.mean((pred - target) ** 2) / (jnp.mean(target ** 2) + 1e-12)

    loss_spec = multi_resolution_spectral_loss(pred, target)

    return 0.1 * loss_time + loss_spec

def loss_fn(param, key, data_init, Nx_train, c, target_snaps, INVERSE_PARAM, GEO_KEYS, solve_kwargs):
    data = set_param(data_init, INVERSE_PARAM, param, GEO_KEYS)

    pred = forward_snapshots(data, Nx_train, c, **solve_kwargs)

    return loss_fn_signal(pred, target_snaps)