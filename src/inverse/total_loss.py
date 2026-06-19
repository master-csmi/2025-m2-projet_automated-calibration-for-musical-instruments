from utils.param_func import set_param
import jax.numpy as jnp
import equinox as eqx
from src.inverse.spectral_loss import multi_resolution_spectral_loss
from utils.solve import forward_snapshots


def loss_fn_signal(pred, target):
    loss_time = jnp.mean((pred - target) ** 2) / (jnp.mean(target ** 2) + 1e-12)
    loss_spec = multi_resolution_spectral_loss(pred, target)

    return 0.0 * loss_time +  loss_spec



def replace_l(data, ell_nn):
    return eqx.tree_at(lambda d: d.l, data, ell_nn)
    

def loss_scalars_only(
    theta,
    data_init,
    Nx,
    c,
    target_snaps,
    inverse_params,
    scales,
    geo_keys,
    solve_kwargs,
):
    params_phys = theta * scales

    data = data_init
    for name, value in zip(inverse_params, params_phys):
        data = set_param(data, name, value, geo_keys)

    pred = forward_snapshots(
        data,
        Nx,
        c,
        **solve_kwargs,
    )

    return loss_fn_signal(pred, target_snaps)


def loss_fn(
    model,
    data_init,
    Nx_train,
    c,
    target_snaps,
    inverse_params,
    scales,
    geo_keys,
    solve_kwargs_train,
):
    params_phys = model.theta * scales

    data = data_init

    for name, value in zip(inverse_params, params_phys):
        data = set_param(data, name, value, geo_keys)

    data = replace_l(data, model.ell_nn)

    pred = forward_snapshots(
        data,
        Nx_train,
        c,
        **solve_kwargs_train
    )

    return loss_fn_signal(pred, target_snaps)