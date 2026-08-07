from utils.param_func import set_param
import jax.numpy as jnp
import equinox as eqx
from src.inverse.spectral_loss import multi_resolution_spectral_loss
from utils.solve import forward_snapshots

def envelope_rms_loss(pred, target, win=256, hop=64):
    """
    Compare l'enveloppe RMS locale des signaux.

    Utile pour Qr, car Qr agit surtout sur l'amortissement,
    donc sur l'évolution lente de l'amplitude.
    """
    n = pred.shape[0]

    # Si le signal est plus court que la fenêtre, on utilise toute la longueur.
    win = min(win, n)
    hop = min(hop, win)

    n_frames = (n - win) // hop + 1

    idx = jnp.arange(win)[None, :] + hop * jnp.arange(n_frames)[:, None]

    pred_frames = pred[idx]
    target_frames = target[idx]

    env_pred = jnp.sqrt(jnp.mean(pred_frames**2, axis=1) + 1e-12)
    env_target = jnp.sqrt(jnp.mean(target_frames**2, axis=1) + 1e-12)

    return jnp.mean((env_pred - env_target) ** 2) / (
        jnp.mean(env_target**2) + 1e-12
    )

def loss_fn_signal(
    pred,
    target,
    time_weight=0.0,
    spec_weight=1.0,
    env_weight=0.0,
    stft_resolutions=None,
    stft_dynamic_db=60.0,
    stft_allow_padding=False,
    env_win=256,
    env_hop=64,
):
    loss = 0.0

    if time_weight != 0.0:
        loss_time = jnp.mean((pred - target) ** 2) / (
            jnp.mean(target ** 2) + 1e-12
        )
        loss = loss + time_weight * loss_time

    if spec_weight != 0.0:
        loss_spec = multi_resolution_spectral_loss(
            pred,
            target,
            resolutions=stft_resolutions,
            dynamic_db=stft_dynamic_db,
            allow_padding=stft_allow_padding,
        )
        loss = loss + spec_weight * loss_spec

    if env_weight != 0.0:
        loss_env = envelope_rms_loss(
            pred,
            target,
            win=env_win,
            hop=env_hop,
        )
        loss = loss + env_weight * loss_env

    return loss



def replace_l(data, ell_nn):
    return eqx.tree_at(lambda d: d.l, data, ell_nn)
    

def loss_scalars_only(
    theta,
    data_init,
    geometry,
    c,
    target_snaps,
    inverse_params,
    scales,
    geo_keys,
    solve_kwargs,
    time_weight=0.0,
    spec_weight=1.0,
    env_weight=0.0,
    stft_resolutions=None,
    stft_dynamic_db=60.0,
    stft_allow_padding=False,
    env_win=256,
    env_hop=64,
):
    params_phys = theta * scales

    data = data_init
    for name, value in zip(inverse_params, params_phys):
        data = set_param(data, name, value, geo_keys)

    pred = forward_snapshots(
        data,
        geometry,
        c,
        **solve_kwargs,
    )

    return loss_fn_signal(
        pred,
        target_snaps,
        time_weight=time_weight,
        spec_weight=spec_weight,
        env_weight=env_weight,
        stft_resolutions=stft_resolutions,
        stft_dynamic_db=stft_dynamic_db,
        stft_allow_padding=stft_allow_padding,
        env_win=env_win,
        env_hop=env_hop,
    )


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
