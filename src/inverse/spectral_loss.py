import jax.numpy as jnp
import jax
from jax import lax


def stft_mag(x, n_fft, hop_length):

    # Hanning window 
    window = jnp.hanning(n_fft)

    n_frames = 1 + (x.shape[0] - n_fft) // hop_length

    def get_frame(i):
        start = i * hop_length
        frame = lax.dynamic_slice(x, (start,), (n_fft,))
        return frame * window

    frames = jax.vmap(get_frame)(jnp.arange(n_frames))

    return jnp.abs(jnp.fft.rfft(frames))


def spectral_loss_one_resolution(pred, target, n_fft, hop_length, eps=1e-7):
    pred_mag = stft_mag(pred, n_fft, hop_length)
    target_mag = stft_mag(target, n_fft, hop_length)

    # échelle linéaire
    loss_lin = jnp.mean(jnp.abs(pred_mag - target_mag))

    # échelle logarithmique
    loss_log = jnp.mean(
        jnp.abs(jnp.log(pred_mag + eps) - jnp.log(target_mag + eps))
    )

    return loss_lin + loss_log


def multi_resolution_spectral_loss(pred, target):
    resolutions = [
        (64, 16),
        (128, 32),
        (256, 64),
    ]

    losses = [
        spectral_loss_one_resolution(pred, target, n_fft, hop)
        for n_fft, hop in resolutions
    ]

    return sum(losses) / len(losses)