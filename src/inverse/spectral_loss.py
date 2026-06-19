import jax
import jax.numpy as jnp
from jax import lax


def stft_mag(x, n_fft, hop_length):
    x = jnp.asarray(x)

    if x.shape[0] < n_fft:
        x = jnp.pad(x, (0, n_fft - x.shape[0]))

    window = jnp.hanning(n_fft)
    n_frames = 1 + (x.shape[0] - n_fft) // hop_length

    def get_frame(i):
        start = i * hop_length
        frame = lax.dynamic_slice(x, (start,), (n_fft,))
        return frame * window

    frames = jax.vmap(get_frame)(jnp.arange(n_frames))
    return jnp.abs(jnp.fft.rfft(frames, axis=-1))


def spectral_loss_one_resolution(pred, target, n_fft, hop_length, dynamic_db=60.0):
    pred_mag = stft_mag(pred, n_fft, hop_length)
    target_mag = stft_mag(target, n_fft, hop_length)

    scale = jnp.max(target_mag) + 1e-12

    pred_mag_norm = pred_mag / scale
    target_mag_norm = target_mag / scale

    eps = 10.0 ** (-dynamic_db / 20.0)

    loss_lin = jnp.mean(jnp.abs(pred_mag_norm - target_mag_norm))

    pred_db = 20.0 * jnp.log10(pred_mag_norm + eps)
    target_db = 20.0 * jnp.log10(target_mag_norm + eps)

    loss_db = jnp.mean(jnp.abs(pred_db - target_db))

    return loss_lin + loss_db


def multi_resolution_spectral_loss(pred, target):
    T = pred.shape[0]

    resolutions = [
        (16, 8),
        (32, 16),
        (64, 32),
        (128, 64),
        (256, 128),
    ]

    resolutions = [(n_fft, hop) for n_fft, hop in resolutions if n_fft <= T]

    if len(resolutions) == 0:
        return jnp.array(0.0)

    losses = [
        spectral_loss_one_resolution(pred, target, n_fft, hop)
        for n_fft, hop in resolutions
    ]

    return sum(losses) / len(losses)

def spectrogram_db(x, dt, n_fft=512, hop_length=128, dynamic_db=80.0):
    mag = stft_mag(x, n_fft, hop_length)

    mag = mag / (jnp.max(mag) + 1e-12)
    eps = 10.0 ** (-dynamic_db / 20.0)

    spec_db = 20.0 * jnp.log10(mag + eps)

    freqs = jnp.fft.rfftfreq(n_fft, d=dt)
    times = jnp.arange(mag.shape[0]) * hop_length * dt

    return times, freqs, spec_db.T