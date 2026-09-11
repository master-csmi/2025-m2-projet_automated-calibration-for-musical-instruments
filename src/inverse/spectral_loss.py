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


def multi_resolution_spectral_loss(
    pred,
    target,
    resolutions=None,
    dynamic_db=60.0,
    allow_padding=False,
):
    T = pred.shape[0]

    if resolutions is None:
        resolutions = [
            (16, 8),
            (32, 16),
            (64, 32),
            (128, 64),
            (256, 128),
        ]

    if not allow_padding:
        resolutions = [(n_fft, hop) for n_fft, hop in resolutions if n_fft <= T]

    if len(resolutions) == 0:
        return jnp.array(0.0)

    losses = [
        spectral_loss_one_resolution(
            pred,
            target,
            n_fft,
            hop,
            dynamic_db=dynamic_db,
        )
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

def spectral_residuals_one_resolution(
    pred,
    target,
    n_fft,
    hop_length,
    dynamic_db=60.0,
    include_linear=True,
    include_db=True,
):
    """
    Retourne un vecteur de résidus spectraux compatible avec Gauss-Newton.

    Les résidus linéaires et en dB sont normalisés pour que chaque bloc
    contribue par sa moyenne quadratique plutôt que par sa taille brute.
    """
    pred_mag = stft_mag(pred, n_fft, hop_length)
    target_mag = stft_mag(target, n_fft, hop_length)

    # Normalisation identique à celle de la loss actuelle :
    # l'échelle dépend uniquement de la cible.
    scale = jnp.max(target_mag) + 1e-12

    pred_mag_norm = pred_mag / scale
    target_mag_norm = target_mag / scale

    eps = 10.0 ** (-dynamic_db / 20.0)

    residual_blocks = []

    if include_linear:
        residual_linear = pred_mag_norm - target_mag_norm

        # Ainsi ||r||² correspond à une moyenne quadratique.
        residual_linear = residual_linear / jnp.sqrt(residual_linear.size)

        residual_blocks.append(residual_linear.reshape(-1))

    if include_db:
        pred_db = 20.0 * jnp.log10(pred_mag_norm + eps)
        target_db = 20.0 * jnp.log10(target_mag_norm + eps)

        residual_db = pred_db - target_db
        residual_db = residual_db / jnp.sqrt(residual_db.size)

        residual_blocks.append(residual_db.reshape(-1))

    return jnp.concatenate(residual_blocks)

def multi_resolution_spectral_residuals(
    pred,
    target,
    resolutions=None,
    dynamic_db=60.0,
    allow_padding=False,
    include_linear=True,
    include_db=True,
):
    """
    Concatène les résidus spectraux de toutes les résolutions.

    La fonction retourne un vecteur r tel que le critère Gauss-Newton soit :

        loss_GN = 0.5 * ||r||²
    """
    signal_length = pred.shape[0]

    if resolutions is None:
        resolutions = [
            (16, 8),
            (32, 16),
            (64, 32),
            (128, 64),
            (256, 128),
        ]

    if not allow_padding:
        resolutions = [
            (n_fft, hop)
            for n_fft, hop in resolutions
            if n_fft <= signal_length
        ]

    if len(resolutions) == 0:
        return jnp.zeros((1,), dtype=pred.dtype)

    residual_blocks = [
        spectral_residuals_one_resolution(
            pred,
            target,
            n_fft,
            hop,
            dynamic_db=dynamic_db,
            include_linear=include_linear,
            include_db=include_db,
        )
        for n_fft, hop in resolutions
    ]

    # Chaque résolution reçoit approximativement le même poids.
    n_resolutions = len(residual_blocks)

    return jnp.concatenate(
        [
            residual / jnp.sqrt(n_resolutions)
            for residual in residual_blocks
        ]
    )