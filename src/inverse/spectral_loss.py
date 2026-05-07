import jax.numpy as jnp


def stft_mag(x, n_fft, hop_length):
    """
    x : shape (N,)
    retourne |STFT(x)|
    """
    x = jnp.asarray(x)

    N = x.shape[0]

    # Si le signal est plus court que n_fft, on ajoute des zéros
    if N < n_fft:
        pad_width = n_fft - N
        x = jnp.pad(x, (0, pad_width))
        N = n_fft

    window = jnp.hanning(n_fft)

    n_frames = 1 + (N - n_fft) // hop_length

    frames = jnp.stack([
        x[i * hop_length : i * hop_length + n_fft] * window
        for i in range(n_frames)
    ])

    return jnp.abs(jnp.fft.rfft(frames, n=n_fft))


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