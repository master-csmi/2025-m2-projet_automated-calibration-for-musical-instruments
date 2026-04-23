import jax.numpy as jnp
import jax
from typing import Literal

@jax.jit(static_argnames=("shape",))
def pressure_at_mouth(
    gamma_final: float,
    t_attack: float = 0.2,
    sharpness: float = 10.0,
    t: jnp.ndarray = None,
    shape: Literal["exponential", "sigmoid", "linear"] = "exponential"
) -> jnp.ndarray:
    """
    Simule la pression d'air soufflée par un instrumentaliste.

    Paramètres
    ----------
    gamma_final : float
        Pression normalisée cible (plateau final).
    t_attack : float
        Durée de la montée en pression (secondes).
    sharpness : float
        Contrôle la transition vers le plateau (plus grand = plus rapide).
    t : jnp.ndarray
        Vecteur temps en secondes.
    shape : str
        Forme de l'attaque : "exponential", "sigmoid" ou "linear".

    Retourne
    --------
    gamma : jnp.ndarray  shape (N,)
        Pression normalisée au cours du temps.
    """
    assert shape in ["exponential", "sigmoid", "linear"]
    assert t is not None

    if shape == "exponential":
        tau    = t_attack / 100.0
        attack = gamma_final * (1.0 - jnp.exp(-t / tau))

    elif shape == "sigmoid":
        k      = 10.0 / t_attack
        t0     = t_attack / 2.0
        attack = gamma_final / (1.0 + jnp.exp(-k * (t - t0)))

    elif shape == "linear":
        attack = gamma_final * jnp.clip(t / t_attack, 0.0, 1.0)

    # Transition différentiable vers le plateau
    w_end     = jax.nn.sigmoid(sharpness * (t - t_attack))
    gamma     = (1.0 - w_end) * attack + w_end * gamma_final

    return t,gamma

@jax.jit
def pressure_at_mouth_alexis(
    gamma_final: float,
    t_attack: float = 0.2,
    t: jnp.ndarray = None,
):
    assert t is not None

    gamma_attack = gamma_final * (1.0 - jnp.cos(jnp.pi * t / t_attack))/2

    gamma = jnp.where(
        t < t_attack,
        gamma_attack,
        gamma_final
    )

    return gamma