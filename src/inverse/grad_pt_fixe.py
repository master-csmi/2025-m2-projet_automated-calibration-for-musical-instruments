import jax
import jax.numpy as jnp

from src.utils.util_func import pressure_func as F_func

def pressure_func_derivative(delta_p, eps=1e-8):
    """
    Dérivée de la régularisation continue :
        delta_p / (delta_p**2 + eps**2)**0.25
    """
    q = delta_p**2 + eps**2
    return (eps**2 + 0.5 * delta_p**2) / (q**1.25)

def G_wplus(
    w_plus, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r, ell_y, y_t
):
    p_tilde_interface = (w_plus - w_minus) / (2.0 * k_char)
    p_interface = (c * S_star / S_L) * p_tilde_interface

    v_bc = (
        zeta * ell_y * F_func(gamma - p_interface)
        + eps * kappa / omega_r * y_t
    )

    v_bc_tilde = (S_star / (c * S_L)) * v_bc

    return 2.0 * v_bc_tilde - w_minus

def dG_wplus_dw(
    w_plus, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r, ell_y, y_t
):
    """
    Dérivée analytique de G_wplus par rapport à w_plus.

    Certains arguments sont conservés afin d'avoir la même interface
    que G_wplus, même s'ils n'interviennent pas dans la dérivée.
    """
    p_tilde_interface = (
        w_plus - w_minus
    ) / (2.0 * k_char)

    p_interface = (
        c * S_star / S_L
    ) * p_tilde_interface

    delta_p = gamma - p_interface

    # dp_interface / dw_plus
    dp_dw = (
        c * S_star / S_L
    ) / (2.0 * k_char)

    # G = 2 * S_star/(c*S_L) * v_bc - w_minus
    scale = 2.0 * S_star / (c * S_L)

    return (
        -scale
        * zeta
        * ell_y
        * pressure_func_derivative(delta_p)
        * dp_dw
    )


def residual_wplus(
    w, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r, ell_y, y_t
):
    return w - G_wplus(
        w, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r, ell_y, y_t
    )


def safe_denominator(x, eps=1e-10):
    return jnp.where(
        jnp.abs(x) < eps,
        jnp.where(x >= 0.0, eps, -eps),
        x
    )


# ============================================================
# Ancienne méthode : point fixe
# ============================================================

def solve_wplus_fixed_point(
    w0, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r,
    ell_y, y_t,
    n_iter=75,
    theta=0.05,
):
    def body(_, w):
        G = G_wplus(
            w, w_minus, k_char, S_L, c, S_star,
            zeta, gamma, eps, kappa, omega_r, ell_y, y_t
        )
        return (1.0 - theta) * w + theta * G

    return jax.lax.fori_loop(0, n_iter, body, w0)


# ============================================================
# Nouvelle méthode : Newton amorti
# ============================================================

def solve_wplus_newton_raw(
    w0, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r,
    ell_y, y_t,
    n_iter=8,
    damping=0.8,
    max_step=0.5,
):
    params = (
        w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )

    def body(_, w):
        R = residual_wplus(w, *params)

        dG_dw = dG_wplus_dw(
            w, w_minus, k_char, S_L, c, S_star,
            zeta, gamma, eps, kappa, omega_r,
            ell_y, y_t
        )

        dR_dw = safe_denominator(1.0 - dG_dw)

        step = R / dR_dw
        step = jnp.clip(step, -max_step, max_step)

        return w - damping * step

    return jax.lax.fori_loop(0, n_iter, body, w0)


@jax.custom_vjp
def solve_wplus_newton(
    w0, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r,
    ell_y, y_t
):
    return solve_wplus_newton_raw(
        w0, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )


def solve_wplus_newton_fwd(
    w0, w_minus, k_char, S_L, c, S_star,
    zeta, gamma, eps, kappa, omega_r,
    ell_y, y_t
):
    w_star = solve_wplus_newton_raw(
        w0, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )

    saved = (
        w_star, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )

    return w_star, saved


def solve_wplus_newton_bwd(saved, grad_w_star):
    (
        w_star, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    ) = saved

    dG_dw = dG_wplus_dw(
        w_star, w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )

    denom = safe_denominator(1.0 - dG_dw)

    lam = grad_w_star / denom

    def G_params(
        w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    ):
        return G_wplus(
            w_star, w_minus, k_char, S_L, c, S_star,
            zeta, gamma, eps, kappa, omega_r, ell_y, y_t
        )

    _, vjp_fun = jax.vjp(
        G_params,
        w_minus, k_char, S_L, c, S_star,
        zeta, gamma, eps, kappa, omega_r,
        ell_y, y_t
    )

    grads_params = vjp_fun(lam)

    grad_w0 = jnp.zeros_like(w_star)

    return (
        grad_w0,
        *grads_params
    )


solve_wplus_newton.defvjp(
    solve_wplus_newton_fwd,
    solve_wplus_newton_bwd
)
