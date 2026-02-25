import jax
import jax.numpy as jnp
from utils.util_function import pressure_func, l

# ----------------------------------------------------------------------------------------------------------------------
# Analytic / reference solution for constant section
# Initial condition: p0(x), v0 = 0
# Boundary condition: impedance with boundary ODE
# Time integration of boundary ODE: Euler or RK2 (Heun)
# ----------------------------------------------------------------------------------------------------------------------

def exact_solution_characteristics(
    x, t, p0_fun, c, L,
    alpha, beta, Z,
    y0, dy0, gamma, epsilon, kappa, f_r, Qr, zeta,
    dt=1e-4,
    method="rk2",   # "euler" or "rk2"
):
    # ----------------------------
    # Coefficients
    # ----------------------------
    a = (1.0 - beta / Z) / (1.0 + beta / Z)
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / Z)

    c1 = jnp.sqrt(alpha) / (2.0 * Z)
    c2 = c1 * b

    omega_r = 2 * jnp.pi * f_r
    # ----------------------------
    # Left-going wave (exact)
    # ----------------------------
    w_plus = p0_fun(x - c * t)

    # ----------------------------
    # Boundary dynamics at x = L
    # ----------------------------
    Nt = int(jnp.ceil(t / dt))
    t_grid = jnp.linspace(0.0, t, Nt)

    # Outgoing wave at boundary
    w_plus_L = p0_fun(L - c * t_grid)

    phi = 0.0
    w_minus_L = []
    

    for wp in w_plus_L:

        def rhs(phi_val):
            return -c1 *(1.0+a) * wp - c2 * phi_val

        if method == "euler":
            # Euler explicit
            phi = phi + dt * rhs(phi)

        elif method == "rk2":
            # RK2 
            k1 = rhs(phi)
            phi_star = phi + dt * k1
            k2 = rhs(phi_star)
            phi = phi + 0.5 * dt * (k1 + k2)

        else:
            raise ValueError(f"Unknown method '{method}' (use 'euler' or 'rk2')")

        wm = a * wp + b * phi
        w_minus_L.append(wm)

    w_minus_L = jnp.array(w_minus_L)

    # ----------------------------
    # Reflected wave propagation
    # ----------------------------
    t_ref = t - (L - x) / c

    w_minus_ref = jnp.where(
        t_ref > 0.0,
        jnp.interp(t_ref, t_grid, w_minus_L, left=0.0, right=0.0),
        0.0
    )

    # ----------------------------
    # Left BC (x=0) reed dynamics
    # ----------------------------
    y = y0
    dy = dy0
    w_plus_reed = []
    for tn in t_grid:
        # velocity entering the domain from reed
        v_plus = zeta * l(y) * pressure_func(gamma - w_plus) + epsilon * kappa / omega_r * dy
        w_plus_reed.append(v_plus)

        # update reed ODE (Euler integration for simplicity)
        ddy = (1/omega_r**2) * (-y - (1/(Qr*omega_r))*dy + epsilon*(gamma - v_plus))
        dy = dy + dt * ddy
        y = y + dt * dy

    w_plus_reed = jnp.array(w_plus_reed)

    # Incoming right-going wave from left BC
    t_ref_left = t - x / c
    w_plus_left = jax.vmap(
    lambda tref, fp: jnp.interp(tref, t_grid, fp, left=0.0, right=0.0)
    )(t_ref_left, w_plus_reed.T)

    # ----------------------------
    # Initial right-going wave
    # ----------------------------
    w_minus_init = p0_fun(x + c * t)

    # ----------------------------
    # Total right-going wave
    # ----------------------------
    w_minus = w_minus_init + w_minus_ref
    w_plus_total = w_plus + w_plus_left

    # ----------------------------
    # Reconstruction
    # ----------------------------
    p = 0.5 * (w_plus_total + w_minus)
    v = 0.5 * (w_plus_total - w_minus)

    return p, v
