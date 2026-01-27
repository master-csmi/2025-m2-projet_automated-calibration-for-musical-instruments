import jax
import jax.numpy as jnp

def exact_solution_characteristics_reed(x, t, p0_fun, c, L, alpha, beta, Z, T, dt,
                                        y0, dy0, gamma, epsilon, kappa, omega_r, Qr, zeta, l_func, F_func):
    """
    Analytic solution of 1D wave system with Gaussian initial condition and
    BCs: right end with phi (like before), left end with reed dynamics (y, dy, l, F)
    """

    # ----------------------------
    # Precompute coefficients for right BC (phi)
    # ----------------------------
    a = (1.0 - beta / (Z * T)) / (1.0 + beta / (Z * T))
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / (Z * T))
    c1 = jnp.sqrt(alpha) / (2.0 * Z * T)
    c2 = c1 * b

    # ----------------------------
    # Left-going wave
    # ----------------------------
    w_plus = p0_fun(x - c * t)

    # ----------------------------
    # Right BC (x=L) phi dynamics
    # ----------------------------
    Nt = int(jnp.ceil(t / dt))
    t_grid = jnp.linspace(0.0, t, Nt)
    wp_L = p0_fun(L - c * t_grid)

    phi = 0.0
    w_minus_L = []
    for wp in wp_L:
        wm = a * wp + b * phi
        w_minus_L.append(wm)
        phi = phi + dt * (-c1 * wp - c2 * phi)
    w_minus_L = jnp.array(w_minus_L)

    # Reflected wave propagation from right BC
    t_ref_right = t - (L - x) / c
    w_minus_ref = jnp.where(
        t_ref_right > 0.0,
        jnp.interp(t_ref_right, t_grid, w_minus_L, left=0.0, right=0.0),
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
        v_plus = zeta * l_func(y) * F_func(gamma - w_plus) + epsilon * kappa / omega_r * dy
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
    # Total right-going wave including left BC contribution
    # ----------------------------
    w_minus = w_minus_init + w_minus_ref
    w_plus_total = w_plus + w_plus_left

    # ----------------------------
    # Reconstruction
    # ----------------------------
    p = 0.5 * (w_plus_total + w_minus)
    v = 0.5 * (w_plus_total - w_minus)

    return p, v
