import jax.numpy as jnp

# ------------------------------------------------------------------------------------------------------------------------------
#                                                 analytic solution for initial p0 Gaussian and v0=0
#                                                 decomposition into w+ = p+v, w- = p-v
# ------------------------------------------------------------------------------------------------------------------------------

def exact_solution_characteristics(x, t, p0_fun, c, L, alpha, beta, ZT, dt=1e-4):

    # ----------------------------
    # Precompute coefficients
    # ----------------------------
    a = (1.0 - beta / ZT) / (1.0 + beta / ZT)
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / ZT)
    c1 = jnp.sqrt(alpha) / (2.0 * ZT)
    c2 = c1 * b

    # ----------------------------
    # Left-going wave
    # ----------------------------
    w_plus = p0_fun(x - c * t)

    # ----------------------------
    # Boundary dynamics (x = L)
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

    # ----------------------------
    # Reflected wave propagation
    # ----------------------------
    t_ref = t - (L - x) / c

    w_minus_ref = jnp.where( #linear interpolation
        t_ref > 0.0,
        jnp.interp(t_ref, t_grid, w_minus_L, left=0.0, right=0.0),
        0.0
    )

    # ----------------------------
    # Initial right-going wave
    # ----------------------------
    w_minus_init = p0_fun(x + c * t)

    # ----------------------------
    # Total right-going wave
    # ----------------------------
    w_minus = w_minus_init + w_minus_ref

    # ----------------------------
    # Reconstruction
    # ----------------------------
    p = 0.5 * (w_plus + w_minus)
    v = 0.5 * (w_plus - w_minus)

    return p, v