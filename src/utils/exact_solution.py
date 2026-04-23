import jax
import jax.numpy as jnp

def interp_linear(x_query, x_grid, y_grid):
    """Interpolation linéaire scalaire, compatible vmap."""
    idx = jnp.searchsorted(x_grid, x_query, side='right') - 1
    idx = jnp.clip(idx, 0, x_grid.shape[0] - 2)
    x0  = x_grid[idx]
    x1  = x_grid[idx + 1]
    y0  = y_grid[idx]
    y1  = y_grid[idx + 1]
    t   = (x_query - x0) / (x1 - x0 + 1e-30)
    return jnp.where(
        x_query < x_grid[0],  0.0,
        jnp.where(
            x_query > x_grid[-1], 0.0,
            y0 + t * (y1 - y0)
        )
    )

def exact_solution_characteristics(
    x, t, p0_fun, c, L,
    alpha, beta, Z,
    dt=1e-4,
    method="rk2",
):
    a  = (1.0 - beta / Z) / (1.0 + beta / Z)
    b  = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / Z)
    c1 = jnp.sqrt(alpha) / (2.0 * Z)
    c2 = c1 * b

    # Forcer Nt suffisamment grand
    Nt     = max(100, int(jnp.ceil(t / dt)))
    dt_eff = t / Nt                                    # dt effectif
    t_grid = jnp.linspace(0.0, float(t), Nt + 1)      # (Nt+1,)
    w_plus_L = jnp.atleast_1d(p0_fun(L - c * t_grid))   # garanti (Nt+1,)                # (Nt+1,)

    # Boucle Python — pas de problème de shape
    phi = 0.0
    w_minus_L = []

    for wp in w_plus_L:
        def rhs(phi_val):
            return -c1 * (1.0 + a) * float(wp) - c2 * phi_val
        if method == "euler":
            phi = phi + dt_eff * rhs(phi)
        else:
            k1  = rhs(phi)
            phi = phi + dt_eff * rhs(phi + 0.5 * dt_eff * k1)
        w_minus_L.append(float(a * wp + b * phi))

    w_minus_L = jnp.array(w_minus_L)   # (Nt+1,) garanti

    # Interpolation
    t_ref = (t - (L - x) / c).reshape(-1)

    w_minus_ref = jax.vmap(
        lambda tr: jnp.where(
            tr > 0.0,
            interp_linear(tr, t_grid, w_minus_L),
            0.0
        )
    )(t_ref)

    w_plus       = p0_fun(x - c * t)
    w_minus_init = p0_fun(x + c * t)
    w_minus      = w_minus_init + w_minus_ref

    p = 0.5 * (w_plus + w_minus)
    v = 0.5 * (w_plus - w_minus)

    return p, v