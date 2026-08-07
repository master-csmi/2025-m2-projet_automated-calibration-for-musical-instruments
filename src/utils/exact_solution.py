import jax
import jax.numpy as jnp


def interp_linear(x_query, x_grid, y_grid):
    idx = jnp.searchsorted(x_grid, x_query, side="right") - 1
    idx = jnp.clip(idx, 0, x_grid.shape[0] - 2)

    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_grid[idx]
    y1 = y_grid[idx + 1]

    theta = (x_query - x0) / (x1 - x0 + 1e-30)

    y = y0 + theta * (y1 - y0)

    return jnp.where(
        (x_query < x_grid[0]) | (x_query > x_grid[-1]),
        0.0,
        y,
    )


def exact_solution_characteristics(
    x,
    t,
    p0_fun,
    c,
    L,
    alpha,
    beta,
    Z,
    dt=1e-4,
    method="rk2",
):
    x = jnp.asarray(x)

    a = (1.0 - beta / Z) / (1.0 + beta / Z)
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / Z)

    c1 = jnp.sqrt(alpha) / (2.0 * Z)
    c2 = c1 * b

    Nt = max(1, int(jnp.ceil(t / dt)))
    dt_eff = t / Nt

    t_grid = jnp.linspace(0.0, t, Nt + 1)

    # Onde incidente au bord droit x=L.
    # Elle provient de x = L - c tau.
    w_plus_L = jax.vmap(lambda tau: p0_fun(L - c * tau))(t_grid)

    phi_values = []
    wm_values = []

    phi = jnp.array(0.0, dtype=x.dtype)

    for wp in w_plus_L:
        def rhs(phi_val):
            return -c1 * (1.0 + a) * wp - c2 * phi_val

        if method == "euler":
            phi_new = phi + dt_eff * rhs(phi)
        elif method == "rk2":
            k1 = rhs(phi)
            phi_new = phi + dt_eff * rhs(phi + 0.5 * dt_eff * k1)
        else:
            raise ValueError(f"Unknown method: {method}")

        phi = phi_new

        wm = a * wp + b * phi
        phi_values.append(phi)
        wm_values.append(wm)

    w_minus_L = jnp.asarray(wm_values)

    # Temps auquel la caractéristique issue de x=L arrive au point x.
    tau_ref = t - (L - x) / c

    w_minus_ref = jax.vmap(
        lambda tau: jnp.where(
            tau >= 0.0,
            interp_linear(tau, t_grid, w_minus_L),
            0.0,
        )
    )(tau_ref)

    # Partie venant de la condition initiale.
    w_plus_init = jax.vmap(lambda xx: p0_fun(xx))(x - c * t)
    w_minus_init = jax.vmap(lambda xx: p0_fun(xx))(x + c * t)

    w_plus = w_plus_init
    w_minus = w_minus_init + w_minus_ref

    p = 0.5 * (w_plus + w_minus)
    v = 0.5 * (w_plus - w_minus)

    return p, v