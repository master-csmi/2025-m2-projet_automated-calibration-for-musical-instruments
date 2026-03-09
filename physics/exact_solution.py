import jax.numpy as jnp
from utils.util_func import l, pressure_func

# Fonction non linéaire de l'embouchure
def F(delta_p):
    return jnp.sign(delta_p) * jnp.sqrt(jnp.abs(delta_p))



# -----------------------------------------------------------------------------------
# Exact solution using characteristics with left (embouchure) and right (pavillon) BC
# -----------------------------------------------------------------------------------
def exact_solution_characteristics(
    x, t, p0_fun, c, L,
    # Pavillon (x=L) parameters
    alpha, beta, Z,
    # Embouchure (x=0) parameters
    y,y_dot,
    gamma, omega_r, Q_r, zeta, kappa, epsilon,
    dt=1e-4,
    method="rk2",
):
    # ----------------------------
    # Pavillon coefficients
    # ----------------------------
    a = (1.0 - beta / Z) / (1.0 + beta / Z)
    b = 2.0 * jnp.sqrt(alpha) / (1.0 + beta / Z)
    c1 = jnp.sqrt(alpha) / (2.0 * Z)
    c2 = c1 * b

    # ----------------------------
    # Initial left- and right-going waves
    # ----------------------------
    w_plus = p0_fun(x - c * t)   # left-going initial
    w_minus_init = p0_fun(x + c * t)  # right-going initial

    # ----------------------------
    # Time grid
    # ----------------------------
    Nt = int(jnp.ceil(t / dt))
    t_grid = jnp.linspace(0.0, t, Nt)

    # ----------------------------
    # Pavillon (x=L) dynamics
    # ----------------------------
    w_plus_L = p0_fun(L - c * t_grid)
    phi = 0.0
    w_minus_L = []

    for wp in w_plus_L:
        def rhs(phi_val):
            return -c1 * (1.0 + a) * wp - c2 * phi_val

        if method == "euler":
            phi = phi + dt * rhs(phi)
        elif method == "rk2":
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
    # Embouchure (x=0) dynamics
    # ----------------------------
    w_plus_0 = []

    # On utilise la même Nt pour avancer y et calculer v(0)
    for n in range(Nt):
        # pression à l'embouchure (approx initiale)
        # ici on utilise w_minus_init pour simplifier
        p0 = w_minus_init[0] if n == 0 else p0_fun(0.0 + c * t_grid[n])

        # EDO de y (embouchure)
        y_ddot = omega_r**2 * (epsilon * (gamma - p0) - y + 1 - (1.0 / (Q_r * omega_r)) * y_dot)

        if method == "euler":
            y_dot = y_dot + dt * y_ddot
            y = y + dt * y_dot
        elif method == "rk2":
            k1 = y_ddot
            y_dot_star = y_dot + dt * k1
            y_star = y + dt * y_dot
            k2 = omega_r**2 * (epsilon * (gamma - p0) - y_star + 1 - (1.0 / (Q_r * omega_r)) * y_dot_star)
            y_dot = y_dot + 0.5 * dt * (k1 + k2)
            y = y + 0.5 * dt * (y_dot + y_dot_star)

        # débit à l'embouchure
        v0 = zeta * l(y) * F(gamma - p0) + epsilon * kappa / omega_r * y_dot
        w_plus_0.append(p0 + v0)

    w_plus_0 = jnp.array(w_plus_0)

    # ----------------------------
    # Propagation des ondes
    # ----------------------------
    # temps retardé pour propagation depuis le pavillon
    t_ref = t - (L - x) / c
    w_minus_ref = jnp.where(
        t_ref > 0.0,
        jnp.interp(t_ref, t_grid, w_minus_L, left=0.0, right=0.0),
        0.0
    )

    # Contribution de l'embouchure
    t_left = t - x / c
    w_plus_ref = jnp.where(
        t_left > 0.0,
        jnp.interp(t_left, t_grid, w_plus_0, left=0.0, right=0.0),
        0.0
    )

    # ----------------------------
    # Total waves
    # ----------------------------
    w_minus = w_minus_init + w_minus_ref
    w_plus_total = w_plus + w_plus_ref

    # ----------------------------
    # Reconstruction pression / débit
    # ----------------------------
    p = 0.5 * (w_plus_total + w_minus)
    v = 0.5 * (w_plus_total - w_minus)

    return p, v