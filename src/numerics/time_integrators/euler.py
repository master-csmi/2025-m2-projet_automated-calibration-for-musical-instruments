import jax
import jax.numpy as jnp
from jax import lax

from src.numerics.dg.rhs import dg_rhs_system
from src.utils.util_func import phi_rhs, reed_rhs, compute_v_bc_left

def smooth_clip(y, low=0.0, high=1.0, k=1500.0):
    y = low + (y - low) * jax.nn.sigmoid(k * (y - low))
    y = high - (high - y) * jax.nn.sigmoid(k * (high - y))
    return y

# ------------------------------------------------------------------------------------------------------------------------------
#                                                     ODE STEPS (1st Order)
# ------------------------------------------------------------------------------------------------------------------------------

def reed_step_implicit_euler(y, z, pL, eps, gamma, omega_r, Q_r, dt):
    """
    Schéma d'Euler implicite pour l'ODE de l'anche.
    Ordre 1, inconditionnellement stable.
    Compatible avec Euler explicite pour le schéma global.
    """
    D = 1.0 + dt * omega_r / Q_r + dt**2 * omega_r**2
    N = (y * (1.0 + dt * omega_r / Q_r)
         + dt * z
         + dt**2 * omega_r**2 * (eps * (gamma - pL) + 1.0))

    y_new = N / D
    y_new = smooth_clip(y_new, low=0.0, high=1.0, k=1500.0)
    z_new = (y_new - y) / dt

    return y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     EULER STEP
# ------------------------------------------------------------------------------------------------------------------------------

@jax.jit(static_argnames=("bc",))
def euler_step_system(
    u_tilde_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, z, gamma, eps, kappa, omega_r, zeta, Q_r,opening,
    S_cells, S_star, S_ext,
    S_quad
):
    S_L = S_cells[0]
    pL = (c * S_star / S_L) * u_tilde_cells[0, 0, 0]   # tilde_p → p physique

    S_R = S_cells[-1]
    pR = (c * S_star / S_R) * u_tilde_cells[-1, 0, 1]  # tilde_p → p physique

    # update phi
    k_phi = phi_rhs(pR, alpha, Z)
    phi_new = phi + dt * k_phi

    # update reed
    y_new, z_new = reed_step_implicit_euler(y, z, pL, eps, gamma, omega_r, Q_r, dt)

    v_bc = compute_v_bc_left(y_new, z_new, pL, zeta, gamma, eps, kappa, omega_r, opening)
    v_bc_tilde = (S_star / (c * S_cells[0])) * v_bc

    # PDE RHS
    k_u = dg_rhs_system(
        u_tilde_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_new, beta, Z, alpha,
        v_bc_tilde, S_cells, S_star, S_ext,
        zeta, gamma, eps, kappa, omega_r, y_new, z_new,opening,
        S_quad
    )
    u_tilde_new = u_tilde_cells + dt * k_u

    return u_tilde_new, phi_new, y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     EULER TIME INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------

def time_integrate_euler(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, y0, z0,
    data,
    S_cells, S_star, S_quad, S_ext,
    snapshot_steps,
    gamma_target=None,
):
    beta, Z, alpha = data.beta, data.Zt, data.alpha
    eps, kappa = data.eps, data.kappa
    omega_r = 2.0 * jnp.pi * data.fr
    Q_r = data.Qr
    zeta = data.zeta
    opening = data.l

    if gamma_target is None:
        gamma_target = jnp.ones((nsteps,)) * data.gamma_final

    snapshot_steps = jnp.asarray(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]

    u_snaps = jnp.zeros((nsnaps,) + u0.shape)
    phi_snaps = jnp.zeros((nsnaps,))
    y_snaps = jnp.zeros((nsnaps,))
    z_snaps = jnp.zeros((nsnaps,))

    def step(carry, inputs):
        u, phi, y, z, snap_idx, u_snaps, phi_snaps, y_snaps, z_snaps = carry
        n, gamma_n = inputs

        u_next, phi_next, y_next, z_next = euler_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z,
            gamma_n,
            eps, kappa, omega_r, zeta, Q_r, opening,
            S_cells, S_star, S_ext,
            S_quad,
        )

        safe_idx = jnp.minimum(snap_idx, nsnaps - 1)
        is_snap = (snap_idx < nsnaps) & (n == snapshot_steps[safe_idx])

        u_snaps = u_snaps.at[safe_idx].set(
            jnp.where(is_snap, u_next, u_snaps[safe_idx])
        )
        phi_snaps = phi_snaps.at[safe_idx].set(
            jnp.where(is_snap, phi_next, phi_snaps[safe_idx])
        )
        y_snaps = y_snaps.at[safe_idx].set(
            jnp.where(is_snap, y_next, y_snaps[safe_idx])
        )
        z_snaps = z_snaps.at[safe_idx].set(
            jnp.where(is_snap, z_next, z_snaps[safe_idx])
        )

        snap_idx = snap_idx + is_snap.astype(jnp.int32)

        return (
            u_next, phi_next, y_next, z_next,
            snap_idx, u_snaps, phi_snaps, y_snaps, z_snaps,
        ), None

    init = (
        u0, phi0, y0, z0,
        jnp.array(0, dtype=jnp.int32),
        u_snaps, phi_snaps, y_snaps, z_snaps,
    )

    final, _ = lax.scan(
        step,
        init,
        (jnp.arange(nsteps), gamma_target),
    )

    u_final, phi_final, y_final, z_final, _, u_snaps, phi_snaps, y_snaps, z_snaps = final

    return (
        u_final, phi_final, y_final, z_final,
        u_snaps, phi_snaps, y_snaps, z_snaps,
    )