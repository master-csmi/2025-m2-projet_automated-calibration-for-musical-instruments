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
    pL = (c * S_star / S_L) * u_tilde_cells[0, 0, 1]   # tilde_p → p physique

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
        zeta, gamma, eps, kappa, omega_r, y_new, z_new,
        S_quad
    )
    u_tilde_new = u_tilde_cells + dt * k_u

    return u_tilde_new, phi_new, y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     EULER TIME INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------

def time_integrate_euler(
    u_tilde_0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0,
    y0, z0,
    data,
    S_cells, S_star,S_quad,
    snapshot_steps,
    gamma_target = None
):  
    beta, Z, alpha, eps, kappa, gamma, omega_r, Q_r, zeta = data.beta, data.Zt, data.alpha, data.eps, data.kappa, data.gamma_final, 2*jnp.pi*data.fr, data.Qr, data.zeta
    section = data.section
    opening = data.l
    snapshot_steps = jnp.array(snapshot_steps)
    nsnaps = snapshot_steps.shape[0]
    snap_idx0 = 0
    y_snaps = jnp.zeros((nsnaps,))
    z_snaps = jnp.zeros((nsnaps,))
    phi_snaps = jnp.zeros((nsnaps,))
    u_tilde_snaps = jnp.zeros((nsnaps,) + u_tilde_0.shape)
    S_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])

    # Si gamma_t non fourni, gamma constant
    if gamma_target is None:
        gamma_target = jnp.ones(nsteps) * gamma

    def step(carry, inputs):
        u, phi, y, z, snap_idx, y_snaps,z_snaps, phi_snaps, u_tilde_snaps = carry
        n, gamma_n = inputs

        u_next, phi_next, y_next, z_next = euler_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z, 
            gamma_n, 
            eps, kappa, omega_r, zeta, Q_r,opening,
            S_cells, S_star, S_ext,
            S_quad
        )
        # stockage avec .at[]
        safe_idx = jnp.minimum(snap_idx, nsnaps - 1)
        target_step = snapshot_steps[safe_idx]

        is_snap = (snap_idx < nsnaps) & (n == target_step)

        y_snaps = y_snaps.at[safe_idx].set(jnp.where(is_snap, y_next, y_snaps[safe_idx]))
        z_snaps = z_snaps.at[safe_idx].set(jnp.where(is_snap, z_next, z_snaps[safe_idx]))
        phi_snaps = phi_snaps.at[safe_idx].set(jnp.where(is_snap, phi_next, phi_snaps[safe_idx]))
        u_tilde_snaps = u_tilde_snaps.at[safe_idx].set(jnp.where(is_snap, u_next, u_tilde_snaps[safe_idx]))

        snap_idx = snap_idx + is_snap.astype(int)

        return (u_next, phi_next, y_next, z_next,
        snap_idx, y_snaps, z_snaps, phi_snaps, u_tilde_snaps), None
    
    (u_tilde_final, phi_final, y_final, z_final, snap_idx_final, y_snaps, z_snaps, phi_snaps, u_tilde_snaps), _ = lax.scan(
        step, (u_tilde_0, phi0, y0, z0, snap_idx0, y_snaps, z_snaps, phi_snaps, u_tilde_snaps), (jnp.arange(nsteps), gamma_target)
    )
    return u_tilde_final, phi_final, y_final, z_final,u_tilde_snaps, phi_snaps, y_snaps, z_snaps, 