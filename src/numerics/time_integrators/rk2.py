import jax
import jax.numpy as jnp
from jax import lax

from src.numerics.dg.rhs import dg_rhs_system
from src.utils.util_func import phi_rhs, reed_rhs, compute_v_bc_left

from jax import checkpoint


# ------------------------------------------------------------------------------------------------------------------------------
#                                                ENSURE VALUES OF y IN [0,1] 
#                                               (smoothly for automated differentiation)
# ------------------------------------------------------------------------------------------------------------------------------

def smooth_clip(y, low=0.0, high=1.0, k=1000.0):
    y = low + (y - low) * jax.nn.sigmoid(k * (y - low))
    y = high - (high - y) * jax.nn.sigmoid(k * (high - y))
    return y


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     ODE STEPS (2nd Order)     
# ------------------------------------------------------------------------------------------------------------------------------

def reed_step_crank_nicolson(y, z, pL, eps, gamma, omega_r, Q_r, dt):
    """
    Schéma de Crank-Nicolson pour l'ODE de l'anche.
    Ordre 2, inconditionnellement stable.
    Compatible avec RK2 pour le schéma global.
    """
    A = 1.0 + 0.25 * dt**2 * omega_r**2 + 0.5 * dt * omega_r / Q_r
    B = 1.0 - 0.25 * dt**2 * omega_r**2 - 0.5 * dt * omega_r / Q_r
    R = dt * omega_r**2 * (eps * (gamma - pL) + 1.0 - y)


    z_new = (B * z + R) / A
    y_new = y + 0.5 * dt * (z + z_new)
    y_new = smooth_clip(y_new, 0.0, 1.0)  # éviter que l'anche ne devienne négative ou dépasse 1 (physiquement non réaliste)

    return y_new, z_new


# ------------------------------------------------------------------------------------------------------------------------------
#                                                     RK2 STEP (PDE + ODEs)
# ------------------------------------------------------------------------------------------------------------------------------

@jax.jit(static_argnames=("bc",))
def rk2_step_system(
    u_tilde_cells, x_nodes, c, dt,
    Mp_inv, Mv_inv, bc,
    phi, beta, Z, alpha,
    y, z, 
    gamma, 
    eps, kappa, Q_r, omega_r, zeta, opening,
    S_cells, S_star,S_ext,
    S_quad
):
    
    S_L = S_cells[0]
    pL  = (c * S_star / S_L)  * u_tilde_cells[0, 0, 1]
    S_R = S_cells[-1]
    pR  = (c * S_star / S_R)  * u_tilde_cells[-1, 0, 1]

    # -------------------
    # stage 1 — tout au temps n
    # -------------------
    k1_phi = phi_rhs(pR, alpha, Z)
    
    v_bc1= compute_v_bc_left(y,z, pL,zeta,gamma,eps,kappa,omega_r, opening)
    v_bc1_tilde = (S_star / (c * S_cells[0])) * v_bc1
    
    k1_u = dg_rhs_system(
        u_tilde_cells, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi, beta, Z, alpha,
        v_bc1_tilde, S_cells, S_star,S_ext,
        zeta, gamma, eps, kappa, omega_r, y, z,S_quad
    )

    # -------------------
    # midpoint — temps n + dt/2
    # -------------------
    u_tilde_mid   = u_tilde_cells + 0.5 * dt * k1_u
    phi_mid = phi     + 0.5 * dt * k1_phi

    # variable physique au midpoint pour les BC
    pL_mid = (c * S_star / S_cells[0])  * u_tilde_mid[0, 0, 1]
    pR_mid = (c * S_star / S_cells[-1]) * u_tilde_mid[-1, 0, 1]

    # -------------------
    # anche — CN sur dt complet, indépendant du midpoint PDE
    # -------------------

    y_new, z_new = reed_step_crank_nicolson(
        y, z, pL_mid, eps, gamma, omega_r, Q_r, dt
    )

    # -------------------
    # stage 2 — y^{n+1} avec pL_mid
    # -------------------
    k2_phi = phi_rhs(pR_mid, alpha, Z)
    v_bc2 = compute_v_bc_left(y_new,z_new, pL_mid,zeta,gamma,eps,kappa,omega_r, opening)
    v_bc2_tilde = (S_star / (c * S_cells[0])) * v_bc2
    k2_u = dg_rhs_system(
        u_tilde_mid, x_nodes, c,
        Mp_inv, Mv_inv, bc,
        phi_mid, beta, Z, alpha,
        v_bc2_tilde, S_cells, S_star,S_ext,
        zeta, gamma, eps, kappa, omega_r, y_new, z_new,S_quad
    )

    # -----
    # --------------
    # update final
    # -------------------
    u_tilde_new   = u_tilde_cells + dt * k2_u
    phi_new = phi     + dt * k2_phi

    #matrice de passage pour récupérer les variables tilde
    P = jnp.array([[S_cells[0]/(c*S_star), 0.0],
                   [0.0, S_star / (c*S_cells[0])]])

    return u_tilde_new, phi_new, y_new, z_new

# ------------------------------------------------------------------------------------------------------------------------------
#                                                     RK2 TIME INTEGRATION
# ------------------------------------------------------------------------------------------------------------------------------

#def time_integrate_rk2(
#    u_tilde_0, x_nodes, c, dt, nsteps,
#    Mp_inv, Mv_inv, bc,
#    phi0, 
#    y0, z0,
#    data,
#    S_cells, S_star,S_quad,
#    snapshot_steps,
#    gamma_target = None,
#):
#    beta, Z, alpha, eps, kappa, gamma, omega_r, Q_r, zeta = data.beta, data.Zt, data.alpha, data.eps, data.kappa, data.gamma_final, 2*jnp.pi*data.fr, data.Qr, data.zeta
#    section = data.section
#    print("Time integration parameters:")
#    print(f"beta={beta}, Z={Z}, alpha={alpha}, eps={eps}, kappa={kappa}, gamma={gamma}, omega_r={omega_r}, Q_r={Q_r}, zeta={zeta}")
#    # Si gamma_t non fourni, gamma constant
#    if gamma_target is None:
#        gamma_target = jnp.ones(nsteps) * gamma
#
#    opening = data.l
#    snapshot_steps = jnp.array(snapshot_steps)
#    nsnaps = snapshot_steps.shape[0]  # ou len(snapshot_steps) selon ton besoin
#
#    snap_idx0 = 0
#    y_snaps = jnp.zeros((nsnaps,))
#    z_snaps = jnp.zeros((nsnaps,))
#    phi_snaps = jnp.zeros((nsnaps,))
#    u_tilde_snaps = jnp.zeros((nsnaps,) + u_tilde_0.shape)
#
#    def step(carry, inputs):
#        u, phi, y, z, snap_idx, y_snaps, z_snaps, phi_snaps, u_tilde_snaps = carry    
#        n, gamma_n = inputs
#
#        u_next, phi_next, y_next, z_next = rk2_step_system(
#            u, x_nodes, c, dt,
#            Mp_inv, Mv_inv, bc,
#            phi, beta, Z, alpha,
#            y, z,
#            gamma_n,
#            eps, kappa, Q_r, omega_r, zeta, opening,
#            S_cells, S_star,
#            S_quad
#        )
#
#        # stockage avec .at[]
#        safe_idx = jnp.minimum(snap_idx, nsnaps - 1)
#        target_step = snapshot_steps[safe_idx]
#
#        is_snap = (snap_idx < nsnaps) & (n == target_step)
#
#        y_snaps = y_snaps.at[safe_idx].set(jnp.where(is_snap, y_next, y_snaps[safe_idx]))
#        z_snaps = z_snaps.at[safe_idx].set(jnp.where(is_snap, z_next, z_snaps[safe_idx]))
#        phi_snaps = phi_snaps.at[safe_idx].set(jnp.where(is_snap, phi_next, phi_snaps[safe_idx]))
#        u_tilde_snaps = u_tilde_snaps.at[safe_idx].set(jnp.where(is_snap, u_next, u_tilde_snaps[safe_idx]))
#
#        snap_idx = snap_idx + is_snap.astype(int)
#
#        return (u_next, phi_next, y_next, z_next,
#        snap_idx, y_snaps, z_snaps, phi_snaps, u_tilde_snaps), None
#
#    #print(f"dt = {dt:.6f}")
#    #print(f"c  = {c}")
#    #print(f"S_star = {S_star:.6f}")
#    #print(f"S_cells min/max = {S_cells.min():.6f} {S_cells.max():.6f}")
#    #print(f"gamma_t min/max = {gamma_target.min():.4f} {gamma_target.max():.4f}")
#    #print(f"y0 = {y0}, z0 = {z0}")
#    
#    (u_tilde_final, phi_final, y_final, z_final,snap_id_final, y_snaps, z_snaps, phi_snaps, u_tilde_snaps), _ = lax.scan(
#         checkpoint(step),
#        (u_tilde_0, phi0, y0, z0, snap_idx0, y_snaps, z_snaps, phi_snaps, u_tilde_snaps),
#        (jnp.arange(nsteps), gamma_target)
#    )
#    print("blablabla",y_snaps.shape)
#    return u_tilde_final, phi_final, y_final, z_final, snap_id_final, y_snaps, z_snaps, phi_snaps, u_tilde_snaps


def time_integrate_rk2(
    u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, y0, z0,
    data,
    S_cells, S_star, S_quad,
    snapshot_steps,
    gamma_target
):
    beta, Z, alpha = data.beta, data.Zt, data.alpha
    eps, kappa = data.eps, data.kappa
    omega_r = 2 * jnp.pi * data.fr
    Q_r = data.Qr
    zeta = data.zeta
    opening = data.l
    S_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])

    nsnaps = snapshot_steps.shape[0]

    u_snaps   = jnp.zeros((nsnaps,) + u0.shape)
    phi_snaps = jnp.zeros((nsnaps,))
    y_snaps   = jnp.zeros((nsnaps,))
    z_snaps   = jnp.zeros((nsnaps,))

    def step(carry, inputs):
        u, phi, y, z, snap_idx, u_snaps, phi_snaps, y_snaps, z_snaps = carry
        n, gamma_n = inputs

        u_next, phi_next, y_next, z_next = rk2_step_system(
            u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z,
            gamma_n,
            eps, kappa, Q_r, omega_r,
            zeta, opening,
            S_cells, S_star, S_ext,
            S_quad
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
            snap_idx, u_snaps, phi_snaps, y_snaps, z_snaps
        ), None

    init = (
        u0, phi0, y0, z0,
        jnp.array(0, dtype=jnp.int32),
        u_snaps, phi_snaps, y_snaps, z_snaps
    )

    final, _ = lax.scan(
        step,
        init,
        (jnp.arange(nsteps), gamma_target)
    )

    u_final, phi_final, y_final, z_final, _, u_snaps, phi_snaps, y_snaps, z_snaps = final

    return (
        u_final, phi_final, y_final, z_final,
        u_snaps, phi_snaps, y_snaps, z_snaps
    )


def time_integrate_rk2_bell(u0, x_nodes, c, dt, nsteps,
    Mp_inv, Mv_inv, bc,
    phi0, y0, z0,
    data,
    S_cells, S_star, S_quad,
    snapshot_steps,
    gamma_target):

    beta, Z, alpha = data.beta, data.Zt, data.alpha
    eps, kappa = data.eps, data.kappa
    omega_r = 2 * jnp.pi * data.fr
    Q_r = data.Qr
    zeta = data.zeta
    opening = data.l

    nsnaps = snapshot_steps.shape[0]

    S_ext = jnp.concatenate([S_cells[:1], S_cells, S_cells[-1:]])
    p_bell_snaps = jnp.zeros((nsnaps,))

    def step(carry, inputs):
        u, phi, y, z, snap_idx, p_bell_snaps = carry
        n, gamma_n = inputs

        u_next, phi_next, y_next, z_next = rk2_step_system(u, x_nodes, c, dt,
            Mp_inv, Mv_inv, bc,
            phi, beta, Z, alpha,
            y, z,
            gamma_n,
            eps, kappa, Q_r, omega_r,
            zeta, opening,
            S_cells, S_star, S_ext,
            S_quad
            )

        safe_idx = jnp.minimum(snap_idx, nsnaps - 1)
        is_snap = (snap_idx < nsnaps) & (n == snapshot_steps[safe_idx])

        p_bell_next = (c * S_star / S_cells[-1]) * u_next[-1, 0, 1]

        p_bell_snaps = p_bell_snaps.at[safe_idx].set(
            jnp.where(is_snap, p_bell_next, p_bell_snaps[safe_idx])
        )

        snap_idx = snap_idx + is_snap.astype(jnp.int32)

        return (u_next, phi_next, y_next, z_next, snap_idx, p_bell_snaps), None
    
    init = (
        u0, phi0, y0, z0,
        jnp.array(0, dtype=jnp.int32),
        p_bell_snaps
    )

    final, _ = lax.scan(step, init, (jnp.arange(nsteps), gamma_target))

    u_final, phi_final, y_final, z_final, _, p_bell_snaps = final

    return u_final, phi_final, y_final, z_final, p_bell_snaps