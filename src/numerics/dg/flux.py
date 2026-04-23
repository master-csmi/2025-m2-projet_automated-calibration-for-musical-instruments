import jax.numpy as jnp

def upwind_flux(U_L, U_R, S_interface, c, S_star):
    # facteur section
    factor = (S_interface / (c * S_star))**2

    # --- invariants gauche ---
    pL, vL = U_L
    w_plus_L  = vL + factor * pL
    w_minus_L = vL - factor * pL

    # --- invariants droite ---
    pR, vR = U_R
    w_plus_R  = vR + factor * pR
    w_minus_R = vR - factor * pR

    # --- upwind ---
    w_plus_star  = w_plus_L     # vitesse +c
    w_minus_star = w_minus_R    # vitesse -c

    # --- reconstruction ---
    v_star = 0.5 * (w_plus_star + w_minus_star)
    p_star = (w_plus_star - w_minus_star) / (2.0 * factor)

    U_star = jnp.array([p_star, v_star])

    # --- matrice A ---
    A = jnp.array([
        [0.0, c * S_interface / S_star],
        [c * S_star / S_interface, 0.0]
    ])

    return A @ U_star

def rusanov_flux(U_tilde_L, U_tilde_R, S_interface, c, S_star):

    # d_t(u_tilde)+d_x(A u_tilde) = 0

    A = jnp.array([[0.0, c * S_interface / S_star],
                   [c * S_star / S_interface, 0.0]])
    
    smax = c  # vitesse maximale du système
    return 0.5 * (A @ U_tilde_L + A @ U_tilde_R) - 0.5 * smax * (U_tilde_R - U_tilde_L)