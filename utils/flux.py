import jax.numpy as jnp


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          flux for system (linear)
# ------------------------------------------------------------------------------------------------------------------------------
def physical_flux(U, S_cell, c=1.0, S_star=1.0):
    """
    Physical flux for
    F(U) = (v, p)
    """
    p_t, v_t = U
    a = S_cell / (c * S_star)
    b = c * S_star / S_cell
    return jnp.array([
        a * v_t,
        b * p_t
    ])

def rusanov_hyperbolic(U_L, U_R, S_face, c=1.0, S_star=1.0):
    """
    Flux upwind pour système hyperbolique : ∂t U + A(x) ∂x U = 0
    U = [p,v]
    """
    # Diagonalisation A = R Λ R^{-1}
    a = c * S_star / S_face
    b = c * S_face / S_star

    # characteristic variables
    wL = U_L[0] + jnp.sqrt(a/b)*U_L[1]
    wR = U_R[0] + jnp.sqrt(a/b)*U_R[1]

    # max wave speed
    lam = c  #use max(c*S*/S, c*S/S*) for local CFL if needed (to see)

    # Rusanov
    F_L = jnp.array([b*U_L[1], a*U_L[0]])
    F_R = jnp.array([b*U_R[1], a*U_R[0]])

    jump = U_R - U_L
    diss = lam * jump

    return 0.5*(F_L + F_R) - 0.5*diss