import jax.numpy as jnp


# ------------------------------------------------------------------------------------------------------------------------------
#                                                          flux for system (linear)
# ------------------------------------------------------------------------------------------------------------------------------
def linear_system_flux(A):
    def Flux(U):
        # U shape (...,2)
        return U @ A.T 
    return Flux

def rusanov_flux(U_L, U_R, c=1.0):
    """
    Correct Rusanov flux for
    F(U) = (v, p)
    """
    smax = c
    F_L = jnp.array([U_L[1], U_L[0]])
    F_R = jnp.array([U_R[1], U_R[0]])

    return 0.5 * (F_L + F_R) - 0.5 * smax * (U_R - U_L)