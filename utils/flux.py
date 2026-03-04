import jax.numpy as jnp

# ------------------------------------------------------------------------------------------------------------------------------
# Flux conservative for variable section system
# ------------------------------------------------------------------------------------------------------------------------------
def flux_conservative(U, S_cell, S_star=1.0, c=1.0):
    """
    Conservative flux for cell with S(x)
    U[0] = S_cell/(c*S_star) * p
    U[1] = c*S_star / S_cell * v
    """
    F = jnp.array([
        S_cell/(c * S_star) * U[1],  # flux for S*p
        c *S_star/S_cell * U[0]      # flux for S*v
    ])
    return F

def rusanov_flux(U_L, U_R, S_L, S_R, S_star=1.0, c=1.0):

    F_L = flux_conservative(U_L, S_L, S_star, c)
    F_R = flux_conservative(U_R, S_R, S_star, c)

    smax = c

    return 0.5*(F_L + F_R) - 0.5*smax*(U_R - U_L)