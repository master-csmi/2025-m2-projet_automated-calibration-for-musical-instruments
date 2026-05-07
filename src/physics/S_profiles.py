import jax.numpy as jnp
import equinox as eqx

# ------------------------------------------------------------------------------------------------------------------------------
#                                                          Fucntion of the instrument section S(x)
# ------------------------------------------------------------------------------------------------------------------------------
def S_of_x(x, type_S="const", **kwargs):
    """
    Cross-sectional area of the instrument at position x.

    Parameters
    ----------
    x : array_like
        Positions along the instrument.
    type_S : str
        Type of the section profile: "const", "exp", "cone", "bump".
    kwargs : dict
        Additional parameters for each type:
        - const: S0
        - exp: S0, k
        - cone: S0, k
        - bump: S0, A, x_c, sigma

    Returns
    -------
    Sx : array_like
        Cross-sectional area at positions x.
    """
    x = jnp.array(x)  # ensure JAX array

    if type_S == "const":
        S0 = kwargs.get("S0", 1.0)
        return jnp.ones_like(x) * S0

    elif type_S == "exp":
        S0 = kwargs.get("S0", 1.0)
        k  = kwargs.get("k", 0.3)
        return S0 * jnp.exp(k * x)

    elif type_S == "cone":
        S0 = kwargs.get("S0", 0.5)
        k  = kwargs.get("k", 1.0)
        return S0 + k * x

    elif type_S == "bump":
        S0    = kwargs.get("S0", 1.0)
        A     = kwargs.get("A", 0.5)
        x_c   = kwargs.get("x_c", 0.5)
        sigma = kwargs.get("sigma", 0.1)
        return S0 * (1 + A * jnp.exp(-(x - x_c)**2 / sigma**2))

    else:
        raise ValueError(f"Unknown type '{type_S}' for S(x)")
    
class SProfile(eqx.Module):
    type_S: str = eqx.field(static=True)
    L_tube: float
    R_tube: float
    L_bell: float
    k_bell: float

    def __init__(self, type_S, L_tube, R_tube, L_bell, k_bell):
        self.type_S = type_S
        self.L_tube = jnp.array(L_tube, dtype=jnp.float64)
        self.R_tube = jnp.array(R_tube, dtype=jnp.float64)
        self.L_bell = jnp.array(L_bell, dtype=jnp.float64)
        self.k_bell = jnp.array(k_bell, dtype=jnp.float64)
    
    
    def __call__(self, x):
        x = jnp.array(x)
        S0_tube = jnp.pi * self.R_tube**2 # cross-sectional area of the tube
        if self.type_S == "const":
            return jnp.ones_like(x) * S0_tube # only a tube with constant cross-sectional area

        elif self.type_S == "exp":
            S_tube = S0_tube * jnp.ones_like(x) 
            S_bell = S0_tube * jnp.exp(self.k_bell * (x - self.L_tube))
            return jnp.where(x < self.L_tube, S_tube, S_bell) # tube with constant cross-sectional area and bell with exponentially increasing cross-sectional area

        elif self.type_S == "cone":
            S_tube = S0_tube * jnp.ones_like(x)
            S_bell = S0_tube + self.k_bell * (x - self.L_tube)
            return jnp.where(x < self.L_tube, S_tube, S_bell) # tube with constant cross-sectional area and bell with linearly increasing cross-sectional area

        else:
            raise ValueError(f"Unknown type '{self.type_S}' for S(x)")