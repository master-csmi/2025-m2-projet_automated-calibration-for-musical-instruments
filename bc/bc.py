import jax.numpy as jnp
from dataclasses import dataclass

# ------------------------------------------------------------------
# Right Hand Side of the ODE for phi at right BC
# ------------------------------------------------------------------
def phi_rhs(pR, alpha, Z, T):
    return -jnp.sqrt(alpha) / (Z * T) * pR

# ------------------------------------------------------------------
# Boundary condition dataclass
# ------------------------------------------------------------------
@dataclass(frozen=True)
class BC:
    type: str
    left: tuple
    right: tuple

# ------------------------------------------------------------------
# Right BC with impedance (stable for S(x) variable)
# ------------------------------------------------------------------
def apply_bc_right_impedance(u_cells, phi, beta, Z, T, alpha, S_star=1.0, c=1.0, S_cells=None):
    """
    Right boundary impedance BC taking into account variable section S(x).
    u_cells: (N, 2, 2)
    S_cells: array of cell sections (length N)
    """
    # section du dernier élément
    if S_cells is None:
        S_R = S_star
    else:
        S_R = S_cells[-1]

    # valeurs à l'intérieur du dernier élément
    pR = u_cells[-1, 0, 1]
    vR = u_cells[-1, 1, 1]

    # calcul de la vitesse locale "caractéristique"
    aR = c * jnp.sqrt(S_R / S_star)

    # outgoing wave
    w_plus = pR + vR

    # reflection coefficient ajusté pour S_R
    r = (1.0 - beta / (Z * T)) / (1.0 + beta / (Z * T))

    # incoming wave depuis phi (ODE)
    w_minus = r * w_plus + (2.0 * jnp.sqrt(alpha) / (1.0 + beta / (Z * T))) * phi

    # reconstruire p et v en tenant compte de la section locale
    p_ext = 0.5 * (w_plus + w_minus)
    v_ext = 0.5 * (w_plus - w_minus)

    # créer le ghost cell (pour P1 on répète la valeur aux deux points de Gauss)
    ghost_R = jnp.stack([
        jnp.array([p_ext, p_ext]),
        jnp.array([v_ext, v_ext])
    ])
    return ghost_R


# ------------------------------------------------------------------
# Apply BCs (left Dirichlet + right impedance)
# ------------------------------------------------------------------
def apply_bc(u_cells, bc_left, phi, beta, Z, T, alpha, S_cells=None, S_star=1.0, c=1.0):
    ghost_L = jnp.stack([
        jnp.array([bc_left[0], bc_left[0]]),
        jnp.array([bc_left[1], bc_left[1]])
    ])
    ghost_R = apply_bc_right_impedance(u_cells, phi, beta, Z, T, alpha,S_star=S_star, c=c, S_cells=S_cells)
    return jnp.concatenate([ghost_L[None, ...], u_cells, ghost_R[None, ...]], axis=0)


# ------------------------------------------------------------------
# Neumann BCs: du/dx = 0  => ghost cell = adjacent cell
# ------------------------------------------------------------------
def apply_bc_neumann(u_cells):
    ghost_L = u_cells[0]
    ghost_R = u_cells[-1]
    return jnp.concatenate(
        [ghost_L[None, ...], u_cells, ghost_R[None, ...]],
        axis=0
    )
