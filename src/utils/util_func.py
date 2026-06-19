import jax
import jax.numpy as jnp
import equinox as eqx
from src.numerics.dg.basis import vphi_at

def project_L2_cell(xL, xR, p0, v0, S_fun, c, S_star, Mp_inv, Mv_inv):
    h = xR - xL

    # Quadrature Gauss (P1 exact)
    xi_q = jnp.array([-1/jnp.sqrt(3), 1/jnp.sqrt(3)])
    w_q  = jnp.array([1.0, 1.0])

    # Mapping vers la cellule
    x_q = 0.5*(xL + xR) + 0.5*h*xi_q          # (2,)

    # Fonctions de base évaluées aux points de quadrature
    ph_q = vphi_at(x_q, xL, xR)               # (2, 2)

    # Section
    S_q = jax.vmap(S_fun)(x_q)                # (2,)

    # Champs initiaux
    p_q = jax.vmap(p0)(x_q)                   # (2,)
    v_q = jax.vmap(v0)(x_q)                   # (2,)

    # Passage en variables tilde
    pt_q = (S_q/(c*S_star)) * p_q
    vt_q = (S_star/(c*S_q)) * v_q

    # Poids quadrature
    weights = w_q * h/2                       # (2,)

    # Intégration vectorisée
    b_p = jnp.sum(weights[:, None] * pt_q[:, None] * ph_q, axis=0)
    b_v = jnp.sum(weights[:, None] * vt_q[:, None] * ph_q, axis=0)

    # Résolution locale
    u_p = Mp_inv @ b_p
    u_v = Mv_inv @ b_v

    return jnp.stack([u_p, u_v])

@eqx.filter_jit
def project_L2(
    xLs, xRs,
    p0, v0,
    S_fun,
    c, S_star,
    Mp_inv, Mv_inv
):

    def project_cell(xL, xR, Mp_i, Mv_i):
        return project_L2_cell(
            xL, xR, p0, v0, S_fun, c, S_star, Mp_i, Mv_i
        )

    return jax.vmap(project_cell)(
        xLs, xRs, Mp_inv, Mv_inv
    )



# Reed opening function (trainable)
class ReedOpening(eqx.Module):

    def __call__(self, y):
        return y * (y > 0)

# Right Hand Side of the ODE for phi at right BC
def phi_rhs(pR, alpha, Z):
    return -jnp.sqrt(alpha)/ (Z) * pR

# Right Hand Side of the ODE for reed dynamics at left BC
def reed_rhs(y, z, p_in, eps, gamma, omega_r, Q_r):
    
    dy = z
    
    dz = (
        omega_r**2 * (eps * (gamma - p_in) - y + 1)
        - (omega_r / Q_r) * z
    )
    
    return dy, dz

def pressure_func(delta_p, eps=1e-8):
    return jnp.sqrt(jnp.abs(delta_p) + eps) * jnp.sign(delta_p)


def compute_v_bc_left(y, y_t, p_in, zeta, gamma, eps, kappa, omega_r,l):
    

    return zeta * l(y) * pressure_func(gamma - p_in) + eps * kappa / omega_r *  y_t

def precompute_S_quad(section, xLs, xRs, nq):
    def compute_cell(xL, xR):
        xq = jnp.linspace(xL, xR, nq)
        return jax.vmap(section)(xq)

    return jax.vmap(compute_cell)(xLs, xRs)  # (N, nq)

def compute_Zt(data):
    L = data.section.L_tube + data.section.L_bell

    S_star = data.section(0)
    S_L = data.section(L)

    return S_star / S_L

def make_openwind_radius_profile(data, n_points=200):
    L = float(data.L_tube + data.L_bell)

    x = jnp.linspace(0.0, L, n_points)
    S = data.section(x)
    R = jnp.sqrt(S / jnp.pi)

    return x, R

def best_time_shift(x, y, dt):
    """
    Cherche le décalage temporel qui maximise la corrélation entre x et y.
    x : signal DG
    y : signal OpenWind
    dt : pas de temps entre snapshots
    """
    x0 = x - jnp.mean(x)
    y0 = y - jnp.mean(y)

    corr_full = jnp.correlate(x0, y0, mode="full")
    lag_index = jnp.argmax(corr_full) - (len(y0) - 1)

    shift_time = lag_index * dt

    corr_max = jnp.max(corr_full) / (
        jnp.linalg.norm(x0) * jnp.linalg.norm(y0) + 1e-12
    )

    return int(lag_index), float(shift_time), float(corr_max)
