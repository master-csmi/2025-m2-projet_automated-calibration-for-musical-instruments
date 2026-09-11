"""
Diagnostic d'ordre pour le coeur DG P1 + RK2.

Ce script isole le schema numerique des conditions aux limites physiques :
- acoustique lineaire a coefficients constants ;
- domaine periodique ;
- solution exacte lisse ;
- norme L2 espace-temps final sur p et v.

Execution depuis la racine du depot :

    python src/test_order2_core.py

Si le coeur DG/RK2 est bien d'ordre 2, la colonne "order" doit tendre vers 2.
"""

import argparse
import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np


jax.config.update("jax_enable_x64", True)


def mass_inv(h):
    mass = (h / 6.0) * jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return jnp.linalg.inv(mass)


def flux(u, c):
    p, v = u
    return jnp.array([c * v, c * p])


def rusanov_flux(u_left, u_right, c):
    return 0.5 * (flux(u_left, c) + flux(u_right, c)) - 0.5 * c * (
        u_right - u_left
    )


@jax.jit
def dg_rhs_periodic(u_cells, c, h, minv):
    """
    RHS DG P1 nodal pour U_t + A U_x = 0, avec flux de Rusanov periodique.

    u_cells shape: (Nx, 2 variables, 2 ddl nodaux)
    """
    u_left = u_cells[:, :, 1]
    u_right = jnp.roll(u_cells, -1, axis=0)[:, :, 0]
    interface_fluxes = jax.vmap(rusanov_flux, in_axes=(0, 0, None))(
        u_left, u_right, c
    )

    flux_left = jnp.roll(interface_fluxes, 1, axis=0)
    flux_right = interface_fluxes

    u_mean_int = 0.5 * (u_cells[:, :, 0] + u_cells[:, :, 1])
    volume = jax.vmap(flux, in_axes=(0, None))(u_mean_int, c)

    rhs_dof0 = -volume + flux_left
    rhs_dof1 = volume - flux_right

    rhs = jnp.stack([rhs_dof0, rhs_dof1], axis=-1)
    return jnp.einsum("ab,nvb->nva", minv, rhs)


@jax.jit
def rk2_step(u, c, h, dt, minv):
    k1 = dg_rhs_periodic(u, c, h, minv)
    u_mid = u + 0.5 * dt * k1
    k2 = dg_rhs_periodic(u_mid, c, h, minv)
    return u + dt * k2


def exact_solution(x, t, c):
    wave = jnp.sin(2.0 * jnp.pi * (x - c * t))
    return jnp.stack([wave, wave], axis=0)


def initial_condition(Nx, c, use_l2_projection):
    x_edges = jnp.linspace(0.0, 1.0, Nx + 1)
    x_left = x_edges[:-1]
    x_right = x_edges[1:]
    h = 1.0 / Nx
    minv = mass_inv(h)

    if not use_l2_projection:
        u0_left = exact_solution(x_left, 0.0, c).T
        u0_right = exact_solution(x_right, 0.0, c).T
        return jnp.stack([u0_left, u0_right], axis=-1)

    xi_q = jnp.array([-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)])
    w_q = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])

    def project_cell(xL, xR):
        xq = 0.5 * (xL + xR) + 0.5 * h * xi_q
        phi0 = 0.5 * (1.0 - xi_q)
        phi1 = 0.5 * (1.0 + xi_q)
        phi = jnp.stack([phi0, phi1], axis=1)
        uq = exact_solution(xq, 0.0, c).T
        b = jnp.einsum("q,qv,qa->va", w_q * 0.5 * h, uq, phi)
        return jnp.einsum("ab,vb->va", minv, b)

    return jax.vmap(project_cell)(x_left, x_right)


def integrate(Nx, c, cfl, t_final, use_l2_projection):
    h = 1.0 / Nx
    dt_nominal = cfl * h / c
    nsteps = int(np.ceil(t_final / dt_nominal))
    dt = t_final / nsteps
    minv = mass_inv(h)

    u0 = initial_condition(Nx, c, use_l2_projection)

    def step(u, _):
        return rk2_step(u, c, h, dt, minv), None

    u_final, _ = jax.lax.scan(step, u0, xs=None, length=nsteps)
    return u_final, dt, nsteps


def l2_error(u_cells, c, t_final):
    Nx = u_cells.shape[0]
    h = 1.0 / Nx
    x_edges = jnp.linspace(0.0, 1.0, Nx + 1)
    x_left = x_edges[:-1]
    x_right = x_edges[1:]

    xi_q = jnp.array([-jnp.sqrt(3.0 / 5.0), 0.0, jnp.sqrt(3.0 / 5.0)])
    w_q = jnp.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0])
    phi0 = 0.5 * (1.0 - xi_q)
    phi1 = 0.5 * (1.0 + xi_q)

    def cell_error(u_cell, xL, xR):
        xq = 0.5 * (xL + xR) + 0.5 * h * xi_q
        uh = u_cell[:, 0][None, :] * phi0[:, None] + u_cell[:, 1][None, :] * phi1[
            :, None
        ]
        ue = exact_solution(xq, t_final, c).T
        diff2 = jnp.sum((uh - ue) ** 2, axis=1)
        exact2 = jnp.sum(ue**2, axis=1)
        return (
            jnp.sum(w_q * 0.5 * h * diff2),
            jnp.sum(w_q * 0.5 * h * exact2),
        )

    err2, ref2 = jax.vmap(cell_error)(u_cells, x_left, x_right)
    return float(jnp.sqrt(jnp.sum(err2) / jnp.sum(ref2)))


def run_convergence(nx_values, c, cfl, t_final, use_l2_projection):
    rows = []

    for Nx in nx_values:
        u_final, dt, nsteps = integrate(Nx, c, cfl, t_final, use_l2_projection)
        jax.block_until_ready(u_final)
        err = l2_error(u_final, c, t_final)
        rows.append({"Nx": Nx, "h": 1.0 / Nx, "dt": dt, "nsteps": nsteps, "err": err})

    for i in range(1, len(rows)):
        e_prev = rows[i - 1]["err"]
        e_curr = rows[i]["err"]
        h_prev = rows[i - 1]["h"]
        h_curr = rows[i]["h"]
        rows[i]["order"] = np.log(e_prev / e_curr) / np.log(h_prev / h_curr)
    rows[0]["order"] = None

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", nargs="+", type=int, default=[50, 100, 200, 400, 800])
    parser.add_argument("--cfl", type=float, default=0.2)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument(
        "--interpolate-init",
        action="store_true",
        help="Utilise les valeurs nodales au lieu de la projection L2.",
    )
    args = parser.parse_args()

    rows = run_convergence(
        nx_values=args.nx,
        c=args.c,
        cfl=args.cfl,
        t_final=args.t_final,
        use_l2_projection=not args.interpolate_init,
    )

    init_label = "interpolation nodale" if args.interpolate_init else "projection L2"
    print("\n=== Test coeur DG P1 + RK2 periodique ===")
    print(f"initialisation : {init_label}")
    print(f"c={args.c}, CFL={args.cfl}, T={args.t_final}")
    print("\nNx        dt              nsteps      rel_L2          order")
    print("-" * 66)
    for row in rows:
        order = "" if row["order"] is None else f"{row['order']:.4f}"
        print(
            f"{row['Nx']:<9d} {row['dt']:<15.8e} {row['nsteps']:<11d} "
            f"{row['err']:<15.8e} {order}"
        )


if __name__ == "__main__":
    main()
