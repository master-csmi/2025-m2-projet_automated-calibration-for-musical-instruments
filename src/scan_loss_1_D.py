import os
import json
import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt

from utils.parse_args import parse_args
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.solve import forward_snapshots
from utils.param_func import set_param
from inverse.total_loss import loss_fn_signal


jax.config.update("jax_enable_x64", True)


# ======================================================================================
# Paramètres à scanner
# ======================================================================================

INVERSE_PARAM = [
    "alpha",
    "beta",
    "Zt",
    "kappa",
    "fr",
    "Qr",
    "gamma_final",
    "zeta",
]

TRUE_VALUE = {
    "alpha": 79196.8507605227,
    "beta": 0.6646516378651401,
    "Zt": 1.0,
    "kappa": 0.71,
    "fr": 170.0,
    "Qr": 125.0,
    "gamma_final": 0.422,
    "zeta": 0.4,
}

SCAN_RANGES = {
    "alpha": (6e4,9e4),
    "beta": (0.1, 1.0),
    "Zt": (0.2, 3.0),
    "kappa": (0.1, 2.0),
    "fr": (50.0, 220.0),
    "Qr": (20.0, 250.0),
    "gamma_final": (0.1, 0.9),
    "zeta": (0.1, 0.9),
}

N_SCAN = 100

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

USE_SPECTRAL = True
TIME_WEIGHT = 0.1
SPECTRAL_WEIGHT = 0.9


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
    n_snaps = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot)
    ).astype(jnp.int32)

    solve_kwargs = dict(
        dt=dt,
        nsteps=nsteps,
        bc=bc,
        phi0=phi0,
        y0=y0,
        z0=z0,
        t_solver=t_solver,
        n_snaps=n_snaps,
    )

    return dt, nsteps, solve_kwargs


def set_true_params(data):
    for name, value in TRUE_VALUE.items():
        data = set_param(
            data,
            name,
            jnp.array(value),
            GEO_KEYS,
        )
    return data


def compute_loss(pred, target):
    return loss_fn_signal(
        pred,
        target,
    )


def main():
    output_dir = "../experiments/gradient/results/scans"
    os.makedirs(output_dir, exist_ok=True)

    # --------------------------------------------------------------------------
    # Config
    # --------------------------------------------------------------------------
    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    train_params = solver_params["train"]
    T_max = train_params["T_max"]
    CFL = train_params["cfl"]
    Nx = train_params["Nx"]
    N_snapshot = train_params["N_snapshot"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    for name in INVERSE_PARAM:
        params["trainable"][name] = True

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    args = parse_args()
    type_S = args.type_S

    data_ref = build_physical_data(params, type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell

    bc = BC(type="full")

    dt, nsteps, solve_kwargs = make_solver_data(
        T_max,
        CFL,
        Nx,
        N_snapshot,
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"dt={dt:.6e}, nsteps={nsteps}")
    print(f"loss: time_weight={TIME_WEIGHT}, spectral_weight={SPECTRAL_WEIGHT}")

    forward_jit = eqx.filter_jit(forward_snapshots)

    # --------------------------------------------------------------------------
    # Cible : tous les paramètres à leur vraie valeur
    # --------------------------------------------------------------------------
    data_true = build_physical_data(params, type_S)
    data_true = set_true_params(data_true)

    target = forward_jit(
        data_true,
        Nx,
        c,
        **solve_kwargs,
    )

    print("target shape =", target.shape)

    # --------------------------------------------------------------------------
    # Scan 1D
    # --------------------------------------------------------------------------
    summary = []


    for param_name in INVERSE_PARAM:
        vmin, vmax = SCAN_RANGES[param_name]
        values = jnp.linspace(vmin, vmax, N_SCAN)

        losses = []

        print(f"\n=== Scan {param_name} ===")

        for value in values:
            data_scan = build_physical_data(params, type_S)

            
            # Tous les paramètres à leur vraie valeur
            data_scan = set_true_params(data_scan)

            # Sauf celui qu'on scanne
            data_scan = set_param(
                data_scan,
                param_name,
                value,
                GEO_KEYS,
            )

            pred = forward_jit(
                data_scan,
                Nx,
                c,
                **solve_kwargs,
            )

            loss = compute_loss(pred, target)
            losses.append(float(loss))

        losses = jnp.array(losses)

        idx_min = int(jnp.argmin(losses))
        value_min = float(values[idx_min])
        loss_min = float(losses[idx_min])
        true_value = TRUE_VALUE[param_name]

        rel_error_min = abs(value_min - true_value) / abs(true_value)

        print(f"  vraie valeur       = {true_value:.6g}")
        print(f"  minimum numérique  = {value_min:.6g}")
        print(f"  erreur relative    = {rel_error_min:.4e}")
        print(f"  loss min           = {loss_min:.4e}")

        summary.append(
            {
                "param": param_name,
                "true": true_value,
                "min": value_min,
                "rel_error": rel_error_min,
                "loss_min": loss_min,
            }
        )

        # Sauvegarde numérique
        data_out = jnp.stack([values, losses], axis=1)
        npy_path = f"{output_dir}/scan_{param_name}.npy"
        jnp.save(npy_path, data_out)

        # Figure
        plt.figure()
        plt.plot(values, losses, label="loss")
        plt.axvline(true_value, linestyle="--", label="true")
        plt.axvline(value_min, linestyle=":", label="min scan")
        plt.xlabel(param_name)
        plt.ylabel("loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        fig_path = f"{output_dir}/scan_{param_name}.png"
        plt.savefig(fig_path, dpi=200)
        plt.close()

        print(f"  figure sauvegardée : {fig_path}")
        print(f"  valeurs sauvegardées : {npy_path}")

    # --------------------------------------------------------------------------
    # Résumé global
    # --------------------------------------------------------------------------
    print("\n=== Résumé des scans ===")
    for item in summary:
        print(
            f"{item['param']:>12} | "
            f"true={item['true']:.6g} | "
            f"min={item['min']:.6g} | "
            f"rel_err={item['rel_error']:.4e} | "
            f"loss_min={item['loss_min']:.4e}"
        )


if __name__ == "__main__":
    main()