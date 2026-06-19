import os
import json
import copy
import time

import jax
import jax.numpy as jnp
import optax
import equinox as eqx

from utils.parse_args import parse_args
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.solve import forward_snapshots
from utils.param_func import set_param
from inverse.total_loss import loss_fn_signal, loss_scalars_only


jax.config.update("jax_enable_x64", True)
print(jax.devices())


INVERSE_PARAM = ["alpha"]

TRUE_VALUE = jnp.array([0.0333])
INIT_VALUE = jnp.array([0.01])
SCALES = jnp.array([0.01])

N_ITER = 1500
LR_TRAIN = 1e-2
LR_PRECISE = 1e-3

TOL_TRAIN = 1e-3
TOL_PRECISE = 1e-3

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")


def set_all_params(data, names, values, geo_keys):
    for name, value in zip(names, values):
        data = set_param(data, name, value, geo_keys)
    return data

def params_from_theta(theta):
    return theta * SCALES

def relative_error(params_current):
    err_vec = jnp.abs((params_current - TRUE_VALUE) / TRUE_VALUE)
    err = float(jnp.linalg.norm(err_vec))
    return err, err_vec


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


def make_optimizer(lr, n_iter):
    scheduler = optax.cosine_decay_schedule(
        init_value=lr,
        decay_steps=n_iter,
        alpha=1e-2,
    )

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=scheduler,
            weight_decay=1e-5,
        ),
    )

    return optimizer, scheduler


def run_joint_optimization(
    theta_current,
    loss_and_grad,
    data_init,
    target_snaps,
    lr,
    n_iter,
    tol,
    label,
):
    optimizer, scheduler = make_optimizer(lr, n_iter)
    opt_state = optimizer.init(theta_current)

    print(f"\n=== Optimisation conjointe : {label} ===")
    print(f"  LR initial = {lr}")
    print(f"{'iter':>5} | {'loss':>12} | {'err_rel':>12} | {'lr':>10} | {'t/iter':>8}")
    print("-" * 60)

    history = {"iter": [], "loss": [], "params": [], "err": [], "lr" : []}

    for i in range(n_iter):
        t_it = time.time()

        current_lr = float(scheduler(i))

        loss_val, grad_val = loss_and_grad(
            theta_current,
            data_init,
            target_snaps,
        )

        updates, opt_state = optimizer.update(
            grad_val,
            opt_state,
            theta_current,
        )

        theta_current = optax.apply_updates(theta_current, updates)

        params_current = params_from_theta(theta_current)
        err, err_vec = relative_error(params_current)
        elapsed = time.time() - t_it

        history["iter"].append(i)
        history["loss"].append(float(loss_val))
        history["params"].append(params_current)
        history["err"].append(err)
        history["lr"].append(current_lr)

        if i % 10 == 0 or i < 5:
            print(
                f"{i:>5} | "
                f"{float(loss_val):>12.4e} | "
                f"{err:>12.4e} | "
                f"{current_lr:>10.3e} | "
                f"{elapsed:>7.2f}s"
            )
            print("       params =", params_current)
            print("       relerr =", err_vec)

        if err < tol:
            print(f"\nConvergence atteinte à l'itération {i}.")
            break

    return theta_current, history


def main():
    start_time = time.time()

    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    train_params = solver_params["train"]
    valid_params = solver_params["valid"]

    T_max_train = train_params["T_max"]
    CFL_train = train_params["cfl"]
    Nx_train = train_params["Nx"]
    N_snapshot_time_train = train_params["N_snapshot"]

    T_max_valid = valid_params["T_max"]
    CFL_valid = valid_params["cfl"]
    Nx_valid = valid_params["Nx"]
    N_snapshot_time_valid = valid_params["N_snapshot"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    args = parse_args()
    type_S = args.type_S

    os.makedirs("../experiments/gradient/results", exist_ok=True)

    data_ref = build_physical_data(params, type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell

    bc = BC(type="full")
    forward_snapshots_jit = eqx.filter_jit(forward_snapshots)

    dt_train, nsteps_train, solve_kwargs_train = make_solver_data(
        T_max_train,
        CFL_train,
        Nx_train,
        N_snapshot_time_train,
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"Train: dt={dt_train:.6e}, nsteps={nsteps_train}, T_max={T_max_train:.4f}")

    print("\n=== Génération cible ===")

    params_true = copy.deepcopy(params)
    for name in INVERSE_PARAM:
        params_true["trainable"][name] = True

    data_true = build_physical_data(params_true, type_S)

    data_true = set_all_params(
        data_true,
        INVERSE_PARAM,
        TRUE_VALUE,
        GEO_KEYS,
    )

    target_snaps = forward_snapshots_jit(
        data_true,
        Nx_train,
        c,
        **solve_kwargs_train,
    )

    print(
        f"  p_bell : max={float(jnp.max(jnp.abs(target_snaps))):.4e}  "
        f"mean={float(jnp.mean(target_snaps)):.4e}  "
        f"std={float(jnp.std(target_snaps)):.4e}"
    )

    params_init = copy.deepcopy(params)
    for name in INVERSE_PARAM:
        params_init["trainable"][name] = True

    data_init = build_physical_data(params_init, type_S)

    print("\n=== Sensibilité initiale ===")

    data_test = set_all_params(
        data_init,
        INVERSE_PARAM,
        INIT_VALUE,
        GEO_KEYS,
    )

    p_init = forward_snapshots_jit(
        data_test,
        Nx_train,
        c,
        **solve_kwargs_train,
    )

    l_init = float(
        loss_fn_signal(
            p_init,
            target_snaps,
        )
    )
    l_norm = float(
        loss_fn_signal(
            target_snaps * 0.0,
            target_snaps,
        )
    )

    print(f"  loss(init) = {l_init:.4e}")
    print(f"  loss(zero) = {l_norm:.4e}")
    print(f"  ratio      = {l_init / l_norm:.4e}")

    theta_current = INIT_VALUE / SCALES

    def train_loss(theta, data_init, target_snaps):
        return loss_scalars_only(
            theta,
            data_init,
            Nx_train,
            c,
            target_snaps,
            INVERSE_PARAM,
            SCALES,
            GEO_KEYS,
            solve_kwargs_train,
        )

    loss_and_grad = jax.jit(jax.value_and_grad(train_loss))

    theta_current, history = run_joint_optimization(
        theta_current=theta_current,
        loss_and_grad=loss_and_grad,
        data_init=data_init,
        target_snaps=target_snaps,
        lr=LR_TRAIN,
        n_iter=N_ITER,
        tol=TOL_TRAIN,
        label="train",
    )

    params_current = params_from_theta(theta_current)
    err_params, err_vec = relative_error(params_current)

    print("\nAfter joint training:")
    print("=" * 50)
    print("Valeurs vraies      :", TRUE_VALUE)
    print("Valeurs finales     :", params_current)
    print("Erreurs relatives   :", err_vec)
    print(f"Erreur relative globale : {err_params:.4e}")
    print(f"Temps total train   : {time.time() - start_time:.2f}s")

    if err_params < TOL_PRECISE:
        print("\nPrécision suffisante atteinte.")
        print("Raffinement précis ignoré.")
        return

    print("\n=== Raffinement précis sur maillage plus fin ===")

    dt_valid, nsteps_valid, solve_kwargs_precise = make_solver_data(
        T_max_valid,
        CFL_valid,
        Nx_valid,
        N_snapshot_time_valid,
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"Precise: dt={dt_valid:.6e}, nsteps={nsteps_valid}, T_max={T_max_valid:.4f}")

    target_snaps_precise = forward_snapshots_jit(
        data_true,
        Nx_valid,
        c,
        **solve_kwargs_precise,
    )

    def precise_loss(theta, data_init, target_snaps):
        return loss_scalars_only(
            theta,
            data_init,
            Nx_valid,
            c,
            target_snaps,
            INVERSE_PARAM,
            SCALES,
            GEO_KEYS,
            solve_kwargs_precise,
        )

    loss_and_grad_precise = jax.jit(jax.value_and_grad(precise_loss))

    theta_current, precise_history = run_joint_optimization(
        theta_current=theta_current,
        loss_and_grad=loss_and_grad_precise,
        data_init=data_init,
        target_snaps=target_snaps_precise,
        lr=LR_PRECISE,
        n_iter=N_ITER // 2,
        tol=TOL_PRECISE,
        label="precise",
    )

    params_current = params_from_theta(theta_current)
    err_params, err_vec = relative_error(params_current)

    print("\nAfter precise joint refinement:")
    print("=" * 50)
    print("Valeurs vraies      :", TRUE_VALUE)
    print("Valeurs finales     :", params_current)
    print("Erreurs relatives   :", err_vec)
    print(f"Erreur relative globale : {err_params:.4e}")
    print(f"Temps total         : {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()