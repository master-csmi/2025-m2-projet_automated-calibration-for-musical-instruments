# ======================================================================================
# Problème inverse — loss FFT + jitter sur le paramètre
# ======================================================================================

import os
import jax
import jax.numpy as jnp
import json
import copy
import time
import optax
import equinox as eqx

from utils.parse_args import parse_args
from numerics.dg.mesh import create_uniform_nodes_with_ghosts, cell_edges_from_nodes
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from src.inverse.spectral_loss import multi_resolution_spectral_loss
from utils.solve import forward_snapshots
from utils.param_func import get_param, set_param
from inverse.total_loss import loss_fn_signal, loss_fn

jax.config.update("jax_enable_x64", True)
print(jax.devices())

# ======================================================================================
# Paramètres du problème inverse
# ======================================================================================
INVERSE_PARAM = "alpha"   # paramètre à retrouver
TRUE_VALUE    = 0.033     # valeur cible
INIT_VALUE    = 0.01      # valeur initiale
N_ITER        = 300       # nombre d'itérations
LR_TRAIN      = 1e-3      # learning rate Adam
LR_VALID      = 1e-4      # learning rate Adam

# Jitter
N_JITTER      = 4         # nombre de réalisations bruitées par itération
SIGMA_JITTER  = 1e-3      # écart-type du bruit sur le paramètre (relatif à l'échelle du param)

# Pondération loss
LAMBDA_L2     = 1.0       # poids terme L2 temporel
LAMBDA_FFT    = 0.1       # poids terme FFT fréquentiel

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")



def main():
    start_time = time.time()

    # --------------------------------------------------------------------------
    # Configs
    # --------------------------------------------------------------------------
    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params   = json.load(f)["solver_params"]

        # parameters for training
        train_params    = solver_params["train"]
        T_max_train     = train_params["T_max"]
        CFL_train       = train_params["cfl"]
        Nx_train        = train_params["Nx"]
        N_snapshot_time_train = train_params["N_snapshot"]

        # parameters for validation
        valid_params   = solver_params["valid"]
        T_max_valid    = valid_params["T_max"]
        CFL_valid      = valid_params["cfl"]
        Nx_valid       = valid_params["Nx"]
        N_snapshot_time_valid = valid_params["N_snapshot"]


    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    c    = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0   = params["init_cond_reed"]["y0"]
    z0   = params["init_cond_reed"]["y_dot0"]

    args   = parse_args()
    type_S = args.type_S

    # --------------------------------------------------------------------------
    # Géométrie + pas de temps fixes pour train
    # --------------------------------------------------------------------------
    data_ref = build_physical_data(params, type_S)
    L_ref    = data_ref.section.L_tube + data_ref.section.L_bell

    x_nodes_ref, _ = create_uniform_nodes_with_ghosts(Nx_train, 0.0, L_ref)
    xLs_ref, xRs_ref = cell_edges_from_nodes(x_nodes_ref)
    dt     = CFL_train * (xRs_ref[0] - xLs_ref[0]) / c
    nsteps = int(jnp.ceil(T_max_train / dt))
    print(f"dt={dt:.6e}, nsteps={nsteps}, T_max={T_max_train:.4f}")

    t_solver = jnp.arange(nsteps) * dt
    n_snaps  = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot_time_train)
    ).astype(jnp.int32)
    x_plot   = jnp.linspace(0.0, L_ref, 1000)

    bc = BC(type="full")
    os.makedirs("../experiments/gradient/results", exist_ok=True)

    solve_kwargs_train = dict(dt=dt, nsteps=nsteps, bc=bc, phi0=phi0, y0=y0, z0=z0, t_solver=t_solver, n_snaps=n_snaps)



        

    # --------------------------------------------------------------------------
    # Génération de la cible
    # --------------------------------------------------------------------------
    print(f"\n=== Génération cible ({INVERSE_PARAM}={TRUE_VALUE}) ===")
    params_true = copy.deepcopy(params)
    params_true["trainable"][INVERSE_PARAM] = True
    data_true    = build_physical_data(params_true, type_S)
    data_true    = set_param(data_true, INVERSE_PARAM, jnp.array(TRUE_VALUE), GEO_KEYS)
    forward_snapshots_jit = eqx.filter_jit(forward_snapshots)
    target_snaps = forward_snapshots_jit(data_true, Nx_train, c, dt, nsteps, bc, phi0, y0, z0, t_solver, n_snaps)
    print(f"  p_bell : max={float(jnp.max(jnp.abs(target_snaps))):.4e}  "
          f"mean={float(jnp.mean(target_snaps)):.4e}  "
          f"std={float(jnp.std(target_snaps)):.4e}")

    # --------------------------------------------------------------------------
    # Init data
    # --------------------------------------------------------------------------
    params_init = copy.deepcopy(params)
    params_init["trainable"][INVERSE_PARAM] = True
    data_init = build_physical_data(params_init, type_S)

    # --------------------------------------------------------------------------
    # Vérification sensibilité avant d'optimiser
    # --------------------------------------------------------------------------
    print(f"\n=== Sensibilité ===")
    p_init = forward_snapshots_jit(set_param(data_init, INVERSE_PARAM, jnp.array(INIT_VALUE), GEO_KEYS), Nx_train, c, **solve_kwargs_train)
    l_init = float(loss_fn_signal(p_init, target_snaps))
    l_norm = float(loss_fn_signal(target_snaps * 0.0, target_snaps))
    print(f"  loss(init)   = {l_init:.4e}")
    print(f"  loss(zero)   = {l_norm:.4e}   (baseline : prédire zéro)")
    print(f"  ratio        = {l_init/l_norm:.4e}   (>1e-3 = bon signal)")



    def loss_train(param, key, data_init, target_snaps):
        return loss_fn(
            param, key, data_init,
            Nx_train, c, target_snaps,
            INVERSE_PARAM, GEO_KEYS, solve_kwargs_train
        )



    loss_and_grad = jax.jit(jax.value_and_grad(loss_train))

    # --------------------------------------------------------------------------
    # Optimisation
    # --------------------------------------------------------------------------
    param_current = jnp.array(INIT_VALUE)
    optimizer     = optax.adam(LR_TRAIN)
    opt_state     = optimizer.init(param_current)
    key           = jax.random.PRNGKey(0)

    print(f"\n=== Descente de gradient ({INVERSE_PARAM}) ===")
    print(f"  Vrai={TRUE_VALUE}  Init={INIT_VALUE}  LR={LR_TRAIN}")
    print(f"  Jitter: N={N_JITTER}, sigma={SIGMA_JITTER}")
    print(f"  Loss: L2 x{LAMBDA_L2} + FFT x{LAMBDA_FFT}")
    print(f"{'iter':>5} | {'loss':>12} | {'valeur':>12} | {'erreur':>10} | {'grad':>10} | {'t/iter':>8}")
    print("-" * 72)

    history = {"iter": [], "loss": [], "val": [], "err": []}

    for i in range(N_ITER):
        t_it = time.time()
        key, subkey = jax.random.split(key)
        loss_val, grad_val = loss_and_grad(param_current, subkey, data_init, target_snaps)
        updates, opt_state = optimizer.update(grad_val, opt_state)
        param_current      = optax.apply_updates(param_current, updates)

        current = float(param_current)
        err     = abs(current - TRUE_VALUE)
        elapsed = time.time() - t_it

        history["iter"].append(i)
        history["loss"].append(float(loss_val))
        history["val"].append(current)
        history["err"].append(err)

        if i % 10 == 0 or i < 5:
            print(f"{i:>5} | {float(loss_val):>12.4e} | {current:>12.6f} "
                  f"| {err:>10.4e} | {float(grad_val):>10.3e} | {elapsed:>7.2f}s")

        if err < 1e-5:
            print(f"\n  ✓ Convergé à l'itération {i} !")
            break

    # --------------------------------------------------------------------------
    # Résumé
    # --------------------------------------------------------------------------
    best_i   = int(jnp.argmin(jnp.array(history["err"])))
    best_val = history["val"][best_i]
    best_err = history["err"][best_i]
    print("After training : \n")
    print(f"\n{'='*50}")
    print(f"  Valeur vraie      : {TRUE_VALUE:.6f}")
    print(f"  Valeur initiale   : {INIT_VALUE:.6f}")
    print(f"  Valeur finale     : {float(param_current):.6f}  (err={abs(float(param_current)-TRUE_VALUE):.2e})")
    print(f"  Meilleure valeur  : {best_val:.6f}  (iter={best_i}, err={best_err:.2e})")
    print(f"  Temps total       : {time.time()-start_time:.2f}s")

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    print(f"\n=== Validation sur un maillage plus fin ===")

    x_nodes_ref, _ = create_uniform_nodes_with_ghosts(Nx_valid, 0.0, L_ref)
    xLs_ref, xRs_ref = cell_edges_from_nodes(x_nodes_ref)
    dt     = CFL_valid * (xRs_ref[0] - xLs_ref[0]) / c
    nsteps = int(jnp.ceil(T_max_valid / dt))
    print(f"dt={dt:.6e}, nsteps={nsteps}, T_max={T_max_valid:.4f}")

    t_solver = jnp.arange(nsteps) * dt
    n_snaps  = jnp.round(
        jnp.linspace(0, nsteps - 1, N_snapshot_time_valid)
    ).astype(jnp.int32)
    x_plot   = jnp.linspace(0.0, L_ref, 1000)

    solve_kwargs_valid = dict(dt=dt, nsteps=nsteps, bc=bc, phi0=phi0, y0=y0, z0=z0, t_solver=t_solver, n_snaps=n_snaps)
    optimizer     = optax.adam(LR_VALID)
    print(f"\n=== Descente de gradient ({INVERSE_PARAM}) ===")
    print(f"  Vrai={TRUE_VALUE}  Init={INIT_VALUE}  LR={LR_VALID}")
    print(f"  Jitter: N={N_JITTER}, sigma={SIGMA_JITTER}")
    print(f"  Loss: L2 x{LAMBDA_L2} + FFT x{LAMBDA_FFT}")
    print(f"{'iter':>5} | {'loss':>12} | {'valeur':>12} | {'erreur':>10} | {'grad':>10} | {'t/iter':>8}")
    print("-" * 72)

    target_snaps_valid = forward_snapshots_jit(
        data_true,
        Nx_valid,
        c,
        **solve_kwargs_valid
        )
    print(f"\n=== Sensibilité ===")
    p_init_valid = forward_snapshots_jit(set_param(data_init, INVERSE_PARAM, jnp.array(INIT_VALUE), GEO_KEYS), Nx_valid, c, **solve_kwargs_valid)
    l_init = float(loss_fn_signal(p_init_valid, target_snaps_valid))
    l_norm = float(loss_fn_signal(target_snaps_valid * 0.0, target_snaps_valid))
    print(f"  loss(init)   = {l_init:.4e}")
    print(f"  loss(zero)   = {l_norm:.4e}   (baseline : prédire zéro)")
    print(f"  ratio        = {l_init/l_norm:.4e}   (>1e-3 = bon signal)")


    def loss_valid(param, key, data_init, target_snaps):
        return loss_fn(
            param, key, data_init,
            Nx_valid, c, target_snaps,
            INVERSE_PARAM, GEO_KEYS, solve_kwargs_valid
        )



    loss_and_grad = jax.jit(jax.value_and_grad(loss_valid))

    history = {"iter": [], "loss": [], "val": [], "err": []}

    for i in range(N_ITER):
        t_it = time.time()
        key, subkey = jax.random.split(key)

        loss_val, grad_val = loss_and_grad(param_current, subkey, data_init, target_snaps_valid)

        updates, opt_state = optimizer.update(grad_val, opt_state)
        param_current      = optax.apply_updates(param_current, updates)

        current = float(param_current)
        err     = abs(current - TRUE_VALUE)
        elapsed = time.time() - t_it

        history["iter"].append(i)
        history["loss"].append(float(loss_val))
        history["val"].append(current)
        history["err"].append(err)

        if i % 10 == 0 or i < 5:
            print(f"{i:>5} | {float(loss_val):>12.4e} | {current:>12.6f} "
                  f"| {err:>10.4e} | {float(grad_val):>10.3e} | {elapsed:>7.2f}s")

        if err < 1e-5:
            print(f"\n Convergé à l'itération {i} !")
            break

    # --------------------------------------------------------------------------
    # Résumé
    # --------------------------------------------------------------------------
    best_i   = int(jnp.argmin(jnp.array(history["err"])))
    best_val = history["val"][best_i]
    best_err = history["err"][best_i]
    print("After training : \n")
    print(f"\n{'='*50}")
    print(f"  Valeur vraie      : {TRUE_VALUE:.6f}")
    print(f"  Valeur initiale   : {INIT_VALUE:.6f}")
    print(f"  Valeur finale     : {float(param_current):.6f}  (err={abs(float(param_current)-TRUE_VALUE):.2e})")
    print(f"  Meilleure valeur  : {best_val:.6f}  (iter={best_i}, err={best_err:.2e})")
    print(f"  Temps total       : {time.time()-start_time:.2f}s")


if __name__ == "__main__":
    main()