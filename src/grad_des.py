import copy
import csv
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from inverse.total_loss import loss_fn_signal
from numerics.dg.mesh import cell_edges_from_nodes, create_uniform_nodes_with_ghosts
from physics.bc import BC
from utils.build_physical_data import build_physical_data
from utils.build_solver import build_solver_geometry
from utils.param_func import set_param
from utils.parse_args import parse_args
from utils.res_openwind import run_openwind_reference
from utils.solve import forward_snapshots
from plot_individual_protocol import make_plot


jax.config.update("jax_enable_x64", True)
print(jax.devices())


DEFAULT_INIT_FACTOR = 0.8
MIN_SCALE = 1e-8
P_CLOSED = 5e3
OPENWIND_CONTROLLED_PARAMS = {"gamma_final", "zeta", "fr", "Qr"}

N_ITER = 200
LR_TRAIN = 5e-3
LR_PRECISE = 1e-3

TOL_TRAIN = 1e-4
TOL_PRECISE = 1e-3

GEO_KEYS = ("L_tube", "R_tube", "L_bell", "k_bell")

PARAM_JSON_PATHS = {
    "epsilon": ("left_bc_params", "epsilon"),
    "beta": ("right_bc_params", "beta"),
    "alpha": ("right_bc_params", "alpha"),
    "zeta": ("left_bc_params", "zeta"),
    "kappa": ("left_bc_params", "kappa"),
    "Zt": ("right_bc_params", "Zt"),
    "fr": ("left_bc_params", "fr"),
    "gamma_final": ("left_bc_params", "mouth_pressure_params", "gamma_final"),
    "t_attack": ("left_bc_params", "mouth_pressure_params", "t_attack"),
    "L_tube": ("instrument_geometry", "tube", "L_tube"),
    "R_tube": ("instrument_geometry", "tube", "R_tube"),
    "L_bell": ("instrument_geometry", "bell", "L_bell"),
    "k_bell": ("instrument_geometry", "bell", "k_bell"),
    "Qr": ("left_bc_params", "Qr"),
}

PARAM_DISPLAY_NAMES = {
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "zeta": r"$\zeta$",
    "Zt": r"$Z_t$",
    "gamma_final": r"$\gamma$",
    "fr": r"$f_r$",
    "Qr": r"$Q_r$",
}


def get_nested(mapping, path):
    value = mapping
    for key in path:
        value = value[key]
    return value


def set_nested(mapping, path, value):
    current = mapping
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = float(value)


def set_single_trainable(params, trainable_name):
    params = copy.deepcopy(params)
    for name in params["trainable"]:
        params["trainable"][name] = name == trainable_name
    return params


def active_inverse_params(params):
    names = [
        name
        for name in PARAM_JSON_PATHS
        if params["trainable"].get(name, False)
    ]
    if len(names) == 0:
        raise ValueError("Aucun parametre n'est marque true dans params['trainable'].")
    return names


def values_from_params(params, names):
    return jnp.asarray(
        [float(get_nested(params, PARAM_JSON_PATHS[name])) for name in names],
        dtype=jnp.float64,
    )


def value_from_params(params, name):
    return float(get_nested(params, PARAM_JSON_PATHS[name]))


def make_initial_values(true_values):
    init = true_values * DEFAULT_INIT_FACTOR
    return jnp.where(jnp.abs(init) < MIN_SCALE, true_values + 1e-2, init)


def make_scales(true_values):
    return jnp.maximum(jnp.abs(true_values), MIN_SCALE)


def set_all_params(data, names, values):
    for name, value in zip(names, values):
        data = set_param(data, name, value, GEO_KEYS)
    return data


def params_from_theta(theta, scales):
    return theta * scales


def relative_error(params_current, true_value):
    denom = jnp.maximum(jnp.abs(true_value), MIN_SCALE)
    err_vec = jnp.abs((params_current - true_value) / denom)
    return float(jnp.linalg.norm(err_vec)), err_vec


def make_solver_data(T_max, CFL, Nx, N_snapshot, L_ref, c, bc, phi0, y0, z0):
    x_nodes, _ = create_uniform_nodes_with_ghosts(Nx, 0.0, L_ref)
    xLs, xRs = cell_edges_from_nodes(x_nodes)

    dt = CFL * (xRs[0] - xLs[0]) / c
    nsteps = int(jnp.ceil(T_max / dt))

    t_solver = jnp.arange(nsteps) * dt
    n_snaps = jnp.round(jnp.linspace(0, nsteps - 1, N_snapshot)).astype(jnp.int32)

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


def make_openwind_bell_target(params_true, type_S, T_max, snapshot_times, args):
    t_ow, _, p_right, _, _, ow_params = run_openwind_reference(
        param_json=params_true,
        T_max=T_max,
        type_S=type_S,
        theta=args.ow_theta,
        l_ele=args.ow_l_ele,
        order=args.ow_order,
    )

    target = np.interp(
        np.asarray(snapshot_times, dtype=float),
        np.asarray(t_ow, dtype=float),
        np.asarray(p_right, dtype=float),
    )

    print(
        f"  OpenWind: dt={ow_params['dt']:.6e}, "
        f"n={len(t_ow)}, h_eff={ow_params['h_eff']:.6e}"
    )
    print(f"  pression OpenWind divisee par P_CLOSED={P_CLOSED:.1f}")

    return jnp.asarray(target / P_CLOSED, dtype=jnp.float64), ow_params


def make_target(target_source, data_true, params_true, geometry, c, solve_kwargs, type_S, T_max, args):
    if target_source == "dg":
        return forward_snapshots(data_true, geometry, c, **solve_kwargs)

    snapshot_times = (solve_kwargs["n_snaps"] + 1) * solve_kwargs["dt"]
    target, _ = make_openwind_bell_target(
        params_true,
        type_S,
        T_max,
        snapshot_times,
        args,
    )
    return target


def make_loss_fn(data_init, geometry, c, target_snaps, inverse_params, scales, solve_kwargs):
    def loss(theta):
        data = data_init
        params_phys = theta * scales

        for name, value in zip(inverse_params, params_phys):
            data = set_param(data, name, value, GEO_KEYS)

        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(pred, target_snaps)

    return loss


def run_optimization(theta_current, loss_and_grad, true_value, scales, lr, n_iter, tol, label):
    optimizer, scheduler = make_optimizer(lr, n_iter)
    opt_state = optimizer.init(theta_current)

    print(f"\n=== Optimisation : {label} ===")
    print(f"  LR initial = {lr}")
    print(f"{'iter':>5} | {'loss':>12} | {'err_rel':>12} | {'lr':>10} | {'t/iter':>8}")
    print("-" * 60)

    history = {"iter": [], "loss": [], "params": [], "err": [], "lr": []}

    for i in range(n_iter):
        t_it = time.time()
        current_lr = float(scheduler(i))

        loss_val, grad_val = loss_and_grad(theta_current)
        updates, opt_state = optimizer.update(grad_val, opt_state, theta_current)
        theta_current = optax.apply_updates(theta_current, updates)

        params_current = params_from_theta(theta_current, scales)
        err, err_vec = relative_error(params_current, true_value)
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
            print(f"\nConvergence atteinte a l'iteration {i}.")
            break

    return theta_current, history


def protocol_param_values(base_value, n_signals, rng):
    factors = rng.uniform(0.75, 1.25, size=n_signals)
    return np.asarray(base_value * factors, dtype=float)


def protocol_target_params(base_params, param_name, target_value, rng):
    params_target = copy.deepcopy(base_params)

    if param_name in OPENWIND_CONTROLLED_PARAMS:
        set_nested(params_target, PARAM_JSON_PATHS[param_name], target_value)
        varied_name = param_name
        varied_value = target_value
    else:
        gamma_base = value_from_params(base_params, "gamma_final")
        varied_value = gamma_base * rng.uniform(0.75, 1.25)
        set_nested(params_target, PARAM_JSON_PATHS["gamma_final"], varied_value)
        varied_name = "gamma_final"

    return params_target, varied_name, float(varied_value)


def params_with_openwind_radiation(base_params, ow_params):
    params = copy.deepcopy(base_params)

    if ow_params.get("alpha") is not None:
        set_nested(params, PARAM_JSON_PATHS["alpha"], ow_params["alpha"])
    if ow_params.get("beta") is not None:
        set_nested(params, PARAM_JSON_PATHS["beta"], ow_params["beta"])

    return params


def make_protocol_loss_and_grad(
    param_name,
    data_init,
    geometry,
    c,
    solve_kwargs,
):
    def loss(theta, scales, target_snaps):
        params_phys = theta * scales
        data = set_param(data_init, param_name, params_phys[0], GEO_KEYS)
        pred = forward_snapshots(data, geometry, c, **solve_kwargs)
        return loss_fn_signal(pred, target_snaps)

    return jax.jit(jax.value_and_grad(loss))


def fit_single_parameter(
    param_name,
    true_value,
    target_snaps,
    loss_and_grad,
    n_iter,
    print_every,
):
    true_arr = jnp.asarray([true_value], dtype=jnp.float64)
    scales = make_scales(true_arr)
    theta = make_initial_values(true_arr) / scales

    optimizer, scheduler = make_optimizer(LR_TRAIN, n_iter)
    opt_state = optimizer.init(theta)

    t0 = time.time()
    final_loss = None
    print_every = max(int(print_every), 1)

    print(
        f"{'iter':>5} | {'loss':>12} | {param_name:>12} | "
        f"{'err_rel':>12} | {'lr':>10} | {'t/iter':>8}"
    )
    print("-" * 72)

    for i in range(n_iter):
        t_it = time.time()
        loss_val, grad_val = loss_and_grad(theta, scales, target_snaps)
        updates, opt_state = optimizer.update(grad_val, opt_state, theta)
        theta = optax.apply_updates(theta, updates)
        final_loss = loss_val

        if i < 5 or i % print_every == 0 or i == n_iter - 1:
            estimated_i = float(params_from_theta(theta, scales)[0])
            rel_error_i = abs(estimated_i - true_value) / max(abs(true_value), MIN_SCALE)
            current_lr = float(scheduler(i))
            elapsed_i = time.time() - t_it
            print(
                f"{i:>5} | "
                f"{float(loss_val):>12.4e} | "
                f"{estimated_i:>12.6g} | "
                f"{rel_error_i:>12.4e} | "
                f"{current_lr:>10.3e} | "
                f"{elapsed_i:>7.2f}s"
            )

    estimated = float(params_from_theta(theta, scales)[0])
    rel_error = abs(estimated - true_value) / max(abs(true_value), MIN_SCALE)

    if final_loss is None:
        final_loss = loss_and_grad(theta, scales, target_snaps)[0]

    return {
        "estimated_value": estimated,
        "relative_error": float(rel_error),
        "final_loss": float(final_loss),
        "elapsed_sec": time.time() - t0,
        "final_lr": float(scheduler(max(n_iter - 1, 0))),
    }


def latex_escape(value):
    return str(value).replace("_", r"\_")


def write_latex_tables(rows, output_dir):
    detail_path = os.path.join(output_dir, "individual_protocol_results.tex")
    summary_path = os.path.join(output_dir, "individual_protocol_summary.tex")

    with open(detail_path, "w") as f:
        f.write("\\begin{tabular}{llrrrrrr}\n")
        f.write("\\hline\n")
        f.write("Param & OW ctrl. & Signal & True & Est. & Rel. err. & Loss & Train time (s) \\\\\n")
        f.write("\\hline\n")
        for row in rows:
            f.write(
                f"{PARAM_DISPLAY_NAMES.get(row['parameter'], latex_escape(row['parameter']))} & "
                f"{row['openwind_controlled']} & "
                f"{row['signal_idx']} & "
                f"{row['true_value']:.6g} & "
                f"{row['estimated_value']:.6g} & "
                f"{row['relative_error']:.3e} & "
                f"{row['final_loss']:.3e} & "
                f"{row['elapsed_sec']:.2f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    params = sorted({row["parameter"] for row in rows})
    with open(summary_path, "w") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\hline\n")
        f.write(
            "Param & Mean rel. err. & Std rel. err. & Mean loss & "
            "N & Total train (s) & Mean train (s) \\\\\n"
        )
        f.write("\\hline\n")
        for param in params:
            param_rows = [row for row in rows if row["parameter"] == param]
            rel = np.asarray([row["relative_error"] for row in param_rows], dtype=float)
            loss = np.asarray([row["final_loss"] for row in param_rows], dtype=float)
            train_time = np.asarray([row["elapsed_sec"] for row in param_rows], dtype=float)
            f.write(
                f"{PARAM_DISPLAY_NAMES.get(param, latex_escape(param))} & "
                f"{float(np.mean(rel)):.3e} & "
                f"{float(np.std(rel)):.3e} & "
                f"{float(np.mean(loss)):.3e} & "
                f"{len(param_rows)} & "
                f"{float(np.sum(train_time)):.2f} & "
                f"{float(np.mean(train_time)):.2f} \\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")

    return detail_path, summary_path


def run_individual_protocol(args, params, solver_params):
    output_dir = args.protocol_output_dir
    os.makedirs(output_dir, exist_ok=True)

    train_params = solver_params["train"]
    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    data_ref = build_physical_data(params, args.type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell
    geometry = build_solver_geometry(data_ref, train_params["Nx"], c)
    bc = BC(type="full")

    _, _, solve_kwargs = make_solver_data(
        train_params["T_max"],
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    snapshot_times = (solve_kwargs["n_snaps"] + 1) * solve_kwargs["dt"]
    rng = np.random.default_rng(args.protocol_seed)
    protocol_params = [p.strip() for p in args.protocol_params.split(",") if p.strip()]

    protocol_cases = []
    rows = []
    csv_path = os.path.join(output_dir, "individual_protocol_results.csv")
    compiled_losses = {}

    for param_name in protocol_params:
        if param_name not in PARAM_JSON_PATHS:
            raise ValueError(f"Parametre inconnu pour le protocole: {param_name}")

        openwind_controlled = param_name in OPENWIND_CONTROLLED_PARAMS
        base_value = value_from_params(params, param_name)
        target_values = protocol_param_values(base_value, args.protocol_n_signals, rng)

        print("\n" + "=" * 80)
        print(f"PROTOCOLE PARAMETRE: {param_name}")
        print(f"OpenWind controle directement ce parametre: {openwind_controlled}")
        print("=" * 80)

        for signal_idx, sampled_value in enumerate(target_values):
            params_target, varied_name, varied_value = protocol_target_params(
                params,
                param_name,
                sampled_value,
                rng,
            )

            print(
                f"\n[{param_name}] signal {signal_idx + 1}/{args.protocol_n_signals} | "
                f"variation OpenWind: {varied_name}={varied_value:.6g}"
            )

            target_snaps, ow_params = make_openwind_bell_target(
                params_target,
                args.type_S,
                train_params["T_max"],
                snapshot_times,
                args,
            )

            dg_base_params = params_with_openwind_radiation(params, ow_params)
            dg_alpha = value_from_params(dg_base_params, "alpha")
            dg_beta = value_from_params(dg_base_params, "beta")

            if param_name == "alpha" and ow_params.get("alpha") is not None:
                true_value = float(ow_params["alpha"])
            elif param_name == "beta" and ow_params.get("beta") is not None:
                true_value = float(ow_params["beta"])
            elif openwind_controlled:
                true_value = float(sampled_value)
            else:
                true_value = base_value

            protocol_cases.append(
                {
                    "parameter": param_name,
                    "signal_idx": signal_idx,
                    "openwind_controlled": openwind_controlled,
                    "openwind_varied_name": varied_name,
                    "openwind_varied_value": varied_value,
                    "sampled_value": float(sampled_value),
                    "true_value": true_value,
                    "target_snaps": target_snaps,
                    "ow_params": ow_params,
                    "dg_base_params": dg_base_params,
                    "dg_alpha": dg_alpha,
                    "dg_beta": dg_beta,
                }
            )

    print("\n" + "=" * 80)
    print(f"PRE-GENERATION OPENWIND TERMINEE: {len(protocol_cases)} signaux")
    print("Debut des entrainements individuels")
    print("=" * 80)

    for case in protocol_cases:
        param_name = case["parameter"]
        target_snaps = case["target_snaps"]
        ow_params = case["ow_params"]
        dg_alpha = case["dg_alpha"]
        dg_beta = case["dg_beta"]
        cache_key = (param_name, dg_alpha, dg_beta)

        print(
            f"\n[{param_name}] entrainement signal "
            f"{case['signal_idx'] + 1}/{args.protocol_n_signals}"
        )

        if cache_key not in compiled_losses:
            params_init = set_single_trainable(case["dg_base_params"], param_name)
            data_init = build_physical_data(params_init, args.type_S)
            compiled_losses[cache_key] = make_protocol_loss_and_grad(
                param_name,
                data_init,
                geometry,
                c,
                solve_kwargs,
            )
            print(
                "  compilation JAX creee pour "
                f"{param_name} avec alpha={dg_alpha:.6g}, beta={dg_beta:.6g}"
            )
        else:
            print(f"  compilation JAX reutilisee pour {param_name}")

        fit = fit_single_parameter(
            param_name,
            case["true_value"],
            target_snaps,
            compiled_losses[cache_key],
            args.protocol_n_iter,
            args.protocol_print_every,
        )

        row = {
            "parameter": param_name,
            "signal_idx": case["signal_idx"],
            "openwind_controlled": case["openwind_controlled"],
            "openwind_varied_name": case["openwind_varied_name"],
            "openwind_varied_value": case["openwind_varied_value"],
            "sampled_value": case["sampled_value"],
            "true_value": case["true_value"],
            "estimated_value": fit["estimated_value"],
            "relative_error": fit["relative_error"],
            "final_loss": fit["final_loss"],
            "elapsed_sec": fit["elapsed_sec"],
            "n_iter": args.protocol_n_iter,
            "target_max_abs": float(jnp.max(jnp.abs(target_snaps))),
            "target_std": float(jnp.std(target_snaps)),
            "ow_alpha": ow_params.get("alpha"),
            "ow_beta": ow_params.get("beta"),
            "ow_Zplus": ow_params.get("Zplus"),
            "dg_alpha": dg_alpha,
            "dg_beta": dg_beta,
            "note": (
                "target parameter injected in OpenWind"
                if case["openwind_controlled"]
                else "parameter not directly controlled by current OpenWind wrapper"
            ),
        }
        rows.append(row)

        print(
            f"  true={case['true_value']:.6g} | "
            f"estimated={fit['estimated_value']:.6g} | "
            f"relerr={fit['relative_error']:.3e} | "
            f"loss={fit['final_loss']:.3e}"
        )

    for param_name in sorted({row["parameter"] for row in rows}):
        param_rows = [row for row in rows if row["parameter"] == param_name]
        train_times = np.asarray([row["elapsed_sec"] for row in param_rows], dtype=float)
        total_train_time = float(np.sum(train_times))
        mean_train_time = float(np.mean(train_times))

        for row in param_rows:
            row["param_train_time_total_sec"] = total_train_time
            row["param_train_time_mean_sec"] = mean_train_time

        print(
            f"Temps entrainement {param_name}: "
            f"total={total_train_time:.2f}s | moyen={mean_train_time:.2f}s"
        )

    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    detail_tex, summary_tex = write_latex_tables(rows, output_dir)
    plot_path = os.path.join(output_dir, "individual_protocol_values.png")
    make_plot(csv_path, plot_path)

    print("\n=== Protocole termine ===")
    print("CSV detaille :", csv_path)
    print("Table LaTeX detaillee :", detail_tex)
    print("Table LaTeX resume :", summary_tex)
    print("Figure parametres :", plot_path)


def main():
    start_time = time.time()
    args = parse_args()

    with open("../experiments/gradient/config/simu.json", "r") as f:
        solver_params = json.load(f)["solver_params"]

    with open("../experiments/gradient/config/param.json", "r") as f:
        params = json.load(f)

    if args.experiment_protocol == "individual":
        run_individual_protocol(args, params, solver_params)
        return

    train_params = solver_params["train"]
    valid_params = solver_params["valid"]

    c = params["physics"]["c"]
    phi0 = params["physics"]["phi0"]
    y0 = params["init_cond_reed"]["y0"]
    z0 = params["init_cond_reed"]["y_dot0"]

    n_iter_train = args.n_iter if args.n_iter is not None else N_ITER
    n_iter_precise = args.n_iter_precise if args.n_iter_precise is not None else N_ITER // 2

    inverse_params = active_inverse_params(params)
    true_value = values_from_params(params, inverse_params)
    init_value = make_initial_values(true_value)
    scales = make_scales(true_value)

    print("\n=== Parametres inverses ===")
    print("  noms          :", inverse_params)
    print("  valeurs vraies:", true_value)
    print("  init          :", init_value)
    print("  scales        :", scales)
    print("  cible         :", args.target_source)
    print("  loss          : spectrale")

    os.makedirs("../experiments/gradient/results", exist_ok=True)

    data_ref = build_physical_data(params, args.type_S)
    L_ref = data_ref.section.L_tube + data_ref.section.L_bell
    bc = BC(type="full")

    geometry_train = build_solver_geometry(data_ref, train_params["Nx"], c)
    dt_train, nsteps_train, solve_kwargs_train = make_solver_data(
        train_params["T_max"],
        train_params["cfl"],
        train_params["Nx"],
        train_params["N_snapshot"],
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"Train: dt={dt_train:.6e}, nsteps={nsteps_train}, T_max={train_params['T_max']:.4f}")

    print("\n=== Generation cible ===")

    params_true = copy.deepcopy(params)
    for name in inverse_params:
        params_true["trainable"][name] = True

    data_true = build_physical_data(params_true, args.type_S)
    data_true = set_all_params(data_true, inverse_params, true_value)

    target_snaps = make_target(
        args.target_source,
        data_true,
        params_true,
        geometry_train,
        c,
        solve_kwargs_train,
        args.type_S,
        train_params["T_max"],
        args,
    )

    print(
        f"  cible : max={float(jnp.max(jnp.abs(target_snaps))):.4e}  "
        f"mean={float(jnp.mean(target_snaps)):.4e}  "
        f"std={float(jnp.std(target_snaps)):.4e}"
    )

    params_init = copy.deepcopy(params)
    for name in inverse_params:
        params_init["trainable"][name] = True

    data_init = build_physical_data(params_init, args.type_S)
    data_test = set_all_params(data_init, inverse_params, init_value)

    pred_init = forward_snapshots(data_test, geometry_train, c, **solve_kwargs_train)
    l_init = float(loss_fn_signal(pred_init, target_snaps))
    l_zero = float(loss_fn_signal(target_snaps * 0.0, target_snaps))

    print("\n=== Sensibilite initiale ===")
    print(f"  loss(init) = {l_init:.4e}")
    print(f"  loss(zero) = {l_zero:.4e}")
    print(f"  ratio      = {l_init / l_zero:.4e}")

    theta_current = init_value / scales
    train_loss = make_loss_fn(
        data_init,
        geometry_train,
        c,
        target_snaps,
        inverse_params,
        scales,
        solve_kwargs_train,
    )
    loss_and_grad = jax.jit(jax.value_and_grad(train_loss))

    theta_current, _ = run_optimization(
        theta_current,
        loss_and_grad,
        true_value,
        scales,
        LR_TRAIN,
        n_iter_train,
        TOL_TRAIN,
        "train",
    )

    params_current = params_from_theta(theta_current, scales)
    err_params, err_vec = relative_error(params_current, true_value)

    print("\nAfter training:")
    print("=" * 50)
    print("Valeurs vraies      :", true_value)
    print("Valeurs finales     :", params_current)
    print("Erreurs relatives   :", err_vec)
    print(f"Erreur relative globale : {err_params:.4e}")
    print(f"Temps total train   : {time.time() - start_time:.2f}s")

    if err_params < TOL_PRECISE:
        print("\nPrecision suffisante atteinte.")
        print("Raffinement precis ignore.")
        return

    if args.skip_precise:
        print("\nRaffinement precis ignore (--skip_precise).")
        return

    print("\n=== Raffinement precis ===")

    geometry_valid = build_solver_geometry(data_ref, valid_params["Nx"], c)
    dt_valid, nsteps_valid, solve_kwargs_precise = make_solver_data(
        valid_params["T_max"],
        valid_params["cfl"],
        valid_params["Nx"],
        valid_params["N_snapshot"],
        L_ref,
        c,
        bc,
        phi0,
        y0,
        z0,
    )

    print(f"Precise: dt={dt_valid:.6e}, nsteps={nsteps_valid}, T_max={valid_params['T_max']:.4f}")

    target_snaps_precise = make_target(
        args.target_source,
        data_true,
        params_true,
        geometry_valid,
        c,
        solve_kwargs_precise,
        args.type_S,
        valid_params["T_max"],
        args,
    )

    precise_loss = make_loss_fn(
        data_init,
        geometry_valid,
        c,
        target_snaps_precise,
        inverse_params,
        scales,
        solve_kwargs_precise,
    )
    loss_and_grad_precise = jax.jit(jax.value_and_grad(precise_loss))

    theta_current, _ = run_optimization(
        theta_current,
        loss_and_grad_precise,
        true_value,
        scales,
        LR_PRECISE,
        n_iter_precise,
        TOL_PRECISE,
        "precise",
    )

    params_current = params_from_theta(theta_current, scales)
    err_params, err_vec = relative_error(params_current, true_value)

    print("\nAfter precise refinement:")
    print("=" * 50)
    print("Valeurs vraies      :", true_value)
    print("Valeurs finales     :", params_current)
    print("Erreurs relatives   :", err_vec)
    print(f"Erreur relative globale : {err_params:.4e}")
    print(f"Temps total         : {time.time() - start_time:.2f}s")


if __name__ == "__main__":
    main()
