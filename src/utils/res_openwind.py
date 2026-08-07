import os
import csv
import numpy as np

from openwind import InstrumentGeometry, InstrumentPhysics, TemporalSolver, Player
from openwind.temporal import RecordingDevice
from openwind.technical.temporal_curves import constant_with_initial_ramp

def make_openwind_radius_profile_np(param_json, type_S, n_points=200):
    geom = param_json["instrument_geometry"]
    tube = geom["tube"]
    bell = geom["bell"]

    L_tube = float(tube["L_tube"])
    R_tube = float(tube["R_tube"])
    L_bell = float(bell["L_bell"])
    k_bell = float(bell["k_bell"])

    L = L_tube + L_bell
    x = np.linspace(0.0, L, n_points)

    S0 = np.pi * R_tube**2

    if type_S == "const":
        S = np.full_like(x, S0)
    elif type_S == "exp":
        S_tube = np.full_like(x, S0)
        S_bell = S0 * np.exp(k_bell * (x - L_tube))
        S = np.where(x < L_tube, S_tube, S_bell)
    elif type_S == "cone":
        S_tube = np.full_like(x, S0)
        S_bell = S0 + k_bell * (x - L_tube)
        S = np.where(x < L_tube, S_tube, S_bell)
    else:
        raise ValueError(f"Unknown type_S: {type_S}")

    return x, np.sqrt(S / np.pi)


def make_openwind_bore_native(param_json, type_S, n_points=200):
    geom = param_json["instrument_geometry"]
    tube = geom["tube"]
    bell = geom["bell"]

    L_tube = float(tube["L_tube"])
    R_tube = float(tube["R_tube"])
    L_bell = float(bell["L_bell"])
    k_bell = float(bell["k_bell"])
    L = L_tube + L_bell

    if type_S == "const":
        return [[0.0, L, R_tube, R_tube, "linear"]]

    if type_S == "exp":
        R_end = float(R_tube * np.exp(0.5 * k_bell * L_bell))
        return [
            [0.0, L_tube, R_tube, R_tube, "linear"],
            [L_tube, L, R_tube, R_end, "exponential"],
        ]

    # Le profil "cone" du DG est lineaire en section, pas en rayon.
    # On garde donc l'ancienne discretisation par points pour ne pas changer
    # la geometrie mathematique du cas.
    x_geom, r_geom = make_openwind_radius_profile_np(
        param_json,
        type_S,
        n_points=n_points,
    )
    return [[float(x), float(r)] for x, r in zip(x_geom, r_geom)]


def run_openwind_only_convergence(
    params,
    T_max,
    type_S,
    output_dir,
):
    os.makedirs(output_dir, exist_ok=True)

    results = []
    ow_solutions = []

    print("\n" + "=" * 80)
    print(f"OpenWind seul : type_S={type_S}")
    print("=" * 80)

    t_ow, p_left, p_right, y_ow, gamma_ow, ow_rad_params = run_openwind_reference(
        param_json=params,
        T_max=T_max,
        type_S=type_S,
    )

    ow_solutions.append(
        {
            "t": np.asarray(t_ow),
            "p": np.asarray(p_right),
        }
    )

    for k in range(len(ow_solutions) - 1):
        coarse = ow_solutions[k]
        fine = ow_solutions[k + 1]

        t_coarse = coarse["t"]
        p_coarse = coarse["p"]

        p_fine_interp = np.interp(
            t_coarse,
            fine["t"],
            fine["p"],
        )

        rel_l2 = np.linalg.norm(p_coarse - p_fine_interp) / (
            np.linalg.norm(p_fine_interp) + 1e-12
        )

        results.append(
            {
                "n_points": coarse["n_points"],
                "n_points_ref": fine["n_points"],
                "h": 1.0 / coarse["n_points"],
                "rel_l2": float(rel_l2),
                "rel_l2_shift": float(rel_l2),
                "corr_shifted": None,
                "lag_index": 0,
                "shift_time": 0.0,
                "order_shift": None,
            }
        )

    for i in range(1, len(results)):
        e_prev = results[i - 1]["rel_l2_shift"]
        e_curr = results[i]["rel_l2_shift"]

        if e_prev > 0 and e_curr > 0:
            results[i]["order_shift"] = float(
                np.log(e_prev / e_curr) / np.log(2.0)
            )

    if len(results) == 0:
        print("Pas assez de solutions OpenWind pour calculer une convergence.")
        return []

    csv_path = os.path.join(
        output_dir,
        f"openwind_only_convergence_{type_S}.csv",
    )

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("CSV OpenWind seul sauvegardé :", csv_path)

    print("\n=== Résumé convergence OpenWind seul ===")
    for r in results:
        print(
            f"N={r['n_points']:5d} -> {r['n_points_ref']:5d} | "
            f"rel_l2_shift={r['rel_l2_shift']:.4e} | "
            f"order={r['order_shift']}"
        )

    return results


def run_openwind_reference(
    param_json,
    T_max,
    type_S,
    n_points=200,
    theta=0.5,
    l_ele=None,
    order=4,
):
    geom = param_json["instrument_geometry"]
    tube = geom["tube"]

    left = param_json["left_bc_params"]
    mouth = left["mouth_pressure_params"]

    R_tube = float(tube["R_tube"])

    gamma_final = float(mouth["gamma_final"])
    t_attack = float(mouth["t_attack"])

    zeta = float(left["zeta"])
    kappa = float(left["kappa"])
    fr = float(left["fr"])
    Qr = float(left["Qr"])

    wr = 2.0 * np.pi * fr

    gamma_time = constant_with_initial_ramp(
        pm_max=gamma_final,
        t_car=t_attack,
        t_down=1e10,
    )

    reed = {
        "excitator_type": "Reed1dof_scaled",
        "gamma": gamma_time,
        "zeta": zeta,
        "kappa": kappa,
        "pulsation": wr,
        "qfactor": Qr,
        "model": "inwards",
        "contact_stifness": 0,
        "contact_exponent": 4,
        "opening": 5e-4,
        "closing_pressure": 5e3,
    }

    player = Player(reed)

    bore = make_openwind_bore_native(
        param_json,
        type_S,
        n_points=n_points,
    )

    print("Géométrie OpenWind native =", type_S in ("const", "exp"))
    print("Nombre de segments géométriques =", len(bore))

    instrument = InstrumentGeometry(bore)

    phy = InstrumentPhysics(
        instrument,
        temperature=20,
        player=player,
        losses=False,
    )

    discr_params = {"order": order}

    if l_ele is not None:
        discr_params["l_ele"] = l_ele

    solver = TemporalSolver(
        phy,
        theta_scheme_parameter=theta,
        **discr_params,
    )
    print(solver)
    solver.discretization_infos()

    print("\n=== DIAGNOSTIC MESH ===")


    nH1_total = 0
    nL2_total = 0
    n_elements_total = 0

    for pipe in solver.t_pipes.data.values():
        nH1_total += pipe.mesh.get_nH1()
        nL2_total += pipe.mesh.get_nL2()
        n_elements_total += len(pipe.mesh.get_orders())

    print("total nH1 =", nH1_total)
    print("total nL2 =", nL2_total)
    print("total elements =", n_elements_total)
    print("elements per pipe =", solver.get_elements_mesh()[:10], "...")

    L_total = float(geom["tube"]["L_tube"] + geom["bell"]["L_bell"])

    try:
        solver.discretization_infos()
    except Exception as e:
        print("Impossible d'afficher discretization_infos :", e)

    rec = RecordingDevice()

    pipe_keys = list(solver.t_pipes.data.keys())
    pipe_values = list(solver.t_pipes.data.values())

    print("\n==============================")
    print("OPENWIND MESH")
    print("==============================")
    print("theta =", theta)
    print("Nombre de pipes =", len(pipe_values))
    print("Premier pipe    =", pipe_keys[0])
    print("Dernier pipe    =", pipe_keys[-1])
    print("Connecteurs finaux =", list(solver.t_connectors.data.keys())[-5:])

    reed_conn = solver.t_connectors.data["entrance_reed1dof_scaled_source"]
    reed_model = reed_conn.reed1dof

    print("\n==============================")
    print("OPENWIND REED DAMPING CHECK")
    print("==============================")

    print("Attributs disponibles dans reed_model :")
    for name in sorted(vars(reed_model).keys()):
        print(" ", name, "=", getattr(reed_model, name))

    def get_ow_param(p):
        for attr in ["get_value", "get", "value"]:
            if hasattr(p, attr):
                v = getattr(p, attr)
                return float(v(0.0) if callable(v) else v)
        raise AttributeError(f"Impossible de lire {p}")

    wr_ow = get_ow_param(reed_model.pulsation)
    Qr_ow = get_ow_param(reed_model.qfactor)
    g_ow = wr_ow / Qr_ow

    print("\nValeurs reed OpenWind :")
    print("wr_ow =", wr_ow)
    print("Qr_ow =", Qr_ow)
    print("g_ow = wr_ow / Qr_ow =", g_ow)
    print("wr_ow / g_ow =", wr_ow / g_ow)

    S_in = np.pi * R_tube**2
    c_ow = reed_conn.Zc * S_in / reed_conn.rho

    print("\n==============================")
    print("OPENWIND REED CONNECTOR")
    print("==============================")
    print("rho =", reed_conn.rho)
    print("Zc  =", reed_conn.Zc)
    print("S_in =", S_in)
    print("c reconstructed =", c_ow)
    

    ow_rad_params = {
        "alpha": None,
        "beta": None,
        "Zplus": None,
        "n_elements_total": int(n_elements_total),
        "h_eff": L_total / n_elements_total,
        "dt": float(solver.get_dt()),
    }

    if "bell_radiation" in solver.t_connectors.data:
        rad_conn = solver.t_connectors.data["bell_radiation"]

        print("\n==============================")
        print("OPENWIND RADIATION CONNECTOR")
        print("==============================")

        for name in ["alpha", "beta", "Zplus", "_zeta", "_opening_factor"]:
            if hasattr(rad_conn, name):
                print(name, "=", getattr(rad_conn, name))

        if hasattr(rad_conn, "_rad_model"):
            print("_rad_model =", rad_conn._rad_model)

        ow_rad_params = {
            "alpha": float(rad_conn.alpha) if hasattr(rad_conn, "alpha") else None,
            "beta": float(rad_conn.beta) if hasattr(rad_conn, "beta") else None,
            "Zplus": float(rad_conn.Zplus) if hasattr(rad_conn, "Zplus") else None,
            "n_elements_total": int(n_elements_total),
            "h_eff": L_total / n_elements_total,
            "dt": float(solver.get_dt()),
        }

    t_list = []
    p_left_pipe = []
    p_right_bore0 = []
    p_right_last_pipe = []
    p_before_right_last_pipe = []

    def callback(solver):
        rec.callback(solver)

        pipes = list(solver.t_pipes.data.values())

        pipe0 = pipes[0]
        pipe_last = pipes[-1]

        P0 = pipe0.PV[0]
        Plast = pipe_last.PV[0]

        p_left_pipe.append(float(P0[0]))
        p_right_bore0.append(float(P0[-1]))
        p_right_last_pipe.append(float(Plast[-1]))

        if len(Plast) >= 2:
            p_before_right_last_pipe.append(float(Plast[-2]))
        else:
            p_before_right_last_pipe.append(float(Plast[-1]))

        t_list.append(float(solver.get_current_time()))

    solver.run_simulation(float(T_max), callback=callback)

    t_ow = np.asarray(t_list)

    p_left_pipe = np.asarray(p_left_pipe)
    p_right_bore0 = np.asarray(p_right_bore0)
    p_right_last_pipe = np.asarray(p_right_last_pipe)
    p_before_right_last_pipe = np.asarray(p_before_right_last_pipe)

    if "entrance_reed1dof_scaled_source_pressure" in rec.values:
        p_left_rec = np.asarray(
            rec.values["entrance_reed1dof_scaled_source_pressure"]
        )
    else:
        print("Attention : entrance_reed1dof_scaled_source_pressure absent.")
        p_left_rec = p_left_pipe

    if "entrance_reed1dof_scaled_source_y" in rec.values:
        y_ow = np.asarray(
            rec.values["entrance_reed1dof_scaled_source_y"]
        )
    else:
        print("Attention : déplacement d'anche OpenWind absent.")
        y_ow = np.zeros_like(t_ow)

    if "bell_radiation_pressure" in rec.values:
        t_ow = np.asarray(rec.ts)
        p_right_rad = np.asarray(rec.values["bell_radiation_pressure"])
    else:
        print("Attention : bell_radiation_pressure absent.")
        p_right_rad = p_right_last_pipe

    gamma_ow = np.asarray([gamma_time(t) for t in t_ow])

    print("\n==============================")
    print("OPENWIND PRESSURE DIAGNOSTICS")
    print("==============================")

    def amp(name, arr):
        print(
            f"{name:30s} "
            f"min={np.min(arr): .4e} "
            f"max={np.max(arr): .4e} "
            f"maxabs={np.max(np.abs(arr)): .4e}"
        )

    amp("p_left_pipe", p_left_pipe)
    amp("p_left_rec", p_left_rec)
    amp("p_right_bore0", p_right_bore0)
    amp("p_right_last_pipe", p_right_last_pipe)
    amp("p_right_rad", p_right_rad)

    if np.max(np.abs(p_right_rad)) > 0:
        print("\nRapports utiles :")
        print(
            "maxabs(p_right_bore0) / maxabs(p_right_rad) =",
            np.max(np.abs(p_right_bore0))
            / (np.max(np.abs(p_right_rad)) + 1e-12),
        )
        print(
            "maxabs(p_right_last_pipe) / maxabs(p_right_rad) =",
            np.max(np.abs(p_right_last_pipe))
            / (np.max(np.abs(p_right_rad)) + 1e-12),
        )

    print("\n=== OpenWind recordings ===")
    for k, v in rec.values.items():
        arr = np.asarray(v)
        print(
            k,
            "shape=", arr.shape,
            "min=", np.min(arr),
            "max=", np.max(arr),
        )

    n = min(
        len(t_ow),
        len(p_left_rec),
        len(p_right_rad),
        len(y_ow),
        len(gamma_ow),
    )

    n_cmp = min(
        len(p_right_last_pipe),
        len(p_before_right_last_pipe),
    )

    if n_cmp > 0 and np.linalg.norm(p_right_last_pipe[:n_cmp]) > 0:
        err_last_vs_before = np.linalg.norm(
            p_right_last_pipe[:n_cmp] - p_before_right_last_pipe[:n_cmp]
        ) / (np.linalg.norm(p_right_last_pipe[:n_cmp]) + 1e-12)

        print("len p_right_last_pipe =", len(p_right_last_pipe))
        print("len p_before_right_last_pipe =", len(p_before_right_last_pipe))
        print("Erreur relative Plast[-1] vs Plast[-2] =", err_last_vs_before)

    ow_rad_params["dt"] = float(solver.get_dt())

    return (
        t_ow[:n],
        p_left_rec[:n],
        p_right_rad[:n],
        y_ow[:n],
        gamma_ow[:n],
        ow_rad_params,
    )
