import numpy as np


def run_openwind_reference(param_json, T_max, type_S):
    from openwind import InstrumentGeometry, InstrumentPhysics, TemporalSolver, Player
    from openwind.temporal import RecordingDevice
    from openwind.technical.temporal_curves import constant_with_initial_ramp

    from utils.build_physical_data import build_physical_data
    from utils.util_func import make_openwind_radius_profile

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

    data = build_physical_data(param_json, type_S)
    x_geom, r_geom = make_openwind_radius_profile(data)

    bore = [
        [float(x), float(r)]
        for x, r in zip(x_geom, r_geom)
    ]

    instrument = InstrumentGeometry(bore)

    phy = InstrumentPhysics(
        instrument,
        temperature=20,
        player=player,
        losses=False,
    )

    solver = TemporalSolver(phy, theta_scheme_parameter=0.5)

    print(solver)
    solver.discretization_infos()
    rec = RecordingDevice()

    pipe_keys = list(solver.t_pipes.data.keys())
    pipe_values = list(solver.t_pipes.data.values())

    print("\n==============================")
    print("OPENWIND MESH")
    print("==============================")
    print("Nombre de pipes =", len(pipe_values))
    print("Premier pipe    =", pipe_keys[0])
    print("Dernier pipe    =", pipe_keys[-1])
    print("Connecteurs finaux =", list(solver.t_connectors.data.keys())[-5:])

    reed_conn = solver.t_connectors.data["entrance_reed1dof_scaled_source"]

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

    gamma_ow = np.asarray([gamma_time(t) for t in t_ow])

    if "entrance_reed1dof_scaled_source_pressure" in rec.values:
        p_left_rec = np.asarray(
            rec.values["entrance_reed1dof_scaled_source_pressure"]
        )
    else:
        print("Attention : entrance_reed1dof_scaled_source_pressure absent.")
        p_left_rec = p_left_pipe

    p_right_rad = p_right_last_pipe

    if "entrance_reed1dof_scaled_source_y" in rec.values:
        y_ow = np.asarray(
            rec.values["entrance_reed1dof_scaled_source_y"]
        )
    else:
        print("Attention : déplacement d'anche OpenWind absent.")
        y_ow = np.zeros_like(t_ow)

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

    n_cmp = min(len(p_right_last_pipe), len(p_before_right_last_pipe))

    err_last_vs_before = np.linalg.norm(
        p_right_last_pipe[:n_cmp] - p_before_right_last_pipe[:n_cmp]
    ) / (np.linalg.norm(p_right_last_pipe[:n_cmp]) + 1e-12)

    print("len p_right_last_pipe =", len(p_right_last_pipe))
    print("len p_before_right_last_pipe =", len(p_before_right_last_pipe))
    print("Erreur relative Plast[-1] vs Plast[-2] =", err_last_vs_before)

    return (
        t_ow[:n],
        p_left_rec[:n],
        p_right_rad[:n],
        y_ow[:n],
        gamma_ow[:n],
        ow_rad_params,
    )