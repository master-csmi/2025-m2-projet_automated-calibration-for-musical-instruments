from src.utils.util_func import ReedOpening
from src.physics.S_profiles import SProfile
from src.utils.diff import PhysicalData
import jax.numpy as jnp

def build_physical_data(params, type_S):

    left = params["left_bc_params"]
    mouth = left["mouth_pressure_params"]
    right = params["right_bc_params"]
    train = params["trainable"]
    instrument = params["instrument_geometry"]

    

    return PhysicalData(
        eps_data   = (left["epsilon"], train["epsilon"]),
        beta_data  = (right["beta"], train["beta"]),
        alpha_data = (right["alpha"], train["alpha"]),
        eta_data   = (left["zeta"], train["zeta"]),
        kappa_data = (left["kappa"], train["kappa"]),
        Zt_data    = (right["Zt"], False),
        wr_data    = (2*jnp.pi*left["f_r"], train["f_r"]),
        gamma_data = (mouth["gamma_final"], train["gamma_final"]),
        t_attack_data = (mouth["t_attack"], train["t_attack"]),
        t_delay_data = (mouth["t_delay"], train["t_delay"]),
        L_tube_data = (instrument["tube"]["L_tube"], train["L_tube"]),
        R_tube_data = (instrument["tube"]["R_tube"], train["R_tube"]),
        L_bell_data = (instrument["bell"]["L_bell"], train["L_bell"]),
        k_bell_data = (instrument["bell"]["k_bell"], train["k_bell"]),
        Qr_data    = (left["Qr"], train["Qr"]),
        sharpness_data = (mouth["sharpness"], train["sharpness"]),

        l = ReedOpening(a=1.0),
        section=SProfile(type_S=type_S, L_tube=instrument["tube"]["L_tube"], 
                         R_tube=instrument["tube"]["R_tube"], L_bell=instrument["bell"]["L_bell"], 
                        k_bell=instrument["bell"]["k_bell"])
    )