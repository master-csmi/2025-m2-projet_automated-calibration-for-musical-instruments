import jax
from jax import numpy as jnp
from typing import Callable, Union
import equinox as eqx 

class PhysicalData:
    eps: float
    bool_eps: bool
    eta: float
    bool_eta: bool
    beta: float
    bool_beta: bool
    alpha: float
    bool_alpha: bool
    kappa: float
    bool_kappa: bool
    Zt: float
    bool_Zt: bool
    wr: float
    bool_wr: bool
    gamma_final: float
    bool_gamma_final: bool
    t_attack: float
    bool_t_attack: bool
    sharpness: float
    sharpness_bool: bool
    t_delay: float
    bool_t_delay: bool
    L_tube: float
    bool_L_tube: bool
    R_tube: float
    bool_R_tube: bool
    L_bell: float
    bool_L_bell: bool
    k_bell: float
    bool_k_bell: bool
    Qr: float
    bool_Qr: bool
    l: Union[Callable, eqx.Module]
    section: Union[Callable, eqx.Module]

    def __init__(self, eps_data, beta_data, alpha_data, eta_data,kappa_data,Zt_data,wr_data,gamma_data,t_attack_data,t_delay_data,L_tube_data,R_tube_data,L_bell_data,k_bell_data,Qr_data,sharpness_data,l,section):
        self.eps = eps_data[0]
        self.bool_eps = eps_data[1]
        self.beta = beta_data[0]
        self.bool_beta = beta_data[1]
        self.alpha = alpha_data[0]
        self.bool_alpha = alpha_data[1]
        self.eta = eta_data[0]
        self.bool_eta = eta_data[1]
        self.kappa = kappa_data[0]
        self.bool_kappa = kappa_data[1]
        self.Zt = Zt_data[0]
        self.bool_Zt = Zt_data[1]
        self.wr = wr_data[0]
        self.bool_wr = wr_data[1]
        self.gamma_final = gamma_data[0]
        self.bool_gamma_final = gamma_data[1]
        self.t_attack = t_attack_data[0]
        self.bool_t_attack = t_attack_data[1]
        self.t_delay = t_delay_data[0]
        self.bool_t_delay = t_delay_data[1]
        self.L_tube = L_tube_data[0]
        self.bool_L_tube = L_tube_data[1]
        self.R_tube = R_tube_data[0]
        self.bool_R_tube = R_tube_data[1]
        self.L_bell = L_bell_data[0]
        self.bool_L_bell = L_bell_data[1]
        self.k_bell = k_bell_data[0]
        self.bool_k_bell = k_bell_data[1]
        self.Qr = Qr_data[0]
        self.bool_Qr = Qr_data[1]
        self.sharpness = sharpness_data[0]
        self.sharpness_bool = sharpness_data[1]
        self.l = l
        self.section = section 

    def tree_flatten(self):
        children = []

        l_in_children = isinstance(self.l, eqx.Module)
        section_in_children = isinstance(self.section, eqx.Module)

        if l_in_children:
            children.append(self.l)
        if section_in_children:
            children.append(self.section)
        if self.bool_eps:
            children.append(self.eps)
        if self.bool_beta:
            children.append(self.beta)
        if self.bool_alpha:
            children.append(self.alpha)
        if self.bool_eta:
            children.append(self.eta)
        if self.bool_kappa:
            children.append(self.kappa)
        if self.bool_Zt:
            children.append(self.Zt)
        if self.bool_wr:
            children.append(self.wr)
        if self.bool_gamma_final:
            children.append(self.gamma_final)
        if self.bool_t_attack:
            children.append(self.t_attack)
        if self.bool_t_delay:
            children.append(self.t_delay)
        if self.bool_L_tube:
            children.append(self.L_tube)
        if self.bool_R_tube:
            children.append(self.R_tube)
        if self.bool_L_bell:
            children.append(self.L_bell)
        if self.bool_k_bell:
            children.append(self.k_bell)
        if self.bool_Qr:
            children.append(self.Qr)
        if self.sharpness_bool:
            children.append(self.sharpness)
        aux_data = (
            l_in_children,       self.l       if not l_in_children       else None,
            section_in_children, self.section if not section_in_children else None,
            self.bool_eps,   self.eps   if not self.bool_eps   else None,
            self.bool_beta,  self.beta  if not self.bool_beta  else None,
            self.bool_alpha, self.alpha if not self.bool_alpha else None,
            self.bool_eta,   self.eta   if not self.bool_eta   else None,
            self.bool_kappa, self.kappa if not self.bool_kappa else None,
            self.bool_Zt,    self.Zt    if not self.bool_Zt    else None,
            self.bool_wr,    self.wr    if not self.bool_wr    else None,
            self.bool_gamma_final, self.gamma_final if not self.bool_gamma_final else None,
            self.bool_t_attack, self.t_attack if not self.bool_t_attack else None,
            self.bool_t_delay, self.t_delay if not self.bool_t_delay else None,
            self.bool_L_tube, self.L_tube if not self.bool_L_tube else None,
            self.bool_R_tube, self.R_tube if not self.bool_R_tube else None,
            self.bool_L_bell, self.L_bell if not self.bool_L_bell else None,
            self.bool_k_bell, self.k_bell if not self.bool_k_bell else None,    
            self.bool_Qr,    self.Qr    if not self.bool_Qr    else None,
            self.sharpness_bool, self.sharpness if not self.sharpness_bool else None
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        it = iter(children)
        (
            l_in_children,       l_static,
            section_in_children, section_static,
            bool_eps,   eps_static,
            bool_beta,  beta_static,
            bool_alpha, alpha_static,
            bool_eta,   eta_static,
            bool_kappa, kappa_static,
            bool_Zt,    Zt_static,
            bool_wr,    wr_static,
            bool_gamma_final, gamma_final_static,
            bool_t_attack, t_attack_static,
            bool_t_delay, t_delay_static,
            bool_L_tube, L_tube_static,
            bool_R_tube, R_tube_static,
            bool_L_bell, L_bell_static,
            bool_k_bell, k_bell_static,
            bool_Qr,    Qr_static,
            sharpness_bool, sharpness_static
        ) = aux_data

        l       = next(it) if l_in_children       else l_static
        section = next(it) if section_in_children else section_static
        eps     = next(it) if bool_eps   else eps_static
        beta    = next(it) if bool_beta  else beta_static
        alpha   = next(it) if bool_alpha else alpha_static
        eta     = next(it) if bool_eta   else eta_static
        kappa   = next(it) if bool_kappa else kappa_static
        Zt      = next(it) if bool_Zt    else Zt_static
        wr      = next(it) if bool_wr    else wr_static
        gamma_final   = next(it) if bool_gamma_final else gamma_final_static
        t_attack = next(it) if bool_t_attack else t_attack_static
        t_delay = next(it) if bool_t_delay else t_delay_static
        L_tube  = next(it) if bool_L_tube else L_tube_static
        R_tube  = next(it) if bool_R_tube else R_tube_static
        L_bell  = next(it) if bool_L_bell else L_bell_static
        k_bell  = next(it) if bool_k_bell else k_bell_static
        Qr      = next(it) if bool_Qr    else Qr_static
        sharpness = next(it) if sharpness_bool else sharpness_static    
        return cls(
            (eps, bool_eps), (beta, bool_beta), (alpha, bool_alpha), (eta, bool_eta), (kappa, bool_kappa),
            (Zt, bool_Zt), (wr, bool_wr), (gamma_final, bool_gamma_final),(t_attack, bool_t_attack), (t_delay, bool_t_delay),
            (L_tube, bool_L_tube), (R_tube, bool_R_tube), 
            (L_bell, bool_L_bell), (k_bell, bool_k_bell), (Qr, bool_Qr),(sharpness, sharpness_bool),
            l, section
        )
jax.tree_util.register_pytree_node_class(PhysicalData)