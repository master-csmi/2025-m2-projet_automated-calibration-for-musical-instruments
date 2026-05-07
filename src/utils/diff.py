import jax
from jax import numpy as jnp
from typing import Callable, Union
import equinox as eqx 

class PhysicalData:
    eps: float
    bool_eps: bool
    zeta: float
    bool_eta: bool
    beta: float
    bool_beta: bool
    alpha: float
    bool_alpha: bool
    kappa: float
    bool_kappa: bool
    Zt: float
    bool_Zt: bool
    f_r: float
    bool_wr: bool
    gamma_final: float
    bool_gamma_final: bool
    t_attack: float
    bool_t_attack: bool
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

    def __init__(self, eps_data, beta_data, alpha_data, zeta_data,kappa_data,Zt_data,fr_data,gamma_data,t_attack_data,L_tube_data,R_tube_data,L_bell_data,k_bell_data,Qr_data,l,section):
        self.eps = eps_data[0]
        self.bool_eps = eps_data[1]
        self.beta = beta_data[0]
        self.bool_beta = beta_data[1]
        self.alpha = alpha_data[0]
        self.bool_alpha = alpha_data[1]
        self.zeta = zeta_data[0]
        self.bool_zeta = zeta_data[1]
        self.kappa = kappa_data[0]
        self.bool_kappa = kappa_data[1]
        self.Zt = Zt_data[0]
        self.bool_Zt = Zt_data[1]
        self.fr = fr_data[0]
        self.bool_fr = fr_data[1]
        self.gamma_final = gamma_data[0]
        self.bool_gamma_final = gamma_data[1]
        self.t_attack = t_attack_data[0]
        self.bool_t_attack = t_attack_data[1]
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
        if self.bool_zeta:
            children.append(self.zeta)
        if self.bool_kappa:
            children.append(self.kappa)
        if self.bool_Zt:
            children.append(self.Zt)
        if self.bool_fr:
            children.append(self.fr)
        if self.bool_gamma_final:
            children.append(self.gamma_final)
        if self.bool_t_attack:
            children.append(self.t_attack)
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
        aux_data = (
            l_in_children,       self.l       if not l_in_children       else None,
            section_in_children, self.section if not section_in_children else None,
            self.bool_eps,   self.eps   if not self.bool_eps   else None,
            self.bool_beta,  self.beta  if not self.bool_beta  else None,
            self.bool_alpha, self.alpha if not self.bool_alpha else None,
            self.bool_zeta,   self.zeta   if not self.bool_zeta   else None,
            self.bool_kappa, self.kappa if not self.bool_kappa else None,
            self.bool_Zt,    self.Zt    if not self.bool_Zt    else None,
            self.bool_fr,    self.fr    if not self.bool_fr    else None,
            self.bool_gamma_final, self.gamma_final if not self.bool_gamma_final else None,
            self.bool_t_attack, self.t_attack if not self.bool_t_attack else None,
            self.bool_L_tube, self.L_tube if not self.bool_L_tube else None,
            self.bool_R_tube, self.R_tube if not self.bool_R_tube else None,
            self.bool_L_bell, self.L_bell if not self.bool_L_bell else None,
            self.bool_k_bell, self.k_bell if not self.bool_k_bell else None,    
            self.bool_Qr,    self.Qr    if not self.bool_Qr    else None
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
            bool_zeta,   zeta_static,
            bool_kappa, kappa_static,
            bool_Zt,    Zt_static,
            bool_fr,    fr_static,
            bool_gamma_final, gamma_final_static,
            bool_t_attack, t_attack_static,
            bool_L_tube, L_tube_static,
            bool_R_tube, R_tube_static,
            bool_L_bell, L_bell_static,
            bool_k_bell, k_bell_static,
            bool_Qr,    Qr_static        
        ) = aux_data

        l       = next(it) if l_in_children       else l_static
        section = next(it) if section_in_children else section_static
        eps     = next(it) if bool_eps   else eps_static
        beta    = next(it) if bool_beta  else beta_static
        alpha   = next(it) if bool_alpha else alpha_static
        zeta    = next(it) if bool_zeta  else zeta_static
        kappa   = next(it) if bool_kappa else kappa_static
        Zt      = next(it) if bool_Zt    else Zt_static
        fr      = next(it) if bool_fr    else fr_static
        gamma_final   = next(it) if bool_gamma_final else gamma_final_static
        t_attack = next(it) if bool_t_attack else t_attack_static
        L_tube  = next(it) if bool_L_tube else L_tube_static
        R_tube  = next(it) if bool_R_tube else R_tube_static
        L_bell  = next(it) if bool_L_bell else L_bell_static
        k_bell  = next(it) if bool_k_bell else k_bell_static
        Qr      = next(it) if bool_Qr    else Qr_static
        return cls(
            (eps, bool_eps), (beta, bool_beta), (alpha, bool_alpha), (zeta, bool_zeta), (kappa, bool_kappa),
            (Zt, bool_Zt), (fr, bool_fr), (gamma_final, bool_gamma_final),(t_attack, bool_t_attack), 
            (L_tube, bool_L_tube), (R_tube, bool_R_tube), 
            (L_bell, bool_L_bell), (k_bell, bool_k_bell), (Qr, bool_Qr),
            l, section
        )
jax.tree_util.register_pytree_node_class(PhysicalData)