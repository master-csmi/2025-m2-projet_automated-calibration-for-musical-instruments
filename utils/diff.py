import jax
from jax import numpy as jnp
from typing import Callable, Union
import equinox as eqx 

class PhysicalData:
    eps: float
    bool_eps: bool
    eta: float
    bool_eta: bool
    kappa: float
    bool_kappa: bool
    Zt: float
    bool_Zt: bool
    wr: float
    bool_wr: bool
    gamma: float
    bool_gamma: bool
    Qr: float
    bool_Qr: bool
    l: Union[Callable, eqx.Module]
    section: Union[Callable, eqx.Module]

    def __init__(self, eps_data,eta_data,kappa_data,Zt_data,wr_data,gamma_data,Qr_data,l,section):
        self.eps = eps_data[0]
        self.bool_eps = eps_data[1]
        self.eta = eta_data[0]
        self.bool_eta = eta_data[1]
        self.kappa = kappa_data[0]
        self.bool_kappa = kappa_data[1]
        self.Zt = Zt_data[0]
        self.bool_Zt = Zt_data[1]
        self.wr = wr_data[0]
        self.bool_wr = wr_data[1]
        self.gamma = gamma_data[0]
        self.bool_gamma = gamma_data[1]
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
        if self.bool_eta:
            children.append(self.eta)
        if self.bool_kappa:
            children.append(self.kappa)
        if self.bool_Zt:
            children.append(self.Zt)
        if self.bool_wr:
            children.append(self.wr)
        if self.bool_gamma:
            children.append(self.gamma)
        if self.bool_Qr:
            children.append(self.Qr)

        aux_data = (
            l_in_children,       self.l       if not l_in_children       else None,
            section_in_children, self.section if not section_in_children else None,
            self.bool_eps,   self.eps   if not self.bool_eps   else None,
            self.bool_eta,   self.eta   if not self.bool_eta   else None,
            self.bool_kappa, self.kappa if not self.bool_kappa else None,
            self.bool_Zt,    self.Zt    if not self.bool_Zt    else None,
            self.bool_wr,    self.wr    if not self.bool_wr    else None,
            self.bool_gamma, self.gamma if not self.bool_gamma else None,
            self.bool_Qr,    self.Qr    if not self.bool_Qr    else None,
        )
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        it = iter(children)
        (
            l_in_children,       l_static,
            section_in_children, section_static,
            bool_eps,   eps_static,
            bool_eta,   eta_static,
            bool_kappa, kappa_static,
            bool_Zt,    Zt_static,
            bool_wr,    wr_static,
            bool_gamma, gamma_static,
            bool_Qr,    Qr_static,
        ) = aux_data

        l       = next(it) if l_in_children       else l_static
        section = next(it) if section_in_children else section_static
        eps     = next(it) if bool_eps   else eps_static
        eta     = next(it) if bool_eta   else eta_static
        kappa   = next(it) if bool_kappa else kappa_static
        Zt      = next(it) if bool_Zt    else Zt_static
        wr      = next(it) if bool_wr    else wr_static
        gamma   = next(it) if bool_gamma else gamma_static
        Qr      = next(it) if bool_Qr    else Qr_static

        return cls(
            (eps, bool_eps), (eta, bool_eta), (kappa, bool_kappa),
            (Zt, bool_Zt), (wr, bool_wr), (gamma, bool_gamma), (Qr, bool_Qr),
            l, section
        )
jax.tree_util.register_pytree_node_class(PhysicalData)