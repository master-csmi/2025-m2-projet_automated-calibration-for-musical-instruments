# Equation

## Formulation semi-discrète de Galerkin en volumes finis

Soit un maillage uniforme (ou non uniforme) de $\Omega$ avec cellules  
$$
I_i = [x_{i-\frac{1}{2}},\, x_{i+\frac{1}{2}}],
$$
de longueur $\Delta x_i$.

On intègre l’équation sur $I_i$ :
$$
\int_{I_i} W(x)\, \partial_t u\, dx + \int_{I_i} \partial_x F(u)\, dx = 0.
$$

où $u=(p,v)^T$

On introduit ensuite une fonction test vectorielle $\phi=(\tilde p, \tilde v)^T$ 
\(\phi = (\tilde{p}, \tilde{v})\) pour construire la formulation faible :

$$
\int_{I_i} W(x)\, \partial_t u \cdot \phi \, dx - \int_{I_i} F(u) \cdot \partial_x \phi\, dx + 
[F(u) \cdot \phi]_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}} = 0.
$$

Ceci constitue la **formulation de Galerkin semi-discrète** (en espace) pour une loi de conservation scalaire (ou vectorielle) de type :
$$
W(x)\, \frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0,
\quad \text{avec } u = (p, v).
$$

## Définition du Flux

$$
\begin{aligned}
u^{*} &= W(x) u^n(x) + \Delta t \partial _xF(u^n(x)) \\
\int_{\Omega_k} u^* \phi _i(x) &= \int_{\Omega_k} W(x) u^n\phi _i(x) + \Delta t \int_{\Omega_k}\partial _xF(u^n(x))\phi _i(x) \\
\sum_j \alpha_{k,j}\int_{\Omega_k} \phi _i \phi _j &= \int_{\Omega_k} W(x)u^n(x)\phi _i(x) - \Delta t \int_{\Omega_k}\partial _xF(u^n(x))\phi _i(x) + [F(u^n)\phi_i]^{x_{j+1/2}}_{x_{j-1/2}} \\
M\alpha_k &= \sum^{N_q}_{q=1} W(x_q)u_n(x_q)\phi(x_q) - \Delta T \sum^{N_q}_{q=1}F(u_n(x_q))\partial \phi(x_q)+ [F(u)\phi]^{x_{j+1/2}}_{x_{j-1/2}}
\end{aligned}
$$

Avec la matrice de masse $ \{M_{ij}\}_{i,j = 1,...,N}=\sum_{q=1}^{N_q}\phi_i(x_q)\phi_j(x_q)$
