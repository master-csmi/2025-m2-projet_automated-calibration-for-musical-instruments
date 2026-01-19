# Equation

## Equation type

$$
W(x)\, \frac{\partial u}{\partial t} + \frac{\partial F(u)}{\partial x} = 0,
\quad \text{avec } u = (p, v).
$$

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

On introduit ensuite une fonction test vectorielle $\phi=(\tilde p, \tilde v)^T$ pour construire la formulation faible :

$$
\int_{I_i} W(x)\, \partial_t u \cdot \phi \, dx - \int_{I_i} F(u) \cdot \partial_x \phi\, dx + 
[F(u) \cdot \phi]_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}} = 0.
$$


Let's take the same base as test functions. 



Ceci constitue la **formulation de Galerkin semi-discrète** (en espace) pour une loi de conservation scalaire (ou vectorielle) de type :


## Linear system expression

Let's compute $u_h(x,t)=\sum_{j=1}^N U_j(t)\phi_j(x)$ with $\phi$ being P1 functions
$$
\begin{aligned}
 \int_{I_i}W(x)\sum_{j=1}^N \dot{U}_j(t)\phi_j(x)\phi_i(x)dx \ - &\int_{I_i} F(u_h).\partial_x\phi_i(x)dx + [F(u).\phi_i]_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}}=0 \\
 \sum_{j=1}^N \dot{U}_j(t)\int_{I_i}W(x)\phi_j(x)\phi_i(x)dx &= \int_{I_i} F(u_h).\partial_x\phi_i(x)dx - [F(u).\phi_i]_{x_{i-\frac{1}{2}}}^{x_{i+\frac{1}{2}}} \\
M\dot U &= R(U) \\
\dot U &= M^{-1}R(U)
\end{aligned}
$$



Avec la matrice de masse $ \{M_{ij}\}_{i,j = 1,...,N}=\sum_{q=1}^{N_q}\phi_i(x_q)\phi_j(x_q)$

## Main Objective

A partir d'une eq, evolution de la pression au  travers de l'instrument à hanche, 
Param non identifiable, fonction non lin l(y) choisit empiriquement,
Probleme inverse, connaissant des solutions (partielles, ponctuellement, ) données -> retrouver les coefficients ou l(y) à partir des mesures. (probleme inverse -> estimation de la fonction)

Comment transformer les données 

- Synthétiser les données
- différentiation automatique pas besoin de calculer le problème adjoint. (même principe que le calcul de l'adjoint)
Param (S=1)

Diagonaliser l'équation des ondes 
p-u et p+u

d_tu+cd_xu=0
d_tp+cdx_u=0

source = 0 

u+p et u-p

### Analitical solving of a simplified model

Let's consider the following transport equation that represent 
$$
d_tu + cAd_xu=0  \ \ \ (1)
$$
with $u=(p,v)^T$ and $p(x,0)=p_0$, $v(x,0)=v_0$:

$$
A = 
\begin{pmatrix}
0 & 1\\[4pt]
1 & 0
\end{pmatrix}
$$

we obtain the following system 1-D hyperbolic system for of the wave equation : 
$$
\begin{aligned}
p_t+cv_x=0 \\
v_t+cp_x=0
\end{aligned}
$$

That can be re-written :
$$
\begin{aligned}
(p+v)_t+c(p+v)_x=0 \\
(p-v)_t-c(p-v)_x=0
\end{aligned}
$$

The solutions of this system are the transport equation solutions : 


$$
\begin{aligned}
(p+v)(x,t)=(p_0+v_0)(x-ct) \\
(p-v)(x,t)=(p_0-v_0)(x+ct)
\end{aligned}
$$

and finally, we can isolate the variables $p$ and $v$:

$$
\begin{aligned}
p(x,t)=\frac{p_0(x-ct)+p_0(x+ct)}{2}+\frac{v_0(x-ct)-v_0(x+ct)}{2} \\
v(x,t)=\frac{p_0(x-ct)-p_0(x+ct)}{2}+\frac{v_0(x-ct)+v_0(x+ct)}{2}
\end{aligned}
$$

Here the two analytical solution of our system that we will try to approach by solving the equation (1)

$$
\begin{aligned}
u^{n+1} &=  u^n(x) + \Delta t \partial _xF(u^n(x)) \\
\int_{\Omega_k} u^{n+1} \phi _i(x) &= \int_{\Omega_k} u^n\phi _i(x) + \Delta t \int_{\Omega_k}\partial _xF(u^n(x))\phi _i(x) \\
\sum_j \alpha_{k,j}\int_{\Omega_k} \phi _i \phi _j &= \int_{\Omega_k} u^n(x)\phi _i(x) - \Delta t \int_{\Omega_k}\partial _xF(u^n(x))\phi _i(x) + [F(u^n)\phi_i]^{x_{j+1/2}}_{x_{j-1/2}} \\
M\alpha_k &= \sum^{N_q}_{q=1} u_n(x_q)\phi(x_q) - \Delta T \sum^{N_q}_{q=1}F(u_n(x_q))\partial \phi(x_q)+ [F(u)\phi]^{x_{j+1/2}}_{x_{j-1/2}} \\
\alpha_ k&= M^{-1} (\sum^{N_q}_{q=1} u_n(x_q)\phi(x_q) - \Delta T \sum^{N_q}_{q=1}F(u_n(x_q))\partial \phi(x_q)+ [F(u)\phi]^{x_{j+1/2}}_{x_{j-1/2}})
\end{aligned}
$$


## Ref

Gary Cohen Sebasiten Pernet, 
