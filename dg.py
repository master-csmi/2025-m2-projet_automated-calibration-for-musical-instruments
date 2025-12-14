# DG P1 for 1D linear wave system (p,v) with p0 = Gaussian, v0 = 0
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time 
import csv
import os
import pandas as pd
import numpy as np
import argparse
from jax import lax

# -------------------------
# mesh / basis 
# -------------------------
def create_uniform_nodes(N_intervals):
    return jnp.linspace(0.0, 1.0, N_intervals + 1)

def cell_edges_from_nodes(x_nodes):
    return x_nodes[:-1], x_nodes[1:]

def phi_at(x, xL, xR):
    h = xR - xL
    xi = 2.0 * (x - xL) / h - 1.0
    phi0 = 0.5 * (1.0 - xi)
    phi1 = 0.5 * (1.0 + xi)
    return jnp.stack([phi0, phi1])

vphi_at = jax.vmap(phi_at, in_axes=(0, None, None))

# analytic local mass matrix for P1
def local_mass_inv(h):
    M = (h / 6.0) * jnp.array([[2.0, 1.0], [1.0, 2.0]])
    return jnp.linalg.inv(M)


# -------------------------
# flux for system (linear)
# -------------------------
def linear_system_flux(A):
    def Flux(U):
        # U shape (...,2)
        return U @ A.T 
    return Flux

def rusanov_flux(U_L, U_R, A,smax):
   
    F_L = (U_L[None, :] @ A.T)[0]
    F_R = (U_R[None, :] @ A.T)[0]

    return 0.5*(F_L + F_R) - 0.5*smax*(U_R - U_L)

# -------------------------
# local volume term for system
# -------------------------
def local_volume_system(u_cell, xL, xR, A, nq=24):
    h = xR - xL
    xq = jnp.linspace(xL, xR, nq)
    w  = jnp.ones(nq) * (h/(nq-1))
    w  = w.at[0].set(h/(2*(nq-1)))
    w  = w.at[-1].set(h/(2*(nq-1)))

    # reconstruction U(x)
    phi_q = vphi_at(xq, xL, xR)       
    p_q = phi_q @ u_cell[0]           
    v_q = phi_q @ u_cell[1]           
    Uq = jnp.stack([p_q, v_q], axis=1)

    Fq = Uq @ A.T                    

    # exact derivative of basis functions
    dphi0 = -1.0 / h
    dphi1 =  1.0 / h

    # Integration of flux * dphi
    V0 = jnp.sum(w[:,None] * Fq * dphi0, axis=0)
    V1 = jnp.sum(w[:,None] * Fq * dphi1, axis=0)

    return jnp.stack([V0, V1], axis=1)   


v_local_volume_system = jax.vmap(local_volume_system, in_axes=(0, 0, 0, None, None))

# -------------------------
# surface term for system
# S = f_right * phiR - f_left * phiL  where phiR=[0,1], phiL=[1,0]
# -------------------------

# implement surface term with that layout

def surface_term_system(u_cells, j, A, smax):
    N = u_cells.shape[0]
    # left interface between j-1 and j
    UL_left = u_cells[(j-1) % N, :, 1]  
    UR_left = u_cells[j, :, 0]          
    f_left = rusanov_flux(UL_left, UR_left, A, smax)

    # right interface between j and j+1
    UL_right = u_cells[j, :, 1]
    UR_right = u_cells[(j+1) % N, :, 0]
    f_right = rusanov_flux(UL_right, UR_right, A, smax)

    S = jnp.zeros((2,2))
    S = S.at[:,1].set(f_right)
    S = S.at[:,0].set(-f_left)
    return S

v_surface_term_system = jax.vmap(surface_term_system, in_axes=(None, 0, None, None))

# -------------------------
# assembly RHS for all elements
# -------------------------
def dg_rhs_system(u_cells, x_nodes, A, smax, M_inv_all):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    N = u_cells.shape[0]
  
    V_all = jax.vmap(lambda Ue, xL, xR: local_volume_system(Ue, xL, xR, A, 24))(u_cells, xLs, xRs)

    S_all = jax.vmap(lambda j: surface_term_system(u_cells, j, A, smax))(jnp.arange(N))
     
    
    def element_rhs(e):
        Vi = V_all[e]  
        Si = S_all[e]
        
        rhs_comp0 = M_inv_all[e] @ (Vi[0] - Si[0])
        rhs_comp1 = M_inv_all[e] @ (Vi[1] - Si[1])
        return jnp.stack([rhs_comp0, rhs_comp1], axis=0)  
    RHS = jax.vmap(element_rhs)(jnp.arange(N))
    return RHS

# -------------------------
# RK2 step
# -------------------------
@jax.jit
def rk2_step_system(u_cells, x_nodes, A, smax, dt, M_inv_all):
    k1 = dg_rhs_system(u_cells, x_nodes, A, smax, M_inv_all)
    u_mid = u_cells + 0.5 * dt * k1
    k2 = dg_rhs_system(u_mid, x_nodes, A, smax, M_inv_all)
    return u_cells + dt * k2

# -------------------------
# Euler step
# -------------------------
@jax.jit
def euler_step_system(u_cells, x_nodes, A, smax, dt, M_inv_all):
    k1 = dg_rhs_system(u_cells, x_nodes, A, smax, M_inv_all)  # (N,2,2)
    return u_cells + dt * k1

# -------------------------
# reconstruction for plotting
# -------------------------
def reconstruct_system(u_cells, x_nodes, x_plot):
    xLs, xRs = cell_edges_from_nodes(x_nodes)
    h = xRs[0]-xLs[0]
    Ncells = u_cells.shape[0]
    idx = jnp.clip(jnp.floor(x_plot / h).astype(int), 0, Ncells-1)
    def eval_point(x, j):
        xL = xLs[j]; xR = xRs[j]
        ph = phi_at(x, xL, xR)
        # compute p and v
        p = jnp.dot(ph, u_cells[j,0])
        v = jnp.dot(ph, u_cells[j,1])
        return jnp.stack([p, v])
    UV = jax.vmap(eval_point)(x_plot, idx)  # (len(x_plot),2)
    return UV[:,0], UV[:,1]

# -------------------------
# analytic solution for initial p0 Gaussian and v0=0
# decomposition into w+ = p+v, w- = p-v
# -------------------------
def analytic_solution_pv(x_plot, p0_fun, c, t):
    # w+ = p0, w- = p0 at t=0
    w_plus = p0_fun((x_plot - c * t) % 1.0)
    w_minus = p0_fun((x_plot + c * t) % 1.0)
    p_exact = 0.5 * (w_plus + w_minus)
    v_exact = 0.5 * (w_plus - w_minus)
    return p_exact, v_exact, w_plus, w_minus


def time_integrate_euler(u0, x_nodes, A, smax, dt, nsteps, M_inv_all):

    def step(u, _):
        u_next = euler_step_system(u, x_nodes, A, smax, dt, M_inv_all)
        return u_next, None

    u_final, _ = lax.scan(step, u0, None, length=nsteps)
    return u_final

def time_integrate_rk2(u0, x_nodes, A, smax, dt, nsteps, M_inv_all):

    def step(u, _):
        u_next = rk2_step_system(u, x_nodes, A, smax, dt, M_inv_all)
        return u_next, None

    u_final, _ = lax.scan(step, u0, None, length=nsteps)
    return u_final


def parse_args():
    parser = argparse.ArgumentParser(description="DG P1 1D linear wave")
    parser.add_argument("--method", type=str, default="rk2",
                        choices=["euler", "rk2"],
                        help="Time integration method")
    parser.add_argument("--N", type=int, default=200,
                        help="Number of cells")
    parser.add_argument("--CFL", type=float, default=0.05,
                        help="CFL number")
    parser.add_argument("--tfinal", type=float, default=0.2,
                        help="Final time")
    return parser.parse_args()


#-------------------------
# Initial compactly supported bump function
#-------------------------
def intit_func(x, phi0=1.0):
    """
    Bump function:
        φ(x) = φ0/4 * exp( 1 - 1/(1 - (4x-2)**2) )  if |x-1/2| < 1/4
               0                                  otherwise
    """
    mask = jnp.abs(x - 0.5) < 0.25
    val = (phi0 / 4.0) * jnp.exp(1.0 - 1.0 / (1.0 - (4.0*x - 2.0)**2))
    return jnp.where(mask, val, 0.0)

# -------------------------
# test: p0 gaussian, v0=0
# -------------------------
def main():


    args = parse_args()

    method = args.method
    N = args.N
    CFL = args.CFL
    t_final = args.tfinal

    # physical params
    c = 1.0
    A = jnp.array([[0.0, c],[c, 0.0]])
    smax = jnp.max(jnp.abs(jnp.linalg.eigvals(A)))
    print('smax',smax)
    CFL = 0.05
    # initial condition: p0 gaussian, v0 = 0

    def p0(x):
        #x0 = 0.5
        #sigma = 0.05
        #return jnp.exp(-0.5 * ((x - x0)/sigma)**2)
        return intit_func(x, phi0=1.0)

    
    def v0(x):
        return 0.0

    Ns = [100, 200, 400, 800]  # number of cells

    res_dir = 'Results'
    csv_file = os.path.join(res_dir, 'convergence_results.csv')
    file_exists = os.path.isfile(csv_file)
    if not file_exists:
        print(f"Creating new CSV file: {csv_file}")
    else : #rewrite it 
        os.remove(csv_file)

    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['N_cells', 'duration_sec', 'L2_error_p', 'L2_error_v', 'Linf_error_p', 'Linf_error_v'])

    p_erros = []
    v_errors = [] 
    durations = []
    for N in Ns:
        print(f"Running simulation with N={N} cells")
        # mesh

        x_nodes = create_uniform_nodes(N)
        xLs, xRs = cell_edges_from_nodes(x_nodes)
        hs = xRs - xLs
        M_inv_all = jax.vmap(local_mass_inv)(hs)
    
        # build u_cells array 
        u_cells = jnp.stack([jnp.stack([jnp.array([p0(xLs[i]), p0(xRs[i])]),
                                        jnp.array([v0(xLs[i]), v0(xRs[i])])]) for i in range(N)], axis=0)

        # time step (CFL)
        h = xRs[0] - xLs[0]
        print('h',h)
        dt = CFL * h / smax
        print('dt',dt)
        t_final = 0.2
        nsteps = int(jnp.ceil(t_final / dt))
        print('nsteps',nsteps)
        

        u = u_cells.copy()
        stat_time = time.time()
    
        if method == "euler":
            u = time_integrate_euler(u, x_nodes, A, smax, dt, nsteps, M_inv_all)
        else:
            u = time_integrate_rk2(u, x_nodes, A, smax, dt, nsteps, M_inv_all)
        end_time = time.time()

        duration = end_time - stat_time
    

        # reconstruct on dense grid
        x_plot = jnp.linspace(0.0, 1.0, 2000)
        p_rec, v_rec = reconstruct_system(u, x_nodes, x_plot)

        # analytic
        p_ex, v_ex, wplus_ex, wminus_ex = analytic_solution_pv(x_plot, p0, c, t_final)

        t_final_2 = 0.3
        u = u_cells.copy()
        nsteps = int(jnp.ceil(t_final_2 / dt))
        print('nsteps',nsteps)
        
        if method == "euler":
            u = time_integrate_euler(u, x_nodes, A, smax, dt, nsteps, M_inv_all)
        elif method == "rk2":
            u = time_integrate_rk2(u, x_nodes, A, smax, dt, nsteps, M_inv_all)
        
        
       
        p_rec2, v_rec2 = reconstruct_system(u, x_nodes, x_plot)

        # analytic
        p_ex2, v_ex2, wplus_ex2, wminus_ex2 = analytic_solution_pv(x_plot, p0, c, t_final_2)
        # errors
        # discrete L2 on plot grid
        dx = x_plot[1] - x_plot[0]
        L2_p = jnp.sqrt(jnp.sum((p_rec - p_ex)**2) * dx)
        L2_v = jnp.sqrt(jnp.sum((v_rec - v_ex)**2) * dx)
        Linf_p = jnp.max(jnp.abs(p_rec - p_ex))
        Linf_v = jnp.max(jnp.abs(v_rec - v_ex))
        print(f"L2 error p: {float(L2_p):.3e}, v: {float(L2_v):.3e}")
        print(f"Linf error p: {float(Linf_p):.3e}, v: {float(Linf_v):.3e}")
        # Store results in CSV
        p_erros.append(float(L2_p))
        v_errors.append(float(L2_v))
        durations.append(duration)

        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([N, duration, float(L2_p), float(L2_v), float(Linf_p), float(Linf_v)])

        print(f"Results appended to {csv_file}")

    # plot comparison
    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.plot(x_plot, p_ex, '-', label='p exact at t_final=0.2')
    plt.plot(x_plot, p_ex2, '-', label='p exact at t_final=0.3')
    plt.plot(x_plot, p_rec, '--', label='p DG at t_final=0.2')
    plt.plot(x_plot, p_rec2, '--', label='p DG at t_final=0.3')

    plt.legend(); plt.grid(True); plt.title(f"p")

    plt.subplot(2,1,2)
    plt.plot(x_plot, v_ex, '-', label='v exact at t_final=0.2')
    plt.plot(x_plot, v_ex2, '-', label='v exact at t_final=0.3')
    plt.plot(x_plot, v_rec, '--', label='v DG at t_final=0.2')
    plt.plot(x_plot, v_rec2, '--', label='v DG at t_final=0.3')
    plt.legend(); plt.grid(True); plt.title(f"v")

    plt.tight_layout()
    output_dir = 'Report&Presentation/Images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, 'dg_solution.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # also compare invariants w+ and w- (numeric)
    wplus_num = p_rec + v_rec
    wminus_num = p_rec - v_rec

    plt.figure(figsize=(10,4))
    plt.plot(x_plot, wplus_ex, '-', label='w+ exact')
    plt.plot(x_plot, wplus_num, '--', label='w+ DG')
    plt.plot(x_plot, wminus_ex, '-', label='w- exact')
    plt.plot(x_plot, wminus_num, '--', label='w- DG')
    plt.legend(); plt.grid(True); plt.title("Riemann invariants at final time")
    plt.savefig(os.path.join(output_dir, 'dg_invariants.png'), dpi=150, bbox_inches='tight')
    plt.close()

    #plot convergence
    p_error_slope=jnp.polyfit(jnp.log10(jnp.array(Ns)), jnp.log10(jnp.array(p_erros)), 1)[0]
    v_error_slope=jnp.polyfit(jnp.log10(jnp.array(Ns)), jnp.log10(jnp.array(v_errors)), 1)[0]
    plt.figure(figsize=(6,5))
    plt.loglog(Ns, p_erros, 'o-', label=f'L2 error p (slope={p_error_slope:.2f})')
    plt.loglog(Ns, v_errors, 's--', label=f'L2 error v (slope={v_error_slope:.2f})')
    plt.xlabel('Number of cells N')
    plt.ylabel('L2 error')
    plt.title('Convergence of DG P1 scheme')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dg_convergence.png'), dpi=150, bbox_inches='tight')
    plt.close()

    duration_slope = np.polyfit(np.array(Ns), np.array(durations), 1)[0]
    plt.figure(figsize=(6,5))
    plt.plot(Ns, durations, 'o-', label=f'Duration (slope={duration_slope:.2f})')
    plt.xlabel('Number of cells N')
    plt.ylabel('Duration (seconds)')
    plt.title('Computation Time vs Number of Cells')
    plt.grid(True, which='both', ls='--')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'dg_duration.png'), dpi=150, bbox_inches='tight')

    print("All done.")
if __name__ == "__main__":
    main()
