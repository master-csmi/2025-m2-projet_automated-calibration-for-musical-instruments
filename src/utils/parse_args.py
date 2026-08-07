import argparse

# Made by chat Gpt
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
    parser.add_argument("--L", type=float, default=1,
                        help="Length of the domain")
    parser.add_argument("--type_S", type=str, default="const",
                        help="Type of the section profile: 'const', 'exp', 'cone', 'bump'") 
    parser.add_argument("--Th_study", type=str, default="without",
                        help="Theoretical study for p and v (Convergence included): 'with', 'without'") 
    parser.add_argument("--target_source", type=str, default="openwind",
                        choices=["openwind", "dg"],
                        help="Target signal source for gradient-descent calibration")
    parser.add_argument("--ow_order", type=int, default=4,
                        help="OpenWind finite-element order for target generation")
    parser.add_argument("--ow_theta", type=float, default=0.5,
                        help="OpenWind theta-scheme parameter for target generation")
    parser.add_argument("--ow_l_ele", type=float, default=None,
                        help="OpenWind element length for target generation")
    parser.add_argument("--dataset_path", type=str,
                        default="experiments/gradient/datasets/openwind_Qr_wr_gamma_zeta_300.npz",
                        help="Pregenerated OpenWind dataset used by train_ly.py")
    parser.add_argument("--n_iter", type=int, default=None,
                        help="Number of training iterations for grad_des.py")
    parser.add_argument("--n_iter_precise", type=int, default=None,
                        help="Number of precise-refinement iterations for grad_des.py")
    parser.add_argument("--skip_precise", action="store_true",
                        help="Skip precise-refinement stage in grad_des.py")
    parser.add_argument("--experiment_protocol", type=str, default="single",
                        choices=["single", "individual"],
                        help="Run one optimization or the individual-parameter experiment protocol")
    parser.add_argument("--protocol_params", type=str,
                        default="alpha,beta,zeta,Zt,gamma_final,fr,Qr",
                        help="Comma-separated parameter list for the individual protocol")
    parser.add_argument("--protocol_n_signals", type=int, default=20,
                        help="Number of OpenWind target signals per parameter")
    parser.add_argument("--protocol_n_iter", type=int, default=200,
                        help="Training iterations for each individual protocol fit")
    parser.add_argument("--protocol_print_every", type=int, default=10,
                        help="Print one training progress line every N iterations in the individual protocol")
    parser.add_argument("--protocol_seed", type=int, default=0,
                        help="Random seed for protocol target values")
    parser.add_argument("--protocol_output_dir", type=str,
                        default="../experiments/gradient/results/individual_protocol",
                        help="Output directory for protocol CSV and LaTeX tables")
    parser.add_argument("--scan_params", type=str,
                        default="alpha,beta,zeta,Zt,gamma_final,fr,Qr",
                        help="Comma-separated parameter list for scan_loss_1_D.py")
    parser.add_argument("--scan_n", type=int, default=120,
                        help="Number of sampled values per parameter for scan_loss_1_D.py")
    parser.add_argument("--scan_nx", type=int, default=None,
                        help="Optional DG cell count override for scan_loss_1_D.py")
    parser.add_argument("--scan_n_snapshot", type=int, default=None,
                        help="Optional snapshot count override for scan_loss_1_D.py")
    parser.add_argument("--scan_range", type=str, default=None,
                        help="Optional scan range min,max. Use with one scanned parameter")
    parser.add_argument("--scan_print_every", type=int, default=10,
                        help="Print one scan progress line every N values")
    parser.add_argument("--scan_output_dir", type=str,
                        default="../experiments/gradient/results/scans",
                        help="Output directory for scan_loss_1_D.py")
    parser.add_argument("--scan_target_source", type=str, default="both",
                        choices=["dg", "openwind", "both"],
                        help="Target signal source for scan_loss_1_D.py")
    parser.add_argument("--scan_stft_resolutions", type=str,
                        default="8:4,16:4,32:8,64:8,80:10,128:16,256:32",
                        help="Comma-separated STFT resolutions n_fft:hop for scan_loss_1_D.py")
    parser.add_argument("--scan_stft_dynamic_db", type=float, default=60.0,
                        help="Dynamic range in dB for the spectral loss used by scan_loss_1_D.py")
    parser.add_argument("--scan_no_stft_padding", action="store_true",
                        help="Do not use zero-padding for STFT resolutions larger than the signal")
    parser.add_argument("--scan_combo_time_weight", type=float, default=1.0,
                        help="Time-loss weight in the combined scan loss")
    parser.add_argument("--scan_combo_spec_weight", type=float, default=1.0,
                        help="STFT-loss weight in the combined scan loss")
    parser.add_argument("--scan_t_max_values", type=str,
                        default="0.01,0.05,0.20",
                        help="Comma-separated T_max values for scan_loss_1_D.py")
    return parser.parse_args()
