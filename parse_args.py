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
    return parser.parse_args()