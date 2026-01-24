# Projet_Automated-Calibration-of-Physical-Models-for-Musical-Instruments

In this project, we want to implement a method to automatically compute the parameters of physical models in order to reproduce the characteristic sounds of reed instruments. 

## Objectives 
 * Implement a Discontinuous Galerkin Method to numerically solve the mixed form of the wave equation
 * Coupling with an ODE and an algebraic equation for the boundary conditions
 * Compute the gradient with respect to all parameters to be able to train a model in a future work. 

## Project Structures
```
2025_Projet_Automated-Calibration-of-Physical-Models-for-Musical-Instruments/
│
├── main.py
├── performance.py
├── parse_args.py # argument parser for the simulation
│
├── dg_solver/
│   ├── mesh.py
│   ├── basis.py
│   ├── mass_matrix.py
│   ├── time_integrators.py
│   └── reconstruction.py
│
├── bc/
│   └── bc.py
│
├── utils/
│   ├── init_func.py # initial condition
|   ├── flux.py 
|   ├── generate_table.py # CSV->Latex  
│   └── S_profiles.py # Different cross-section functions
│
├── Results/
├── Report_and_Presentation/
|   ├── report.tex # latex report of the Project
|   ├── presentation.tex # latex slides for the oral presentation
│   └── Images/
│
├── requirements.txt
├── test.py 
└── README.md
```

### Notes
- `main.py` runs the simulation and generates figures.
- `performance.py` runs the simulation and generates performance figures.
- `dg_solver/` contains the functions needed to numerically solve the mixed form of the wave equation using a Discontinuous Galerkin method.
- `bc/` contains the boundary conditions implementation
- `utils/` contains the initial functions and the different profiles of cross-section
- `Results/` and `Report_and_Presentation/Images` are created automatically during simulations.

## Commands 

Here are the commands to execute the code completely. 

### 1. Create a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt  
```
### 2. Execute Python Script
```bash
python3 main.py
```
to run the simulation

with the following available options


### 3. Available options

| Option       | Type   | Default | Description |
|-------------|--------|---------|-------------|
| `--method`  | string | `rk2`   | Time integration method (`euler`, `rk2`) |
| `--N`       | int    | `200`   | Number of spatial cells |
| `--CFL`     | float  | `0.05`  | CFL number |
| `--tfinal`  | float  | `0.2`   | Final simulation time |
| `--L`       | float  | `1.0`   | Length of the spatial domain |
| `--type_S`  | string | `const` | Cross-section profile (`const`, `exp`, `cone`, `bump`) |

### 4. Performance Study

```bash
python3 performance.py
```
to run the performance study (only use option `--L`)

### 5. Tests
To run the tests locally (ensure the virtual environment is activated and dependencies are installed):

```bash
pytest test.py -v
```

The test verifies the convergence order of the numerical solution for different time integration schemes (Euler vs RK2).