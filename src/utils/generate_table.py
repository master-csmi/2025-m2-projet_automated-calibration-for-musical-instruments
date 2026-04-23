# ===================================================================================================
# generate_table.py
# Reads performance CSV files and generates LaTeX tables for Euler and RK2 methods
# ===================================================================================================

import os

def csv_to_latex_table(csv_file, tex_file, method_name):
    """
    Reads a CSV file of the form:
        cross-section: const
        N,wall_time,nsteps
        100,0.238,500
        ...
    and writes a LaTeX table to tex_file.
    """

    # Read CSV data
    data = {"const": {}, "exp": {}, "cone": {}}
    current = None

    with open(csv_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("cross-section:"):
                current = line.split(":")[1].strip()
                continue
            if line.startswith("N,") or current is None:
                continue

            N, wall_time, _ = line.split(",")
            data[current][int(N)] = float(wall_time)

    Ns = sorted(data["const"].keys())

    # Write LaTeX table
    with open(tex_file, "w") as f:
        f.write(f"""\\begin{{table}}[h!]
\\centering
\\begin{{tabular}}{{c ccc}}
\\hline
$N \\backslash S(x)$ & Const. & Exp. & Cone \\\\
\\hline
""")

        for N in Ns:
            f.write(
                f"{N} & "
                f"{data['const'][N]:.3f} & "
                f"{data['exp'][N]:.3f} & "
                f"{data['cone'][N]:.3f} \\\\\n"
            )

        f.write(f"""\\hline
\\end{{tabular}}
\\caption{{Execution time (s) of the DG P1 solver using the {method_name.upper()} scheme.}}
\\label{{tab:performance-{method_name.lower()}}}
\\end{{table}}
""")

    print(f"LaTeX table written to {tex_file}")



def main():
    res_dir = "Results"
    os.makedirs(res_dir, exist_ok=True)

    # Euler
    csv_file_euler = os.path.join(res_dir, "performance_euler.csv")
    tex_file_euler = os.path.join(res_dir, "table_performance_euler.tex")
    csv_to_latex_table(csv_file_euler, tex_file_euler, "euler")

    # RK2
    csv_file_rk2 = os.path.join(res_dir, "performance_rk2.csv")
    tex_file_rk2 = os.path.join(res_dir, "table_performance_rk2.tex")
    csv_to_latex_table(csv_file_rk2, tex_file_rk2, "rk2")


if __name__ == "__main__":
    main()
