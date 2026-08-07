import argparse
import csv
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DISPLAY_NAMES = {
    "gamma_final": r"$\gamma$",
    "zeta": r"$\zeta$",
    "fr": r"$f_r$",
    "Qr": r"$Q_r$",
    "alpha": r"$\alpha$",
    "beta": r"$\beta$",
    "Zt": r"$Z_t$",
}


def load_rows(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV vide: {csv_path}")

    required = {"parameter", "signal_idx", "true_value", "estimated_value"}
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Colonnes manquantes dans {csv_path}: {sorted(missing)}")

    return rows


def group_by_parameter(rows):
    groups = {}
    order = []

    for row in rows:
        param = row["parameter"]
        if param not in groups:
            groups[param] = []
            order.append(param)
        groups[param].append(row)

    for param_rows in groups.values():
        param_rows.sort(key=lambda row: int(row["signal_idx"]))

    return order, groups


def make_plot(csv_path, output_path):
    rows = load_rows(csv_path)
    param_order, groups = group_by_parameter(rows)

    n_params = len(param_order)
    n_cols = min(2, n_params)
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6.5 * n_cols, 4.0 * n_rows),
        squeeze=False,
    )
    axes_flat = axes.ravel()

    for ax, param in zip(axes_flat, param_order):
        param_rows = groups[param]
        x = [int(row["signal_idx"]) + 1 for row in param_rows]
        true_values = [float(row["true_value"]) for row in param_rows]
        estimated_values = [float(row["estimated_value"]) for row in param_rows]

        x_true = [value - 0.08 for value in x]
        x_estimated = [value + 0.08 for value in x]
        rel_errors = [float(row["relative_error"]) for row in param_rows]

        ax.scatter(x_true, true_values, marker="o", s=48, label="Theorique")
        ax.scatter(x_estimated, estimated_values, marker="s", s=48, label="Entraine")

        for x_i, y_i, rel_error in zip(x_estimated, estimated_values, rel_errors):
            ax.annotate(
                f"{rel_error:.2e}",
                (x_i, y_i),
                textcoords="offset points",
                xytext=(5, 5),
                ha="left",
                va="bottom",
                fontsize=8,
            )

        ax.set_title(DISPLAY_NAMES.get(param, param))
        ax.set_xlabel("Signal")
        ax.set_ylabel("Valeur du parametre")
        ax.set_xticks(x)
        ax.grid(True, alpha=0.3)
        ax.legend()

        all_values = true_values + estimated_values
        ymin = min(all_values)
        ymax = max(all_values)
        if ymin == ymax:
            margin = max(abs(ymin) * 0.05, 1e-6)
        else:
            margin = 0.08 * (ymax - ymin)
        ax.set_ylim(ymin - margin, ymax + margin)

    for ax in axes_flat[n_params:]:
        ax.axis("off")

    fig.suptitle("Parametres theoriques vs entraines avec erreur relative", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot true and estimated values from individual_protocol_results.csv"
    )
    parser.add_argument("csv_path", help="Path to individual_protocol_results.csv")
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path. Default: individual_protocol_values.png next to the CSV",
    )
    args = parser.parse_args()

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(args.csv_path),
            "individual_protocol_values.png",
        )

    make_plot(args.csv_path, output_path)
    print(f"Figure ecrite: {output_path}")


if __name__ == "__main__":
    main()
