from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def parse_float(value: str) -> float:
    return float(str(value).replace(",", "."))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot iteration budget, method, wall-clock time, and PSNR.")
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=PROJECT_ROOT
        / "outputs"
        / "agentic_gs_phase1_unified_eval"
        / "hotdog_latest_budgets_1000_3000_5000_7000_summary.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_plots" / "hotdog_budget_time_psnr.png",
    )
    parser.add_argument(
        "--scatter-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_plots" / "hotdog_psnr_vs_time_scatter.png",
    )
    parser.add_argument("--subset", type=str, default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_csv = args.summary_csv if args.summary_csv.is_absolute() else PROJECT_ROOT / args.summary_csv
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    rows_by_method: dict[str, list[dict[str, float]]] = defaultdict(list)

    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["subset"] != args.subset:
                continue
            rows_by_method[row["method"]].append(
                {
                    "iterations": float(row["budget_iterations"]),
                    "time": parse_float(row["train_wall_seconds"]),
                    "psnr": parse_float(row["mean_psnr"]),
                    "gaussians": float(row["final_gaussians"]),
                }
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    method_labels = {
        "fixed_3dgs_same_split": "Fixed 3DGS",
        "agentic_policy": "Agentic policy",
    }

    for method, rows in sorted(rows_by_method.items()):
        rows = sorted(rows, key=lambda item: item["iterations"])
        xs = [row["iterations"] for row in rows]
        times = [row["time"] for row in rows]
        psnrs = [row["psnr"] for row in rows]
        label = method_labels.get(method, method)
        axes[0].plot(xs, times, marker="o", linewidth=2, label=label)
        axes[1].plot(xs, psnrs, marker="o", linewidth=2, label=label)

    axes[0].set_title("Training Time vs Iteration Budget")
    axes[0].set_xlabel("Iterations")
    axes[0].set_ylabel("Training wall-clock seconds")
    axes[1].set_title("PSNR vs Iteration Budget")
    axes[1].set_xlabel("Iterations")
    axes[1].set_ylabel("Test PSNR")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Hotdog: Fixed 3DGS vs Agentic Policy")
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)

    scatter_output = args.scatter_output if args.scatter_output.is_absolute() else PROJECT_ROOT / args.scatter_output
    fig, ax = plt.subplots(figsize=(7, 5.2))
    marker_by_method = {
        "fixed_3dgs_same_split": "o",
        "agentic_policy": "s",
    }
    scatter_for_colorbar = None
    for method, rows in sorted(rows_by_method.items()):
        rows = sorted(rows, key=lambda item: item["iterations"])
        scatter = ax.scatter(
            [row["time"] for row in rows],
            [row["psnr"] for row in rows],
            c=[row["iterations"] for row in rows],
            cmap="viridis",
            marker=marker_by_method.get(method, "o"),
            s=95,
            edgecolors="black",
            linewidths=0.7,
            label=method_labels.get(method, method),
        )
        scatter_for_colorbar = scatter
        for row in rows:
            ax.annotate(
                f"{int(row['iterations'])}",
                (row["time"], row["psnr"]),
                textcoords="offset points",
                xytext=(5, 5),
                fontsize=8,
            )

    ax.set_title("Hotdog: Test PSNR vs Training Time")
    ax.set_xlabel("Training wall-clock seconds")
    ax.set_ylabel("Test PSNR")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Method")
    if scatter_for_colorbar is not None:
        cbar = fig.colorbar(scatter_for_colorbar, ax=ax)
        cbar.set_label("Iteration budget")
    fig.tight_layout()
    fig.savefig(scatter_output, dpi=180)
    plt.close(fig)
    print(output)
    print(scatter_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
