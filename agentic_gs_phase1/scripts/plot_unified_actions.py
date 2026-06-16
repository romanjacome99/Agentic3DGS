from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs.spaces import CONTINUOUS_ACTIONS, DISCRETE_ACTIONS


DEFAULT_BUDGETS = [1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000]
DISCRETE_KEYS = list(DISCRETE_ACTIONS.keys())
CONTINUOUS_KEYS = list(CONTINUOUS_ACTIONS.keys())


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def trace_path(args: argparse.Namespace, budget: int) -> Path:
    run_name = f"{args.run_prefix}_{args.scene}_{budget}_{args.run_suffix}"
    return (
        args.eval_root
        / run_name
        / args.scene
        / "agentic"
        / args.scene
        / f"episode_{args.episode:04d}"
        / "action_sets.csv"
    )


def load_traces(args: argparse.Namespace) -> dict[int, list[dict[str, Any]]]:
    traces: dict[int, list[dict[str, Any]]] = {}
    missing = []
    for budget in args.budgets:
        path = trace_path(args, budget)
        if not path.exists():
            missing.append(str(path))
            continue
        rows = read_csv(path)
        decoded_rows = []
        for row in rows:
            item: dict[str, Any] = {
                "budget": budget,
                "block": to_int(row.get("block")),
                "iteration": to_int(row.get("iteration")),
                "reward": to_float(row.get("reward")),
            }
            for key in DISCRETE_KEYS:
                item[key] = row.get(key, "")
            for key in CONTINUOUS_KEYS:
                item[key] = to_float(row.get(key))
            decoded_rows.append(item)
        decoded_rows.sort(key=lambda row: (row["iteration"], row["block"]))
        traces[budget] = decoded_rows
    if missing:
        print("Missing action traces:", file=sys.stderr)
        for path in missing:
            print(f"  {path}", file=sys.stderr)
    return traces


def categorical_labels(key: str, traces: dict[int, list[dict[str, Any]]]) -> list[str]:
    preferred = [str(value) for value in DISCRETE_ACTIONS[key]]
    present = {str(row.get(key, "")) for rows in traces.values() for row in rows}
    labels = [label for label in preferred if label in present]
    labels.extend(sorted(present - set(labels)))
    return labels or preferred


def plot_discrete_actions(traces: dict[int, list[dict[str, Any]]], output_path: Path, scene: str) -> None:
    budgets = sorted(traces)
    fig, axes = plt.subplots(
        len(DISCRETE_KEYS),
        1,
        figsize=(13, 2.0 * len(DISCRETE_KEYS)),
        sharex=True,
        constrained_layout=True,
    )
    if len(DISCRETE_KEYS) == 1:
        axes = [axes]

    base_colors = list(plt.get_cmap("tab10").colors) + list(plt.get_cmap("Set3").colors)
    for ax, key in zip(axes, DISCRETE_KEYS):
        labels = categorical_labels(key, traces)
        label_to_code = {label: idx for idx, label in enumerate(labels)}
        cmap = ListedColormap(base_colors[: max(1, len(labels))])
        norm = BoundaryNorm([i - 0.5 for i in range(len(labels) + 1)], cmap.N)

        for y_index, budget in enumerate(budgets):
            rows = traces[budget]
            xs = [row["iteration"] for row in rows]
            ys = [y_index] * len(rows)
            codes = [label_to_code.get(str(row.get(key, "")), 0) for row in rows]
            ax.scatter(xs, ys, c=codes, cmap=cmap, norm=norm, marker="s", s=18, linewidths=0)

        ax.set_ylabel(key)
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels([str(budget) for budget in budgets], fontsize=8)
        ax.grid(True, axis="x", alpha=0.25)
        ax.set_ylim(-0.6, len(budgets) - 0.4)
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            ticks=range(len(labels)),
            pad=0.01,
            aspect=18,
        )
        cbar.ax.set_yticklabels(labels, fontsize=7)

    axes[0].set_title(f"{scene}: policy discrete actions by iteration budget")
    axes[-1].set_xlabel("Training iteration")
    fig.supxlabel("Each row is one experiment budget")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_continuous_actions(traces: dict[int, list[dict[str, Any]]], output_path: Path, scene: str) -> None:
    budgets = sorted(traces)
    cmap = plt.get_cmap("viridis")
    colors = {budget: cmap(idx / max(1, len(budgets) - 1)) for idx, budget in enumerate(budgets)}

    fig, axes = plt.subplots(
        len(CONTINUOUS_KEYS),
        1,
        figsize=(13, 1.85 * len(CONTINUOUS_KEYS)),
        sharex=True,
        constrained_layout=True,
    )
    if len(CONTINUOUS_KEYS) == 1:
        axes = [axes]

    for ax, key in zip(axes, CONTINUOUS_KEYS):
        for budget in budgets:
            rows = traces[budget]
            xs = [row["iteration"] for row in rows]
            values = [row[key] for row in rows]
            ax.plot(xs, values, color=colors[budget], linewidth=1.5, alpha=0.85, label=str(budget))
        ax.set_ylabel(key)
        ax.grid(True, alpha=0.25)

    axes[0].set_title(f"{scene}: policy continuous actions by iteration budget")
    axes[-1].set_xlabel("Training iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Budget", loc="outside lower center", ncol=min(8, len(budgets)))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def summarize_actions(traces: dict[int, list[dict[str, Any]]], output_path: Path) -> None:
    summary: dict[str, Any] = {}
    for budget, rows in sorted(traces.items()):
        budget_summary: dict[str, Any] = {"blocks": len(rows)}
        for key in DISCRETE_KEYS:
            counts: dict[str, int] = {}
            for row in rows:
                value = str(row.get(key, ""))
                counts[value] = counts.get(value, 0) + 1
            budget_summary[key] = counts
        for key in CONTINUOUS_KEYS:
            values = [float(row[key]) for row in rows]
            budget_summary[key] = {
                "mean": sum(values) / len(values) if values else 0.0,
                "min": min(values) if values else 0.0,
                "max": max(values) if values else 0.0,
            }
        summary[str(budget)] = budget_summary
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot action_sets.csv traces from unified agentic GS evaluations.")
    parser.add_argument("--eval-root", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_unified_eval")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_plots")
    parser.add_argument("--scene", default="hotdog")
    parser.add_argument("--run-prefix", default="unified_ppo15k_latest")
    parser.add_argument("--run-suffix", default="min50k")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.eval_root = args.eval_root.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    traces = load_traces(args)
    if not traces:
        raise SystemExit("No action traces found.")

    stem = f"{args.scene}_policy_actions_{min(traces)}_{max(traces)}"
    discrete_path = args.output_dir / f"{stem}_discrete.png"
    continuous_path = args.output_dir / f"{stem}_continuous.png"
    summary_path = args.output_dir / f"{stem}_summary.json"

    plot_discrete_actions(traces, discrete_path, args.scene)
    plot_continuous_actions(traces, continuous_path, args.scene)
    summarize_actions(traces, summary_path)

    print(discrete_path)
    print(continuous_path)
    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
