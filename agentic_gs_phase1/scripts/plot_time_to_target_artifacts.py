from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.ticker import AutoMinorLocator, FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs.spaces import CONTINUOUS_ACTIONS, DISCRETE_ACTIONS


DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "agentic_gs_phase1_reports"
    / "hotdog_time_to_target_ppo5scene_fast_latest_targets_21_34_max10000_20260530_194210_enriched"
)

METHOD_COLORS = {
    "baseline": "#0072B2",
    "agentic": "#D55E00",
}
ACCELERATION_COLOR = "#009E73"
GRID_COLOR = "#D6D6D6"
TEXT_COLOR = "#202020"
SPINE_COLOR = "#4D4D4D"
DISCRETE_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#6F4E7C",
]


def configure_plot_style(tex_mode: str) -> str:
    if tex_mode == "auto":
        use_tex = shutil.which("latex") is not None
    else:
        use_tex = tex_mode == "on"

    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    matplotlib.rcParams.update(
        {
            "text.usetex": use_tex,
            "font.family": "serif",
            "font.serif": ["cmr10", "Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
            "mathtext.fontset": "cm",
            "axes.formatter.use_mathtext": True,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.titleweight": "regular",
            "axes.labelcolor": TEXT_COLOR,
            "axes.edgecolor": SPINE_COLOR,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "legend.fontsize": 9.5,
            "legend.title_fontsize": 10,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return "latex" if use_tex else "computer-modern"


def format_axes(ax: plt.Axes, *, minor_y: bool = True) -> None:
    ax.grid(True, which="major", color=GRID_COLOR, linewidth=0.85, alpha=0.65)
    ax.grid(True, which="minor", color=GRID_COLOR, linewidth=0.45, alpha=0.28)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    if minor_y:
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(axis="both", which="major", length=4.5, width=0.8, color=SPINE_COLOR)
    ax.tick_params(axis="both", which="minor", length=2.5, width=0.6, color=SPINE_COLOR)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)


def format_legend(legend: Any) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_edgecolor("#CFCFCF")
    frame.set_alpha(0.94)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value).replace(",", "."))
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value).replace(",", ".")))
    except (TypeError, ValueError):
        return default


def save_figure(fig: plt.Figure, output_base: Path) -> list[Path]:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    paths = [output_base.with_suffix(".png"), output_base.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return paths


def methods_in(rows: list[dict[str, str]]) -> list[str]:
    preferred = ["baseline", "agentic"]
    present = {row.get("method", row.get("sample_method", "")) for row in rows}
    methods = [method for method in preferred if method in present]
    methods.extend(sorted(present - set(methods)))
    return methods


def scene_title_from_rows(rows: list[dict[str, str]]) -> str:
    for row in rows:
        scene = row.get("scene") or row.get("sample_scene")
        if scene:
            return scene.replace("_", " ").title()
    return "Scene"


def plot_time_to_psnr(
    curve_rows: list[dict[str, str]],
    comparison_rows: list[dict[str, str]],
    output_dir: Path,
    scene_title: str,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)

    for method in methods_in(curve_rows):
        rows = [row for row in curve_rows if row.get("method") == method]
        rows.sort(key=lambda row: to_float(row.get("train_wall_seconds")))
        xs = [to_float(row.get("train_wall_seconds")) for row in rows]
        ys = [to_float(row.get("test_psnr")) for row in rows]
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.25,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=METHOD_COLORS.get(method),
            label=method.title(),
        )

    targets = sorted({to_float(row.get("target_psnr")) for row in comparison_rows if row.get("target_psnr")})
    for target in targets:
        ax.axhline(target, color="#9A9A9A", linewidth=0.55, alpha=0.16, zorder=0)

    ax.set_title(fr"{scene_title}: test PSNR over wall-clock time")
    ax.set_xlabel(r"Training wall-clock time (s)")
    ax.set_ylabel(r"Held-out test PSNR (dB)")
    format_axes(ax)
    format_legend(ax.legend(title=r"Method", loc="lower right"))
    return save_figure(fig, output_dir / "time_to_psnr")


def plot_time_to_target(comparison_rows: list[dict[str, str]], output_dir: Path, scene_title: str) -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.2, 5.1), constrained_layout=True)
    target_psnr = [to_float(row.get("target_psnr")) for row in comparison_rows]
    series = [
        ("baseline", "baseline_time_seconds"),
        ("agentic", "agentic_time_seconds"),
    ]
    for method, column in series:
        xs = []
        ys = []
        for target, row in zip(target_psnr, comparison_rows):
            seconds = row.get(column, "")
            if seconds == "":
                continue
            xs.append(target)
            ys.append(to_float(seconds))
        ax.plot(
            xs,
            ys,
            marker="o",
            linewidth=2.25,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.6,
            color=METHOD_COLORS.get(method),
            label=method.title(),
        )

    ax.set_title(fr"{scene_title}: time required to reach target PSNR")
    ax.set_xlabel(r"Target held-out test PSNR (dB)")
    ax.set_ylabel(r"Interpolated training wall-clock time (s)")
    format_axes(ax)

    speedup_x = [
        to_float(row.get("speedup_x"), default=float("nan"))
        for row in comparison_rows
        if row.get("speedup_x", "") != ""
    ]
    speedup_targets = [
        to_float(row.get("target_psnr"))
        for row in comparison_rows
        if row.get("speedup_x", "") != ""
    ]
    if speedup_x:
        ax2 = ax.twinx()
        ax2.plot(
            speedup_targets,
            speedup_x,
            color=ACCELERATION_COLOR,
            marker="s",
            linestyle="--",
            linewidth=1.75,
            markersize=4.4,
            markeredgecolor="white",
            markeredgewidth=0.45,
            label=r"Acceleration factor",
        )
        ax2.axhline(1.0, color=ACCELERATION_COLOR, linewidth=0.9, alpha=0.32)
        ax2.set_ylabel(r"Acceleration factor (baseline / agentic)")
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:.2f}x"))
        lower = min([1.0, *speedup_x])
        upper = max([1.0, *speedup_x])
        pad = max(0.03, (upper - lower) * 0.18)
        ax2.set_ylim(max(0.0, lower - pad), upper + pad)
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_color(SPINE_COLOR)
        ax2.tick_params(axis="y", colors=ACCELERATION_COLOR)
        ax2.yaxis.label.set_color(ACCELERATION_COLOR)
        for x_value, y_value in zip(speedup_targets, speedup_x):
            ax2.annotate(
                f"{y_value:.2f}x",
                (x_value, y_value),
                textcoords="offset points",
                xytext=(0, 7 if y_value >= 1.0 else -13),
                ha="center",
                fontsize=7.2,
                color=ACCELERATION_COLOR,
            )
        handles, labels = ax.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        format_legend(
            ax.legend(
                handles + handles2,
                labels + labels2,
                title=r"Series",
                loc="upper center",
                bbox_to_anchor=(0.5, -0.14),
                ncol=3,
            )
        )
    else:
        format_legend(ax.legend(title=r"Method"))
    return save_figure(fig, output_dir / "time_to_target_psnr")


def plot_gaussians(
    curve_rows: list[dict[str, str]],
    block_rows: list[dict[str, str]],
    output_dir: Path,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), constrained_layout=True)

    for method in methods_in(curve_rows):
        rows = [row for row in curve_rows if row.get("method") == method]
        rows.sort(key=lambda row: to_int(row.get("iteration")))
        axes[0].plot(
            [to_int(row.get("iteration")) for row in rows],
            [to_int(row.get("final_gaussians")) for row in rows],
            marker="o",
            linewidth=2.15,
            markersize=4.8,
            markeredgecolor="white",
            markeredgewidth=0.55,
            color=METHOD_COLORS.get(method),
            label=method.title(),
        )

    max_sample = max((to_int(row.get("sample_iteration")) for row in block_rows), default=0)
    for method in methods_in(block_rows):
        rows = [
            row
            for row in block_rows
            if row.get("sample_method") == method and to_int(row.get("sample_iteration")) == max_sample
        ]
        rows.sort(key=lambda row: to_float(row.get("elapsed_seconds")))
        if not rows:
            continue
        axes[1].plot(
            [to_float(row.get("elapsed_seconds")) for row in rows],
            [to_int(row.get("gaussian_count")) for row in rows],
            linewidth=2.15,
            color=METHOD_COLORS.get(method),
            label=method.title(),
        )

    for ax in axes:
        format_axes(ax)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1000:.0f}k"))

    axes[0].set_title(r"Final Gaussians by iteration budget")
    axes[0].set_xlabel(r"Iteration budget")
    axes[0].set_ylabel(r"Final Gaussian count")
    format_legend(axes[0].legend(title=r"Method"))

    axes[1].set_title(fr"Gaussian trajectory at {max_sample} iterations")
    axes[1].set_xlabel(r"Training wall-clock time (s)")
    axes[1].set_ylabel(r"Gaussian count")
    format_legend(axes[1].legend(title=r"Method"))

    return save_figure(fig, output_dir / "gaussians")


def action_rows_for_method(action_rows: list[dict[str, str]], method: str) -> list[dict[str, str]]:
    rows = [row for row in action_rows if row.get("sample_method") == method]
    return sorted(rows, key=lambda row: (to_int(row.get("sample_iteration")), to_int(row.get("iteration")), to_int(row.get("block"))))


def categorical_labels(key: str, rows: list[dict[str, str]]) -> list[str]:
    preferred = [str(value) for value in DISCRETE_ACTIONS[key]]
    present = {str(row.get(key, "")) for row in rows}
    labels = [label for label in preferred if label in present]
    labels.extend(sorted(present - set(labels)))
    return labels or preferred


def plot_discrete_actions(action_rows: list[dict[str, str]], output_dir: Path, method: str) -> list[Path]:
    rows = action_rows_for_method(action_rows, method)
    budgets = sorted({to_int(row.get("sample_iteration")) for row in rows})
    fig, axes = plt.subplots(
        len(DISCRETE_ACTIONS),
        1,
        figsize=(13.5, 2.05 * len(DISCRETE_ACTIONS)),
        sharex=True,
        constrained_layout=True,
    )
    if len(DISCRETE_ACTIONS) == 1:
        axes = [axes]

    base_colors = DISCRETE_PALETTE + list(plt.get_cmap("Set3").colors)
    for ax, key in zip(axes, DISCRETE_ACTIONS):
        labels = categorical_labels(key, rows)
        label_to_code = {label: idx for idx, label in enumerate(labels)}
        cmap = ListedColormap(base_colors[: max(1, len(labels))])
        norm = BoundaryNorm([idx - 0.5 for idx in range(len(labels) + 1)], cmap.N)

        for y_index, budget in enumerate(budgets):
            budget_rows = [row for row in rows if to_int(row.get("sample_iteration")) == budget]
            xs = [to_int(row.get("iteration")) for row in budget_rows]
            ys = [y_index] * len(budget_rows)
            codes = [label_to_code.get(str(row.get(key, "")), 0) for row in budget_rows]
            ax.scatter(
                xs,
                ys,
                c=codes,
                cmap=cmap,
                norm=norm,
                marker="s",
                s=18,
                linewidths=0,
                alpha=0.94,
            )

        ax.set_ylabel(key.replace("_", " "))
        ax.set_yticks(range(len(budgets)))
        ax.set_yticklabels([str(budget) for budget in budgets], fontsize=8)
        ax.set_ylim(-0.6, len(budgets) - 0.4)
        format_axes(ax, minor_y=False)
        ax.grid(False, axis="y")
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            ticks=range(len(labels)),
            pad=0.01,
            aspect=18,
        )
        cbar.ax.set_yticklabels([label.replace("_", " ") for label in labels], fontsize=7)
        cbar.outline.set_edgecolor("#CFCFCF")

    axes[0].set_title(fr"{method.title()}: discrete action sets by iteration budget")
    axes[-1].set_xlabel(r"Training iteration")
    return save_figure(fig, output_dir / f"action_sets_{method}_discrete")


def plot_continuous_actions(action_rows: list[dict[str, str]], output_dir: Path, method: str) -> list[Path]:
    rows = action_rows_for_method(action_rows, method)
    budgets = sorted({to_int(row.get("sample_iteration")) for row in rows})
    cmap = plt.get_cmap("viridis")
    colors = {budget: cmap(idx / max(1, len(budgets) - 1)) for idx, budget in enumerate(budgets)}

    fig, axes = plt.subplots(
        len(CONTINUOUS_ACTIONS),
        1,
        figsize=(13.5, 1.85 * len(CONTINUOUS_ACTIONS)),
        sharex=True,
        constrained_layout=True,
    )
    if len(CONTINUOUS_ACTIONS) == 1:
        axes = [axes]

    for ax, key in zip(axes, CONTINUOUS_ACTIONS):
        for budget in budgets:
            budget_rows = [row for row in rows if to_int(row.get("sample_iteration")) == budget]
            ax.plot(
                [to_int(row.get("iteration")) for row in budget_rows],
                [to_float(row.get(key)) for row in budget_rows],
                color=colors[budget],
                linewidth=1.45,
                alpha=0.88,
                label=str(budget),
            )
        ax.set_ylabel(key.replace("_", " "))
        format_axes(ax)

    axes[0].set_title(fr"{method.title()}: continuous action sets by iteration budget")
    axes[-1].set_xlabel(r"Training iteration")
    handles, labels = axes[0].get_legend_handles_labels()
    format_legend(fig.legend(handles, labels, title=r"Budget", loc="outside lower center", ncol=min(7, len(budgets))))
    return save_figure(fig, output_dir / f"action_sets_{method}_continuous")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot time-to-target report artifacts.")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--action-method", choices=["agentic", "baseline"], default="agentic")
    parser.add_argument(
        "--tex-font",
        choices=["auto", "on", "off"],
        default="auto",
        help="Use real LaTeX if available/requested, otherwise use Computer Modern math/text fonts.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    font_mode = configure_plot_style(args.tex_font)
    print(f"Using {font_mode} plot font mode")
    report_dir = args.report_dir.resolve()
    output_dir = (args.output_dir or report_dir).resolve()

    curve_rows = read_csv(report_dir / "sampled_test_curve.csv")
    comparison_rows = read_csv(report_dir / "time_to_target_comparison.csv")
    block_rows = read_csv(report_dir / "sampled_training_blocks.csv")
    action_rows = read_csv(report_dir / "sampled_action_sets.csv")

    written: list[Path] = []
    scene_title = scene_title_from_rows(curve_rows)
    written.extend(plot_time_to_psnr(curve_rows, comparison_rows, output_dir, scene_title))
    written.extend(plot_time_to_target(comparison_rows, output_dir, scene_title))
    written.extend(plot_gaussians(curve_rows, block_rows, output_dir))
    written.extend(plot_discrete_actions(action_rows, output_dir, args.action_method))
    written.extend(plot_continuous_actions(action_rows, output_dir, args.action_method))

    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
