from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.scripts.report_acceleration import (
    discover_dirs,
    final_render_metrics,
    parse_baseline,
    parse_policy,
    resolve_path,
    to_float,
    to_int,
)


DISCRETE_ACTION_KEYS = [
    "block_steps",
    "densify_mode",
    "densification_interval",
    "prune_mode",
    "opacity_reset",
    "stop",
]

CONTINUOUS_ACTION_KEYS = [
    "densify_threshold_mult",
    "prune_opacity_threshold",
    "position_lr_mult",
    "feature_lr_mult",
    "opacity_lr_mult",
    "scaling_lr_mult",
    "rotation_lr_mult",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def group_curves(curves: list[list[dict[str, Any]]]) -> dict[str, list[list[dict[str, Any]]]]:
    grouped: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for curve in curves:
        if curve:
            grouped[str(curve[0]["scene"])].append(curve)
    return grouped


def label_for_curve(curve: list[dict[str, Any]]) -> str:
    first = curve[0]
    method = first["method"]
    split = first.get("metric_split", "")
    run_name = Path(str(first["run_dir"])).name
    if method == "baseline_3dgs":
        parent = Path(str(first["run_dir"])).parent.name
        run_name = f"{parent}/{run_name}"
    return f"{method} ({split}, {run_name})"


def plot_quality_vs_time(curves: list[list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    paths = []
    for scene, scene_curves in group_curves(curves).items():
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for curve in scene_curves:
            xs = [to_float(row.get("wall_seconds")) for row in curve]
            psnr_values = [to_float(row.get("psnr")) for row in curve]
            ssim_values = [to_float(row.get("ssim")) for row in curve]
            label = label_for_curve(curve)
            style = "--o" if curve[0]["method"] == "baseline_3dgs" else "-o"
            axes[0].plot(xs, psnr_values, style, label=label, linewidth=1.8, markersize=4)
            axes[1].plot(xs, ssim_values, style, label=label, linewidth=1.8, markersize=4)

            final_metrics = final_render_metrics(Path(str(curve[0]["run_dir"])))
            if final_metrics and curve[0]["method"] == "agentic_policy":
                final_x = max(xs) if xs else 0.0
                axes[0].scatter(
                    [final_x],
                    [final_metrics["test_psnr"]],
                    marker="*",
                    s=130,
                    label=f"agentic final test ({Path(str(curve[0]['run_dir'])).name})",
                )
                axes[1].scatter([final_x], [final_metrics["test_ssim"]], marker="*", s=130)

        axes[0].set_ylabel("PSNR")
        axes[1].set_ylabel("SSIM")
        axes[1].set_xlabel("Training wall-clock seconds")
        axes[0].set_title(f"{scene}: quality vs training wall-clock")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"quality_vs_wall_clock_{safe_name(scene)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_compute_vs_time(curves: list[list[dict[str, Any]]], output_dir: Path) -> list[Path]:
    paths = []
    for scene, scene_curves in group_curves(curves).items():
        fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
        for curve in scene_curves:
            xs = [to_float(row.get("wall_seconds")) for row in curve]
            gaussians = [to_int(row.get("gaussian_count")) for row in curve]
            peak_vram_gb = [to_float(row.get("peak_vram_mb")) / 1024.0 for row in curve]
            label = label_for_curve(curve)
            style = "--o" if curve[0]["method"] == "baseline_3dgs" else "-o"
            axes[0].plot(xs, gaussians, style, label=label, linewidth=1.8, markersize=4)
            axes[1].plot(xs, peak_vram_gb, style, label=label, linewidth=1.8, markersize=4)

        axes[0].set_ylabel("Gaussian count")
        axes[1].set_ylabel("Peak VRAM GB")
        axes[1].set_xlabel("Training wall-clock seconds")
        axes[0].set_title(f"{scene}: compute vs training wall-clock")
        for ax in axes:
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        fig.tight_layout()
        path = output_dir / f"compute_vs_wall_clock_{safe_name(scene)}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)
    return paths


def decode_action_rows(policy_dir: Path) -> tuple[str, list[dict[str, Any]]]:
    rows = read_csv(policy_dir / "agentic_blocks.csv")
    decoded = []
    scene = policy_dir.parent.name
    for row in rows:
        action_text = row.get("action_json", "{}")
        try:
            action = json.loads(action_text)
        except json.JSONDecodeError:
            action = {}
        item = {
            "scene": row.get("scene", scene),
            "block": to_int(row.get("block")),
            "iteration": to_int(row.get("iteration")),
            "elapsed_seconds": to_float(row.get("elapsed_seconds")),
            "reward": to_float(row.get("reward")),
            "validation_psnr": to_float(row.get("validation_psnr")),
            **action,
        }
        decoded.append(item)
    if decoded:
        scene = str(decoded[0]["scene"])
    return scene, decoded


def categorical_codes(values: list[Any]) -> tuple[list[int], list[str]]:
    labels = []
    codes = []
    for value in values:
        text = str(value)
        if text not in labels:
            labels.append(text)
        codes.append(labels.index(text))
    return codes, labels


def plot_policy_actions(policy_dirs: list[Path], output_dir: Path) -> list[Path]:
    paths = []
    for policy_dir in policy_dirs:
        scene, rows = decode_action_rows(policy_dir)
        if not rows:
            continue
        xs = [to_float(row.get("elapsed_seconds")) for row in rows]
        run_label = safe_name(f"{scene}_{policy_dir.name}")

        fig, axes = plt.subplots(len(DISCRETE_ACTION_KEYS), 1, figsize=(10, 1.8 * len(DISCRETE_ACTION_KEYS)), sharex=True)
        if len(DISCRETE_ACTION_KEYS) == 1:
            axes = [axes]
        for ax, key in zip(axes, DISCRETE_ACTION_KEYS):
            values = [row.get(key, "") for row in rows]
            codes, labels = categorical_codes(values)
            ax.step(xs, codes, where="post", linewidth=1.8)
            ax.set_ylabel(key)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Training wall-clock seconds")
        fig.suptitle(f"{scene}: discrete policy decisions")
        fig.tight_layout()
        path = output_dir / f"actions_discrete_{run_label}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(len(CONTINUOUS_ACTION_KEYS), 1, figsize=(10, 1.8 * len(CONTINUOUS_ACTION_KEYS)), sharex=True)
        if len(CONTINUOUS_ACTION_KEYS) == 1:
            axes = [axes]
        for ax, key in zip(axes, CONTINUOUS_ACTION_KEYS):
            values = [to_float(row.get(key)) for row in rows]
            ax.plot(xs, values, "-o", linewidth=1.8, markersize=3)
            ax.set_ylabel(key)
            ax.grid(True, alpha=0.25)
        axes[-1].set_xlabel("Training wall-clock seconds")
        fig.suptitle(f"{scene}: continuous policy controls")
        fig.tight_layout()
        path = output_dir / f"actions_continuous_{run_label}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(xs, [to_float(row.get("validation_psnr")) for row in rows], "-o", linewidth=1.8, markersize=3)
        axes[0].set_ylabel("Val PSNR")
        axes[1].plot(xs, [to_float(row.get("reward")) for row in rows], "-o", linewidth=1.8, markersize=3)
        axes[1].set_ylabel("Reward")
        axes[2].plot(xs, [to_float(row.get("position_lr_mult")) for row in rows], label="position")
        axes[2].plot(xs, [to_float(row.get("feature_lr_mult")) for row in rows], label="feature")
        axes[2].plot(xs, [to_float(row.get("opacity_lr_mult")) for row in rows], label="opacity")
        axes[2].plot(xs, [to_float(row.get("scaling_lr_mult")) for row in rows], label="scaling")
        axes[2].plot(xs, [to_float(row.get("rotation_lr_mult")) for row in rows], label="rotation")
        axes[2].set_ylabel("LR mult")
        axes[2].set_xlabel("Training wall-clock seconds")
        axes[2].legend(fontsize=8, ncol=3)
        for ax in axes:
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"{scene}: policy decisions and validation response")
        fig.tight_layout()
        path = output_dir / f"actions_summary_{run_label}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        paths.append(path)

    return paths


def collect_curves(baseline_dirs: list[Path], policy_dirs: list[Path]) -> list[list[dict[str, Any]]]:
    curves = []
    for model_dir in baseline_dirs:
        _, curve = parse_baseline(model_dir)
        if curve:
            curves.append(curve)
    for episode_dir in policy_dirs:
        _, curve = parse_policy(episode_dir)
        if curve:
            curves.append(curve)
    return curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot 3DGS baseline vs policy wall-clock curves and policy actions.")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "outputs" / "3dgs_baseline")
    parser.add_argument("--policy-root", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_eval")
    parser.add_argument("--baseline-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--policy-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_plots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dirs = [resolve_path(path) for path in args.baseline_dirs]
    policy_dirs = [resolve_path(path) for path in args.policy_dirs]
    if not baseline_dirs:
        baseline_dirs = discover_dirs(resolve_path(args.baseline_root), "baseline_eval_metrics.csv")
    if not policy_dirs:
        policy_dirs = discover_dirs(resolve_path(args.policy_root), "agentic_blocks.csv")

    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curves = collect_curves(baseline_dirs, policy_dirs)
    paths = []
    paths.extend(plot_quality_vs_time(curves, output_dir))
    paths.extend(plot_compute_vs_time(curves, output_dir))
    paths.extend(plot_policy_actions(policy_dirs, output_dir))

    manifest = {
        "baseline_dirs": [str(path) for path in baseline_dirs],
        "policy_dirs": [str(path) for path in policy_dirs],
        "plots": [str(path) for path in paths],
    }
    with (output_dir / "plot_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote {len(paths)} plots to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
