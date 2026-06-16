from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def discover_dirs(root: Path, marker: str) -> list[Path]:
    if not root.exists():
        return []
    if (root / marker).exists():
        return [root]
    return sorted(path.parent for path in root.rglob(marker))


def final_render_metrics(model_dir: Path) -> dict[str, float]:
    results = read_json(model_dir / "results.json")
    if not results:
        return {}
    methods = sorted(results.keys())
    if not methods:
        return {}
    method = methods[-1]
    metrics = results.get(method, {})
    return {
        "test_psnr": to_float(metrics.get("PSNR")),
        "test_ssim": to_float(metrics.get("SSIM")),
        "test_lpips": to_float(metrics.get("LPIPS")),
    }


def baseline_train_wall_seconds(model_dir: Path, final_iteration: int, train_rows: list[dict[str, str]]) -> tuple[float, str]:
    timings = read_json(model_dir / "baseline_command_timings.json")
    if timings and "train" in timings:
        return to_float(timings["train"].get("wall_seconds")), "measured_command_wall"
    iteration_ms = [to_float(row.get("iteration_ms")) for row in train_rows if to_float(row.get("iteration_ms")) > 0.0]
    if iteration_ms and final_iteration > 0:
        return final_iteration * (sum(iteration_ms) / len(iteration_ms)) / 1000.0, "estimated_from_logged_iteration_ms"
    return 0.0, "missing"


def parse_baseline(model_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = read_json(model_dir / "baseline_manifest.json") or {}
    config = manifest.get("config", {})
    scene = Path(manifest.get("scene_path", model_dir.parent.name)).name
    eval_rows = read_csv(model_dir / "baseline_eval_metrics.csv")
    train_rows = read_csv(model_dir / "baseline_train_metrics.csv")
    test_rows = [row for row in eval_rows if row.get("split") == "test"] or eval_rows
    final_iteration = max([to_int(row.get("iteration")) for row in test_rows] + [to_int(config.get("iterations"))])
    train_wall, wall_source = baseline_train_wall_seconds(model_dir, final_iteration, train_rows)
    final_eval = max(test_rows, key=lambda row: to_int(row.get("iteration")), default={})
    final_train = max(train_rows, key=lambda row: to_int(row.get("iteration")), default={})
    rendered_metrics = final_render_metrics(model_dir)

    summary = {
        "method": "baseline_3dgs",
        "scene": scene,
        "run_dir": str(model_dir),
        "metric_split": "test",
        "final_iteration": final_iteration,
        "train_wall_seconds": train_wall,
        "wall_time_source": wall_source,
        "final_psnr": rendered_metrics.get("test_psnr", to_float(final_eval.get("psnr"))),
        "final_ssim": rendered_metrics.get("test_ssim", to_float(final_eval.get("ssim"))),
        "final_lpips": rendered_metrics.get("test_lpips", ""),
        "final_gaussians": to_int(final_eval.get("gaussian_count"), to_int(final_train.get("gaussian_count"))),
        "peak_vram_mb": max(
            [to_float(row.get("vram_peak_allocated_mb")) for row in train_rows + eval_rows]
            + [0.0]
        ),
    }

    curve = []
    for row in sorted(test_rows, key=lambda item: to_int(item.get("iteration"))):
        iteration = to_int(row.get("iteration"))
        if final_iteration > 0 and train_wall > 0:
            wall_seconds = train_wall * iteration / final_iteration
        else:
            wall_seconds = 0.0
        curve.append(
            {
                "method": "baseline_3dgs",
                "scene": scene,
                "run_dir": str(model_dir),
                "metric_split": "test",
                "iteration": iteration,
                "wall_seconds": wall_seconds,
                "psnr": to_float(row.get("psnr")),
                "ssim": to_float(row.get("ssim")),
                "gaussian_count": to_int(row.get("gaussian_count")),
                "peak_vram_mb": to_float(row.get("vram_peak_allocated_mb")),
            }
        )
    return summary, curve


def parse_policy(episode_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    metadata = read_json(episode_dir / "agentic_episode_metadata.json") or {}
    scene = metadata.get("scene", episode_dir.parent.name)
    rows = read_csv(episode_dir / "agentic_blocks.csv")
    rendered_metrics = final_render_metrics(episode_dir)
    cumulative = 0.0
    curve = []
    for row in rows:
        elapsed = to_float(row.get("elapsed_seconds"), -1.0)
        if elapsed < 0.0:
            cumulative += to_float(row.get("block_seconds")) + to_float(row.get("validation_seconds"))
            elapsed = cumulative
        else:
            cumulative = elapsed
        curve.append(
            {
                "method": "agentic_policy",
                "scene": scene,
                "run_dir": str(episode_dir),
                "metric_split": "validation_subset",
                "iteration": to_int(row.get("iteration")),
                "wall_seconds": elapsed,
                "psnr": to_float(row.get("validation_psnr")),
                "ssim": to_float(row.get("validation_ssim")),
                "gaussian_count": to_int(row.get("gaussian_count")),
                "peak_vram_mb": to_float(row.get("peak_vram_gb")) * 1024.0,
            }
        )
    final = curve[-1] if curve else {}
    summary = {
        "method": "agentic_policy",
        "scene": scene,
        "run_dir": str(episode_dir),
        "metric_split": "validation_subset",
        "final_iteration": to_int(final.get("iteration")),
        "train_wall_seconds": to_float(final.get("wall_seconds")),
        "wall_time_source": "measured_env_elapsed",
        "final_psnr": rendered_metrics.get("test_psnr", to_float(final.get("psnr"))),
        "final_ssim": rendered_metrics.get("test_ssim", to_float(final.get("ssim"))),
        "final_lpips": rendered_metrics.get("test_lpips", ""),
        "final_gaussians": to_int(final.get("gaussian_count")),
        "peak_vram_mb": max([to_float(row.get("peak_vram_mb")) for row in curve] + [0.0]),
    }
    return summary, curve


def value_at_budget(curve: list[dict[str, Any]], budget: float) -> dict[str, Any] | None:
    eligible = [row for row in curve if to_float(row.get("wall_seconds")) <= budget]
    if eligible:
        return max(eligible, key=lambda row: to_float(row.get("wall_seconds")))
    return min(curve, key=lambda row: to_float(row.get("wall_seconds")), default=None)


def time_to_target(curve: list[dict[str, Any]], target: float) -> float | str:
    for row in sorted(curve, key=lambda item: to_float(item.get("wall_seconds"))):
        if to_float(row.get("psnr")) >= target:
            return to_float(row.get("wall_seconds"))
    return ""


def build_budget_rows(curves: list[list[dict[str, Any]]], budgets: list[float]) -> list[dict[str, Any]]:
    rows = []
    for curve in curves:
        if not curve:
            continue
        for budget in budgets:
            row = value_at_budget(curve, budget)
            if row is None:
                continue
            rows.append(
                {
                    "method": row["method"],
                    "scene": row["scene"],
                    "run_dir": row["run_dir"],
                    "metric_split": row["metric_split"],
                    "budget_seconds": budget,
                    "observed_wall_seconds": row["wall_seconds"],
                    "psnr": row["psnr"],
                    "ssim": row["ssim"],
                    "gaussian_count": row["gaussian_count"],
                    "peak_vram_mb": row["peak_vram_mb"],
                }
            )
    return rows


def build_target_rows(curves: list[list[dict[str, Any]]], targets: list[float]) -> list[dict[str, Any]]:
    rows = []
    for curve in curves:
        if not curve:
            continue
        first = curve[0]
        for target in targets:
            rows.append(
                {
                    "method": first["method"],
                    "scene": first["scene"],
                    "run_dir": first["run_dir"],
                    "metric_split": first["metric_split"],
                    "target_psnr": target,
                    "time_to_target_seconds": time_to_target(curve, target),
                }
            )
    return rows


def build_speedup_rows(target_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baselines = {}
    for row in target_rows:
        if row["method"] != "baseline_3dgs":
            continue
        time_value = to_float(row["time_to_target_seconds"], -1.0)
        if time_value <= 0:
            continue
        key = (row["scene"], row["target_psnr"])
        baselines[key] = min(time_value, baselines.get(key, time_value))

    rows = []
    for row in target_rows:
        if row["method"] != "agentic_policy":
            continue
        policy_time = to_float(row["time_to_target_seconds"], -1.0)
        baseline_time = baselines.get((row["scene"], row["target_psnr"]), 0.0)
        if policy_time <= 0 or baseline_time <= 0:
            speedup = ""
            time_saved = ""
        else:
            speedup = baseline_time / policy_time
            time_saved = baseline_time - policy_time
        rows.append(
            {
                "scene": row["scene"],
                "target_psnr": row["target_psnr"],
                "baseline_time_seconds": baseline_time or "",
                "policy_time_seconds": policy_time if policy_time > 0 else "",
                "speedup_x": speedup,
                "time_saved_seconds": time_saved,
                "policy_run_dir": row["run_dir"],
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate baseline and agentic 3DGS runs into acceleration tables.")
    parser.add_argument("--baseline-root", type=Path, default=PROJECT_ROOT / "outputs" / "3dgs_baseline")
    parser.add_argument("--policy-root", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_eval")
    parser.add_argument("--baseline-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--policy-dirs", nargs="*", type=Path, default=[])
    parser.add_argument("--time-budgets", nargs="+", type=float, default=[30.0, 60.0, 120.0, 300.0])
    parser.add_argument("--target-psnrs", nargs="*", type=float, default=[])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_reports")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    baseline_dirs = [resolve_path(path) for path in args.baseline_dirs]
    policy_dirs = [resolve_path(path) for path in args.policy_dirs]
    if not baseline_dirs:
        baseline_dirs = discover_dirs(resolve_path(args.baseline_root), "baseline_eval_metrics.csv")
    if not policy_dirs:
        policy_dirs = discover_dirs(resolve_path(args.policy_root), "agentic_blocks.csv")

    summaries = []
    curves = []
    for model_dir in baseline_dirs:
        summary, curve = parse_baseline(model_dir)
        summaries.append(summary)
        curves.append(curve)
    for episode_dir in policy_dirs:
        summary, curve = parse_policy(episode_dir)
        summaries.append(summary)
        curves.append(curve)

    output_dir = resolve_path(args.output_dir)
    budget_rows = build_budget_rows(curves, args.time_budgets)
    targets = list(args.target_psnrs)
    if not targets:
        baseline_final_psnr = [to_float(row["final_psnr"]) for row in summaries if row["method"] == "baseline_3dgs"]
        if baseline_final_psnr:
            reference = min(baseline_final_psnr)
            targets = [round(reference - 1.0, 3), round(reference - 0.5, 3), round(reference, 3)]
    target_rows = build_target_rows(curves, targets)
    speedup_rows = build_speedup_rows(target_rows)

    write_csv(output_dir / "run_summary.csv", summaries)
    write_csv(output_dir / "time_budget_curve.csv", budget_rows)
    write_csv(output_dir / "time_to_target.csv", target_rows)
    write_csv(output_dir / "speedup_summary.csv", speedup_rows)
    print(f"Wrote reports to {output_dir}")
    print(f"Baseline runs: {sum(1 for row in summaries if row['method'] == 'baseline_3dgs')}")
    print(f"Policy runs: {sum(1 for row in summaries if row['method'] == 'agentic_policy')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

