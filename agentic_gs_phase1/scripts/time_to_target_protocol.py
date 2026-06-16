from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "outputs" / "agentic_gs_phase1" / "ppo_5scene_fast_15k_long" / "checkpoints" / "latest.pth"
DEFAULT_CONFIG = PROJECT_ROOT / "agentic_gs_phase1" / "configs" / "phase1_fast.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "agentic_gs_phase1_unified_eval"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "agentic_gs_phase1_reports" / "hotdog_time_to_target_ppo5scene_fast_latest"
DEFAULT_TARGETS = [25.0, 27.0, 29.0, 31.0, 33.0, 35.0]
DEFAULT_SAMPLE_ITERATIONS = [100, 250, 500, 750, 1000, 1500, 2000, 2500, 3000, 4000, 5000]
METHOD_LABELS = {
    "fixed_3dgs_same_split": "baseline",
    "baseline": "baseline",
    "agentic_policy": "agentic",
    "agentic": "agentic",
}


@dataclass(frozen=True)
class CurvePoint:
    method: str
    scene: str
    iteration: int
    train_wall_seconds: float
    test_psnr: float
    test_ssim: float
    final_gaussians: int
    model_size_mb: float
    model_size_path: Path
    peak_vram_gb: float
    run_dir: Path
    model_path: Path
    action_sets_csv: Path | None
    action_sets_json: Path | None


@dataclass(frozen=True)
class TargetEstimate:
    method: str
    target_psnr: float
    time_seconds: float | None
    status: str
    iteration_at_or_after: int | None
    psnr_at_or_after: float | None
    previous_iteration: int | None
    previous_psnr: float | None
    run_dir: Path | None
    gaussian_count_at_or_after: int | None
    model_size_mb_at_or_after: float | None
    model_size_path_at_or_after: Path | None
    action_sets_csv: Path | None
    action_sets_json: Path | None


def resolve_path(path: str | Path) -> Path:
    value = Path(path)
    if not value.is_absolute():
        value = PROJECT_ROOT / value
    return value.resolve()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = csv_fieldnames(rows)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_fieldnames(rows: list[dict[str, Any]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    return fieldnames


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def model_artifact_path(model_path: Path) -> Path:
    point_cloud_dir = model_path / "point_cloud"
    if point_cloud_dir.exists():
        return point_cloud_dir
    return model_path


def optional_existing_path(path: Path) -> Path | None:
    return path if path.exists() else None


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def find_subset(summary: dict[str, Any], subset_name: str) -> dict[str, Any] | None:
    for subset in summary.get("eval_subsets", []):
        if subset.get("subset") == subset_name:
            return subset
    return None


def parse_summary(summary_path: Path, scene: str) -> list[CurvePoint]:
    summaries = read_json(summary_path)
    if not isinstance(summaries, list):
        return []
    points: list[CurvePoint] = []
    for item in summaries:
        if item.get("scene") != scene:
            continue
        method = METHOD_LABELS.get(str(item.get("method", "")))
        if method is None:
            continue
        test_subset = find_subset(item, "test")
        if test_subset is None:
            continue
        model_path = resolve_path(item.get("model_path", summary_path.parent))
        model_size_path = model_artifact_path(model_path)
        points.append(
            CurvePoint(
                method=method,
                scene=scene,
                iteration=as_int(item.get("final_iteration")),
                train_wall_seconds=as_float(item.get("train_wall_seconds")),
                test_psnr=as_float(test_subset.get("mean_psnr")),
                test_ssim=as_float(test_subset.get("mean_ssim")),
                final_gaussians=as_int(item.get("final_gaussians")),
                model_size_mb=directory_size_bytes(model_size_path) / (1024.0 * 1024.0),
                model_size_path=model_size_path,
                peak_vram_gb=as_float(item.get("peak_vram_gb")),
                run_dir=summary_path.parent,
                model_path=model_path,
                action_sets_csv=optional_existing_path(model_path / "action_sets.csv"),
                action_sets_json=optional_existing_path(model_path / "action_sets.json"),
            )
        )
    return points


def discover_points(output_root: Path, run_prefix: str, scene: str) -> list[CurvePoint]:
    if not output_root.exists():
        return []
    points: list[CurvePoint] = []
    for summary_path in sorted(output_root.glob(f"{run_prefix}*/unified_eval_summary.json")):
        points.extend(parse_summary(summary_path, scene))

    best_by_key: dict[tuple[str, int], CurvePoint] = {}
    for point in points:
        key = (point.method, point.iteration)
        previous = best_by_key.get(key)
        if previous is None or point.run_dir.stat().st_mtime > previous.run_dir.stat().st_mtime:
            best_by_key[key] = point
    return sorted(best_by_key.values(), key=lambda point: (point.method, point.train_wall_seconds, point.iteration))


def summarize_missing(
    points: list[CurvePoint],
    sample_iterations: list[int],
    methods: list[str],
) -> list[int]:
    present = {(point.method, point.iteration) for point in points}
    missing = []
    for iteration in sample_iterations:
        if any((method, iteration) not in present for method in methods):
            missing.append(iteration)
    return missing


def is_complete_run(run_dir: Path, scene: str, methods: list[str]) -> bool:
    summary_path = run_dir / "unified_eval_summary.json"
    if not summary_path.exists():
        return False
    points = parse_summary(summary_path, scene)
    found = {(point.method, point.iteration) for point in points}
    return bool(points) and all(any(method == item[0] for item in found) for method in methods)


def run_sample(
    python_bin: str,
    checkpoint: Path,
    config: Path,
    scene: str,
    output_root: Path,
    run_prefix: str,
    iteration: int,
    methods: list[str],
    validation_cameras: int,
    max_render_views: int,
    force_full_iterations: bool,
    skip_render_images: bool,
    dry_run: bool,
) -> None:
    method_args = ["baseline" if method == "baseline" else "agentic" for method in methods]
    run_name = f"{run_prefix}_{iteration}_min50k"
    command = [
        python_bin,
        str(PROJECT_ROOT / "agentic_gs_phase1" / "scripts" / "run_unified_eval.py"),
        "--checkpoint",
        str(checkpoint),
        "--config",
        str(config),
        "--scenes",
        scene,
        "--methods",
        *method_args,
        "--eval-subsets",
        "validation",
        "test",
        "--run-name",
        run_name,
        "--output-root",
        str(output_root),
        "--max-episode-iterations",
        str(iteration),
        "--validation-cameras",
        str(validation_cameras),
        "--max-render-views",
        str(max_render_views),
    ]
    if force_full_iterations:
        command.append("--force-full-iterations")
    if skip_render_images:
        command.append("--skip-render-images")

    print(" ".join(f'"{part}"' if " " in part else part for part in command))
    if dry_run:
        return
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)


def interpolate_time(previous: CurvePoint, current: CurvePoint, target: float) -> float:
    psnr_delta = current.test_psnr - previous.test_psnr
    if psnr_delta <= 0:
        return current.train_wall_seconds
    ratio = (target - previous.test_psnr) / psnr_delta
    ratio = max(0.0, min(1.0, ratio))
    return previous.train_wall_seconds + ratio * (current.train_wall_seconds - previous.train_wall_seconds)


def pareto_envelope(points: list[CurvePoint], method: str) -> list[CurvePoint]:
    method_points = sorted(
        [point for point in points if point.method == method],
        key=lambda point: (point.train_wall_seconds, -point.test_psnr, point.iteration),
    )
    envelope: list[CurvePoint] = []
    best_psnr = float("-inf")
    for point in method_points:
        if point.test_psnr > best_psnr:
            envelope.append(point)
            best_psnr = point.test_psnr
    return envelope


def estimate_target(points: list[CurvePoint], method: str, target: float) -> TargetEstimate:
    method_points = pareto_envelope(points, method)
    if not method_points:
        return TargetEstimate(
            method=method,
            target_psnr=target,
            time_seconds=None,
            status="missing_curve",
            iteration_at_or_after=None,
            psnr_at_or_after=None,
            previous_iteration=None,
            previous_psnr=None,
            run_dir=None,
            gaussian_count_at_or_after=None,
            model_size_mb_at_or_after=None,
            model_size_path_at_or_after=None,
            action_sets_csv=None,
            action_sets_json=None,
        )

    previous: CurvePoint | None = None
    for point in method_points:
        if point.test_psnr >= target:
            if previous is None:
                return TargetEstimate(
                    method=method,
                    target_psnr=target,
                    time_seconds=point.train_wall_seconds,
                    status="upper_bound_first_sample",
                    iteration_at_or_after=point.iteration,
                    psnr_at_or_after=point.test_psnr,
                    previous_iteration=None,
                    previous_psnr=None,
                    run_dir=point.run_dir,
                    gaussian_count_at_or_after=point.final_gaussians,
                    model_size_mb_at_or_after=point.model_size_mb,
                    model_size_path_at_or_after=point.model_size_path,
                    action_sets_csv=point.action_sets_csv,
                    action_sets_json=point.action_sets_json,
                )
            if previous.test_psnr >= target:
                status = "observed_sample"
                seconds = point.train_wall_seconds
            else:
                status = "interpolated_between_samples"
                seconds = interpolate_time(previous, point, target)
            return TargetEstimate(
                method=method,
                target_psnr=target,
                time_seconds=seconds,
                status=status,
                iteration_at_or_after=point.iteration,
                psnr_at_or_after=point.test_psnr,
                previous_iteration=previous.iteration,
                previous_psnr=previous.test_psnr,
                run_dir=point.run_dir,
                gaussian_count_at_or_after=point.final_gaussians,
                model_size_mb_at_or_after=point.model_size_mb,
                model_size_path_at_or_after=point.model_size_path,
                action_sets_csv=point.action_sets_csv,
                action_sets_json=point.action_sets_json,
            )
        previous = point

    last = method_points[-1]
    return TargetEstimate(
        method=method,
        target_psnr=target,
        time_seconds=None,
        status="not_reached",
        iteration_at_or_after=last.iteration,
        psnr_at_or_after=last.test_psnr,
        previous_iteration=None,
        previous_psnr=None,
        run_dir=last.run_dir,
        gaussian_count_at_or_after=last.final_gaussians,
        model_size_mb_at_or_after=last.model_size_mb,
        model_size_path_at_or_after=last.model_size_path,
        action_sets_csv=last.action_sets_csv,
        action_sets_json=last.action_sets_json,
    )


def exact_enough(estimate: TargetEstimate) -> bool:
    return estimate.status in {"observed_sample", "interpolated_between_samples"}


def build_curve_rows(points: list[CurvePoint]) -> list[dict[str, Any]]:
    envelope_keys = {
        (point.method, point.iteration, point.run_dir)
        for method in sorted({point.method for point in points})
        for point in pareto_envelope(points, method)
    }
    return [
        {
            "scene": point.scene,
            "method": point.method,
            "iteration": point.iteration,
            "train_wall_seconds": f"{point.train_wall_seconds:.6f}",
            "test_psnr": f"{point.test_psnr:.6f}",
            "test_ssim": f"{point.test_ssim:.6f}",
            "final_gaussians": point.final_gaussians,
            "model_size_mb": f"{point.model_size_mb:.6f}",
            "model_size_path": str(point.model_size_path),
            "peak_vram_gb": f"{point.peak_vram_gb:.6f}",
            "pareto_envelope": "yes" if (point.method, point.iteration, point.run_dir) in envelope_keys else "no",
            "run_dir": str(point.run_dir),
            "model_path": str(point.model_path),
            "action_sets_csv": "" if point.action_sets_csv is None else str(point.action_sets_csv),
            "action_sets_json": "" if point.action_sets_json is None else str(point.action_sets_json),
        }
        for point in sorted(points, key=lambda item: (item.method, item.iteration))
    ]


def build_prefixed_artifact_rows(points: list[CurvePoint], filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for point in sorted(points, key=lambda item: (item.method, item.iteration)):
        source_path = point.model_path / filename
        if not source_path.exists():
            continue
        for raw_row in read_csv_rows(source_path):
            rows.append(
                {
                    "sample_scene": point.scene,
                    "sample_method": point.method,
                    "sample_iteration": point.iteration,
                    "sample_train_wall_seconds": f"{point.train_wall_seconds:.6f}",
                    "sample_test_psnr": f"{point.test_psnr:.6f}",
                    "sample_final_gaussians": point.final_gaussians,
                    "sample_model_size_mb": f"{point.model_size_mb:.6f}",
                    "sample_model_size_path": str(point.model_size_path),
                    "sample_run_dir": str(point.run_dir),
                    "sample_model_path": str(point.model_path),
                    "source_csv": str(source_path),
                    **raw_row,
                }
            )
    return rows


def build_long_rows(estimates: list[TargetEstimate]) -> list[dict[str, Any]]:
    rows = []
    for estimate in estimates:
        rows.append(
            {
                "method": estimate.method,
                "target_psnr": f"{estimate.target_psnr:.3f}",
                "time_seconds": "" if estimate.time_seconds is None else f"{estimate.time_seconds:.6f}",
                "status": estimate.status,
                "iteration_at_or_after": "" if estimate.iteration_at_or_after is None else estimate.iteration_at_or_after,
                "psnr_at_or_after": "" if estimate.psnr_at_or_after is None else f"{estimate.psnr_at_or_after:.6f}",
                "previous_iteration": "" if estimate.previous_iteration is None else estimate.previous_iteration,
                "previous_psnr": "" if estimate.previous_psnr is None else f"{estimate.previous_psnr:.6f}",
                "run_dir": "" if estimate.run_dir is None else str(estimate.run_dir),
                "gaussian_count_at_or_after": ""
                if estimate.gaussian_count_at_or_after is None
                else estimate.gaussian_count_at_or_after,
                "model_size_mb_at_or_after": ""
                if estimate.model_size_mb_at_or_after is None
                else f"{estimate.model_size_mb_at_or_after:.6f}",
                "model_size_path_at_or_after": ""
                if estimate.model_size_path_at_or_after is None
                else str(estimate.model_size_path_at_or_after),
                "action_sets_csv": "" if estimate.action_sets_csv is None else str(estimate.action_sets_csv),
                "action_sets_json": "" if estimate.action_sets_json is None else str(estimate.action_sets_json),
            }
        )
    return rows


def build_comparison_rows(scene: str, targets: list[float], estimates: list[TargetEstimate]) -> list[dict[str, Any]]:
    by_key = {(estimate.method, estimate.target_psnr): estimate for estimate in estimates}
    rows = []
    for target in targets:
        baseline = by_key.get(("baseline", target))
        agentic = by_key.get(("agentic", target))
        baseline_time = baseline.time_seconds if baseline is not None else None
        agentic_time = agentic.time_seconds if agentic is not None else None
        claim_ready = bool(baseline and agentic and exact_enough(baseline) and exact_enough(agentic))
        if claim_ready and baseline_time is not None and agentic_time is not None and agentic_time > 0:
            speedup = baseline_time / agentic_time
            time_saved = baseline_time - agentic_time
        else:
            speedup = None
            time_saved = None
        rows.append(
            {
                "scene": scene,
                "target_psnr": f"{target:.3f}",
                "baseline_time_seconds": "" if baseline_time is None else f"{baseline_time:.6f}",
                "baseline_status": "" if baseline is None else baseline.status,
                "baseline_iteration_at_or_after": "" if baseline is None or baseline.iteration_at_or_after is None else baseline.iteration_at_or_after,
                "baseline_gaussians_at_or_after": ""
                if baseline is None or baseline.gaussian_count_at_or_after is None
                else baseline.gaussian_count_at_or_after,
                "baseline_model_size_mb_at_or_after": ""
                if baseline is None or baseline.model_size_mb_at_or_after is None
                else f"{baseline.model_size_mb_at_or_after:.6f}",
                "baseline_model_size_path_at_or_after": ""
                if baseline is None or baseline.model_size_path_at_or_after is None
                else str(baseline.model_size_path_at_or_after),
                "baseline_action_sets_csv": ""
                if baseline is None or baseline.action_sets_csv is None
                else str(baseline.action_sets_csv),
                "baseline_action_sets_json": ""
                if baseline is None or baseline.action_sets_json is None
                else str(baseline.action_sets_json),
                "agentic_time_seconds": "" if agentic_time is None else f"{agentic_time:.6f}",
                "agentic_status": "" if agentic is None else agentic.status,
                "agentic_iteration_at_or_after": "" if agentic is None or agentic.iteration_at_or_after is None else agentic.iteration_at_or_after,
                "agentic_gaussians_at_or_after": ""
                if agentic is None or agentic.gaussian_count_at_or_after is None
                else agentic.gaussian_count_at_or_after,
                "agentic_model_size_mb_at_or_after": ""
                if agentic is None or agentic.model_size_mb_at_or_after is None
                else f"{agentic.model_size_mb_at_or_after:.6f}",
                "agentic_model_size_path_at_or_after": ""
                if agentic is None or agentic.model_size_path_at_or_after is None
                else str(agentic.model_size_path_at_or_after),
                "agentic_action_sets_csv": ""
                if agentic is None or agentic.action_sets_csv is None
                else str(agentic.action_sets_csv),
                "agentic_action_sets_json": ""
                if agentic is None or agentic.action_sets_json is None
                else str(agentic.action_sets_json),
                "speedup_x": "" if speedup is None else f"{speedup:.6f}",
                "time_saved_seconds": "" if time_saved is None else f"{time_saved:.6f}",
                "claim_ready": "yes" if claim_ready else "no",
            }
        )
    return rows


def write_markdown_report(
    path: Path,
    points: list[CurvePoint],
    comparison_rows: list[dict[str, Any]],
    checkpoint: Path,
    config: Path,
    scene: str,
    sample_iterations: list[int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    scene_title = scene.replace("_", " ").title()
    lines = [
        f"# {scene_title} Time-to-Target PSNR Protocol",
        "",
        "Primary criterion: the method that claims speed must reach the same held-out test PSNR in less training wall-clock time.",
        "",
        "## Inputs",
        "",
        f"- Policy checkpoint: `{checkpoint}`",
        f"- Evaluation config: `{config}`",
        f"- Scene: `{scene}`",
        "- Quality metric: full dataset test PSNR from `run_unified_eval.py --eval-subsets validation test`",
        "- Primary clock: `train_wall_seconds`; offline test rendering and metric time are excluded.",
        f"- Sample iteration budgets: `{', '.join(str(item) for item in sample_iterations)}`",
        "",
        "## Interpretation",
        "",
        "- `interpolated_between_samples`: target crossing is linearly interpolated between two measured test-PSNR samples.",
        "- `upper_bound_first_sample`: the first available sample already exceeds the target, so the real crossing time is earlier or equal.",
        "- `not_reached`: the sampled curve never reached the target.",
        "- `claim_ready` is `yes` only when both methods have observed or interpolated crossings rather than only bounds.",
        "- Crossing estimates use the PSNR-time Pareto envelope; dominated samples are kept in `sampled_test_curve.csv` but are not used for interpolation.",
        "- `sampled_test_curve.csv` and `time_to_target_by_method.csv` include Gaussian count, saved point-cloud model size, and action-set artifact paths.",
        "- `sampled_action_sets.csv` consolidates the selected policy actions; `sampled_training_blocks.csv` consolidates per-block Gaussian counts and timing.",
        "",
        "## Comparison",
        "",
        "| Target PSNR | Baseline seconds | Agentic seconds | Speedup x | Time saved | Status | Claim ready |",
        "|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in comparison_rows:
        status = f"baseline={row['baseline_status']}; agentic={row['agentic_status']}"
        lines.append(
            "| {target_psnr} | {baseline_time_seconds} | {agentic_time_seconds} | {speedup_x} | "
            "{time_saved_seconds} | {status} | {claim_ready} |".format(**row, status=status)
        )
    lines.extend(
        [
            "",
            "## Sampled Test Curve",
            "",
            "| Method | Iteration | Train seconds | Test PSNR | Test SSIM | Gaussians | Model MB | Envelope |",
            "|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    envelope_keys = {
        (point.method, point.iteration, point.run_dir)
        for method in sorted({point.method for point in points})
        for point in pareto_envelope(points, method)
    }
    for point in sorted(points, key=lambda item: (item.method, item.iteration)):
        lines.append(
            f"| {point.method} | {point.iteration} | {point.train_wall_seconds:.3f} | "
            f"{point.test_psnr:.3f} | {point.test_ssim:.4f} | {point.final_gaussians} | "
            f"{point.model_size_mb:.3f} | {'yes' if (point.method, point.iteration, point.run_dir) in envelope_keys else 'no'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and aggregate time-to-target PSNR evaluation.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=str, default="hotdog")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-prefix", type=str, default="unified_ppo5scene_fast_latest_hotdog")
    parser.add_argument("--sample-iterations", nargs="+", type=int, default=DEFAULT_SAMPLE_ITERATIONS)
    parser.add_argument("--targets", nargs="+", type=float, default=DEFAULT_TARGETS)
    parser.add_argument("--methods", nargs="+", choices=["baseline", "agentic"], default=["baseline", "agentic"])
    parser.add_argument("--validation-cameras", type=int, default=6)
    parser.add_argument("--max-render-views", type=int, default=0)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--run-missing", action="store_true", help="Run missing sampled iteration budgets before reporting.")
    parser.add_argument("--force-full-iterations", action="store_true", help="Force the policy to consume each sampled iteration budget.")
    parser.add_argument("--skip-render-images", action="store_true", help="Compute metrics without saving reconstructed PNG images.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = resolve_path(args.checkpoint)
    config = resolve_path(args.config)
    output_root = resolve_path(args.output_root)
    report_dir = resolve_path(args.report_dir)
    sample_iterations = sorted({int(value) for value in args.sample_iterations if int(value) > 0})
    targets = sorted({float(value) for value in args.targets})
    methods = list(dict.fromkeys(args.methods))

    if args.run_missing:
        existing = discover_points(output_root, args.run_prefix, args.scene)
        missing = summarize_missing(existing, sample_iterations, methods)
        for iteration in missing:
            run_dir = output_root / f"{args.run_prefix}_{iteration}_min50k"
            if is_complete_run(run_dir, args.scene, methods):
                continue
            run_sample(
                python_bin=args.python_bin,
                checkpoint=checkpoint,
                config=config,
                scene=args.scene,
                output_root=output_root,
                run_prefix=args.run_prefix,
                iteration=iteration,
                methods=methods,
                validation_cameras=int(args.validation_cameras),
                max_render_views=int(args.max_render_views),
                force_full_iterations=bool(args.force_full_iterations),
                skip_render_images=bool(args.skip_render_images),
                dry_run=bool(args.dry_run),
            )

    points = discover_points(output_root, args.run_prefix, args.scene)
    selected_points = [point for point in points if point.method in methods and point.iteration in sample_iterations]
    estimates = [
        estimate_target(selected_points, method, target)
        for method in methods
        for target in targets
    ]
    comparison_rows = build_comparison_rows(args.scene, targets, estimates)

    report_dir.mkdir(parents=True, exist_ok=True)
    write_csv(report_dir / "sampled_test_curve.csv", build_curve_rows(selected_points))
    write_csv(report_dir / "sampled_action_sets.csv", build_prefixed_artifact_rows(selected_points, "action_sets.csv"))
    write_csv(report_dir / "sampled_training_blocks.csv", build_prefixed_artifact_rows(selected_points, "agentic_blocks.csv"))
    write_csv(report_dir / "time_to_target_by_method.csv", build_long_rows(estimates))
    write_csv(report_dir / "time_to_target_comparison.csv", comparison_rows)
    write_markdown_report(
        report_dir / f"{args.scene}_time_to_target_protocol.md",
        selected_points,
        comparison_rows,
        checkpoint,
        config,
        args.scene,
        sample_iterations,
    )
    print(f"Wrote {args.scene} time-to-target report to {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
