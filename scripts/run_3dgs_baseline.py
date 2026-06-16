from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "baseline_3dgs.json"


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return json.load(config_file)


def resolve_path(value: str | Path, base: Path = ROOT) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def discover_scenes(archive_root: Path) -> list[str]:
    if not archive_root.exists():
        return []
    return [
        item.name
        for item in sorted(archive_root.iterdir())
        if item.is_dir() and (item / "transforms_train.json").exists() and (item / "transforms_test.json").exists()
    ]


def resolve_scene(scene: str, archive_root: Path) -> Path:
    scene_path = Path(scene)
    if scene_path.exists():
        return scene_path.resolve()
    candidate = archive_root / scene
    if candidate.exists():
        return candidate.resolve()
    available = discover_scenes(archive_root)
    hint = f" Available scenes: {', '.join(available)}." if available else ""
    raise FileNotFoundError(f"Could not find scene '{scene}' under {archive_root}.{hint}")


def add_list_args(command: list[str], flag: str, values: list[int]) -> None:
    if values:
        command.append(flag)
        command.extend(str(value) for value in values)


def normalize_iteration_lists(config: dict[str, Any]) -> None:
    final_iteration = int(config["iterations"])
    for key in ["test_iterations", "save_iterations", "checkpoint_iterations"]:
        values = sorted({int(value) for value in config.get(key, []) if 0 < int(value) <= final_iteration})
        if key in {"test_iterations", "save_iterations"} and final_iteration not in values:
            values.append(final_iteration)
        config[key] = values


def build_train_command(python_bin: str, gs_dir: Path, scene_path: Path, model_path: Path, config: dict[str, Any]) -> list[str]:
    command = [
        python_bin,
        str(gs_dir / "train.py"),
        "-s",
        str(scene_path),
        "-m",
        str(model_path),
        "--iterations",
        str(config["iterations"]),
        "--data_device",
        str(config["data_device"]),
        "--baseline_log_interval",
        str(config["baseline_log_interval"]),
        "--baseline_histogram_interval",
        str(config["baseline_histogram_interval"]),
    ]

    if config.get("eval", True):
        command.append("--eval")
    if config.get("white_background", True):
        command.append("--white_background")
    if config.get("disable_viewer", True):
        command.append("--disable_viewer")

    resolution = int(config.get("resolution", -1))
    if resolution != -1:
        command.extend(["-r", str(resolution)])

    add_list_args(command, "--test_iterations", [int(value) for value in config.get("test_iterations", [])])
    add_list_args(command, "--save_iterations", [int(value) for value in config.get("save_iterations", [])])
    add_list_args(command, "--checkpoint_iterations", [int(value) for value in config.get("checkpoint_iterations", [])])
    command.extend(str(value) for value in config.get("extra_train_args", []))
    return command


def build_render_command(python_bin: str, gs_dir: Path, model_path: Path, config: dict[str, Any]) -> list[str]:
    command = [python_bin, str(gs_dir / "render.py"), "-m", str(model_path)]
    if config.get("skip_train_render", True):
        command.append("--skip_train")
    if config.get("skip_test_render", False):
        command.append("--skip_test")
    return command


def build_metrics_command(python_bin: str, gs_dir: Path, model_path: Path) -> list[str]:
    return [python_bin, str(gs_dir / "metrics.py"), "-m", str(model_path)]


def run_command(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> dict[str, Any]:
    printable = " ".join(f'"{part}"' if " " in part else part for part in command)
    print(printable)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().isoformat(timespec="seconds")
    start = time.perf_counter()
    return_code = 0
    with log_path.open("w", encoding="utf-8", errors="replace") as log_file:
        log_file.write(printable + "\n\n")
        if dry_run:
            return {
                "command": command,
                "started_at": started_at,
                "finished_at": datetime.now().isoformat(timespec="seconds"),
                "wall_seconds": 0.0,
                "return_code": 0,
                "dry_run": True,
            }
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        return_code = process.wait()
        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, command)
    return {
        "command": command,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
        "wall_seconds": time.perf_counter() - start,
        "return_code": return_code,
        "dry_run": False,
    }


def write_manifest(model_path: Path, scene_path: Path, config: dict[str, Any], commands: dict[str, list[str]]) -> None:
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "scene_path": str(scene_path),
        "model_path": str(model_path),
        "config": config,
        "commands": commands,
    }
    model_path.mkdir(parents=True, exist_ok=True)
    with (model_path / "baseline_manifest.json").open("w", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2)


def write_command_timings(model_path: Path, timings: dict[str, dict[str, Any]]) -> None:
    with (model_path / "baseline_command_timings.json").open("w", encoding="utf-8") as timings_file:
        json.dump(timings, timings_file, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed 3D Gaussian Splatting baseline on archive scenes.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--scene", type=str)
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--render", dest="render_after_train", action="store_true")
    parser.add_argument("--no-render", dest="render_after_train", action="store_false")
    parser.add_argument("--metrics", dest="metrics_after_render", action="store_true")
    parser.add_argument("--no-metrics", dest="metrics_after_render", action="store_false")
    parser.set_defaults(render_after_train=None, metrics_after_render=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(resolve_path(args.config))

    if args.scene is not None:
        config["scene"] = args.scene
    if args.iterations is not None:
        config["iterations"] = args.iterations
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.render_after_train is not None:
        config["render_after_train"] = args.render_after_train
    if args.metrics_after_render is not None:
        config["metrics_after_render"] = args.metrics_after_render
    normalize_iteration_lists(config)

    gs_dir = resolve_path(config["official_3dgs_dir"])
    archive_root = resolve_path(config["archive_root"])
    output_root = resolve_path(config["output_root"])
    scene_path = resolve_scene(config["scene"], archive_root)
    model_path = resolve_path(args.model_path) if args.model_path else (output_root / scene_path.name / f"fixed_{config['iterations']}").resolve()

    for required in [gs_dir / "train.py", gs_dir / "render.py", gs_dir / "metrics.py"]:
        if not required.exists():
            raise FileNotFoundError(f"Missing official 3DGS file: {required}")

    commands = {
        "train": build_train_command(args.python_bin, gs_dir, scene_path, model_path, config),
        "render": build_render_command(args.python_bin, gs_dir, model_path, config),
        "metrics": build_metrics_command(args.python_bin, gs_dir, model_path),
    }
    write_manifest(model_path, scene_path, config, commands)

    log_dir = model_path / "logs"
    timings = {}
    if not args.skip_train:
        timings["train"] = run_command(commands["train"], gs_dir, log_dir / "train.log", args.dry_run)
    if config.get("render_after_train", True):
        timings["render"] = run_command(commands["render"], gs_dir, log_dir / "render.log", args.dry_run)
    if config.get("metrics_after_render", True):
        timings["metrics"] = run_command(commands["metrics"], gs_dir, log_dir / "metrics.log", args.dry_run)
    write_command_timings(model_path, timings)

    print(f"Baseline output: {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
