from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action
from agentic_gs_phase1.policies import ActorCritic

from gaussian_renderer import render
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from agentic_gs_phase1.envs.gs_env import SPARSE_ADAM_AVAILABLE
except Exception:
    SPARSE_ADAM_AVAILABLE = False

try:
    from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

    LPIPS_AVAILABLE = True
except Exception:
    LearnedPerceptualImagePatchSimilarity = None
    LPIPS_AVAILABLE = False


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge_config(result[key], value)
        else:
            result[key] = value
    return result


def make_policy(config: dict[str, Any], obs_dim: int, device: torch.device) -> ActorCritic:
    policy_cfg = config.get("policy", {})
    policy = ActorCritic(
        obs_dim=obs_dim,
        hidden_width=int(policy_cfg.get("hidden_width", 256)),
        hidden_layers=int(policy_cfg.get("hidden_layers", 3)),
        activation=str(policy_cfg.get("activation", "gelu")),
    )
    return policy.to(device)


def stock_reset_due(env: AgenticGSEnv) -> bool:
    if env.iteration <= 0:
        return False
    if env.dataset.white_background and env.iteration == int(env.opt.densify_from_iter):
        return True
    return env.iteration % int(env.opt.opacity_reset_interval) == 0


def fixed_3dgs_action(env: AgenticGSEnv) -> dict[str, Any]:
    action = default_action()
    action["discrete"] = dict(action["discrete"])
    action["discrete"]["block_steps"] = DISCRETE_ACTIONS["block_steps"].index(100)
    action["discrete"]["densify_mode"] = DISCRETE_ACTIONS["densify_mode"].index("default")
    action["discrete"]["densification_interval"] = DISCRETE_ACTIONS["densification_interval"].index(100)
    action["discrete"]["prune_mode"] = DISCRETE_ACTIONS["prune_mode"].index("opacity_and_size")
    action["discrete"]["opacity_reset"] = DISCRETE_ACTIONS["opacity_reset"].index(
        "force_reset" if stock_reset_due(env) else "no_reset"
    )
    action["discrete"]["stop"] = DISCRETE_ACTIONS["stop"].index("continue")
    return action


def camera_names(cameras: list[Any]) -> list[str]:
    return [camera.image_name for camera in cameras]


def get_subset_cameras(env: AgenticGSEnv, subset: str) -> list[Any]:
    if subset == "validation":
        return list(env.validation_cameras)
    if subset == "train":
        return list(env.train_cameras)
    if subset == "test":
        return list(env.scene.getTestCameras())
    raise ValueError(f"Unknown subset: {subset}")


def make_lpips_metric(device: torch.device):
    if not LPIPS_AVAILABLE:
        return None
    try:
        metric = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True)
        return metric.to(device).eval()
    except Exception as exc:
        print(f"LPIPS unavailable for this run: {exc}")
        return None


@torch.no_grad()
def render_and_score_subset(
    env: AgenticGSEnv,
    method_dir: Path,
    subset: str,
    max_render_views: int = 0,
    save_render_images: bool = True,
) -> dict[str, Any]:
    cameras = get_subset_cameras(env, subset)
    if max_render_views > 0:
        cameras = cameras[:max_render_views]
    renders_dir = method_dir / "reconstructed_views" / subset / "renders"
    gt_dir = method_dir / "reconstructed_views" / subset / "gt"
    if save_render_images:
        renders_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    start = time.perf_counter()
    lpips_metric = make_lpips_metric(torch.device("cuda"))
    for idx, viewpoint in enumerate(cameras):
        rendering = torch.clamp(
            render(
                viewpoint,
                env.gaussians,
                env.pipe,
                env.background,
                use_trained_exp=env.dataset.train_test_exp,
                separate_sh=SPARSE_ADAM_AVAILABLE,
            )["render"],
            0.0,
            1.0,
        )
        gt = torch.clamp(viewpoint.original_image[0:3, :, :].to("cuda"), 0.0, 1.0)
        image_name = f"{idx:05d}_{viewpoint.image_name}.png"
        if save_render_images:
            torchvision.utils.save_image(rendering, renders_dir / image_name)
            torchvision.utils.save_image(gt, gt_dir / image_name)
        lpips_value = None
        if lpips_metric is not None:
            lpips_metric.reset()
            lpips_value = float(lpips_metric(rendering.unsqueeze(0), gt.unsqueeze(0)).detach().item())
        rows.append(
            {
                "subset": subset,
                "view_index": idx,
                "image_name": viewpoint.image_name,
                "psnr": float(psnr(rendering.unsqueeze(0), gt.unsqueeze(0)).mean().item()),
                "ssim": float(ssim(rendering, gt).item()),
                "l1": float(l1_loss(rendering, gt).item()),
                "lpips": lpips_value,
            }
        )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    render_seconds = time.perf_counter() - start

    metrics_path = method_dir / f"eval_metrics_{subset}.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["subset", "view_index", "image_name", "psnr", "ssim", "l1", "lpips"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    valid_lpips = [row["lpips"] for row in rows if row["lpips"] is not None and np.isfinite(row["lpips"])]
    summary = {
        "subset": subset,
        "view_count": len(rows),
        "mean_psnr": float(np.mean([row["psnr"] for row in rows])) if rows else 0.0,
        "mean_ssim": float(np.mean([row["ssim"] for row in rows])) if rows else 0.0,
        "mean_l1": float(np.mean([row["l1"] for row in rows])) if rows else 0.0,
        "mean_lpips": float(np.mean(valid_lpips)) if valid_lpips else None,
        "lpips_available": bool(valid_lpips),
        "render_seconds": render_seconds,
        "render_seconds_per_view": render_seconds / max(1, len(rows)),
        "metrics_csv": str(metrics_path),
        "renders_dir": str(renders_dir) if save_render_images else "",
        "gt_dir": str(gt_dir) if save_render_images else "",
        "render_images_saved": bool(save_render_images),
    }
    return summary


def save_action_trace(method_dir: Path, actions: list[dict[str, Any]]) -> None:
    json_path = method_dir / "action_sets.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(actions, f, indent=2)
    csv_path = method_dir / "action_sets.csv"
    if not actions:
        csv_path.write_text("", encoding="utf-8")
        return
    keys = sorted({key for row in actions for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(actions)


def train_fixed_baseline(env: AgenticGSEnv) -> list[dict[str, Any]]:
    actions = []
    done = False
    while not done:
        action = fixed_3dgs_action(env)
        _, reward, done, info = env.step(action)
        action_row = {
            "method": "fixed_3dgs_same_split",
            "block": info["block_index"],
            "iteration": info["iteration"],
            "reward": reward,
            **info["action"],
        }
        actions.append(action_row)
    return actions


def train_agentic(
    env: AgenticGSEnv,
    policy: ActorCritic,
    device: torch.device,
    deterministic: bool,
    force_full_iterations: bool,
) -> list[dict[str, Any]]:
    actions = []
    done = False
    obs = env._build_observation()
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, _, _ = policy.act(obs_tensor, deterministic=deterministic)
        if force_full_iterations:
            action["discrete"]["stop"] = DISCRETE_ACTIONS["stop"].index("continue")
        obs, reward, done, info = env.step(action)
        action_row = {
            "method": "agentic_policy",
            "block": info["block_index"],
            "iteration": info["iteration"],
            "reward": reward,
            **info["action"],
        }
        actions.append(action_row)
    return actions


def run_method(
    method: str,
    scene: str,
    scene_index: int,
    config: dict[str, Any],
    run_dir: Path,
    policy: ActorCritic | None,
    device: torch.device,
    eval_subsets: list[str],
    max_render_views: int,
    save_render_images: bool,
    deterministic: bool,
    force_full_iterations: bool,
) -> dict[str, Any]:
    method_dir = run_dir / method
    env = AgenticGSEnv(config, run_dir=method_dir, seed=int(config.get("seed", 0)) + scene_index * 1000)
    env.reset(scene, episode_id=0)
    if method == "baseline":
        actions = train_fixed_baseline(env)
        method_name = "fixed_3dgs_same_split"
    elif method == "agentic":
        if policy is None:
            raise ValueError("Agentic evaluation requires a policy checkpoint.")
        actions = train_agentic(env, policy, device, deterministic, force_full_iterations)
        method_name = "agentic_policy"
    else:
        raise ValueError(f"Unknown method: {method}")

    env.save_model()
    output_episode_dir = Path(env.model_path)
    save_action_trace(output_episode_dir, actions)
    subset_summaries = [
        render_and_score_subset(env, output_episode_dir, subset, max_render_views, save_render_images)
        for subset in eval_subsets
    ]
    summary = {
        "method": method_name,
        "scene": scene,
        "model_path": str(output_episode_dir),
        "final_iteration": env.iteration,
        "train_wall_seconds": env.elapsed_seconds,
        "final_gaussians": int(env.gaussians.get_xyz.shape[0]),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0,
        "train_camera_names": camera_names(env.train_cameras),
        "validation_camera_names": camera_names(env.validation_cameras),
        "eval_subsets": subset_summaries,
        "action_sets_json": str(output_episode_dir / "action_sets.json"),
        "action_sets_csv": str(output_episode_dir / "action_sets.csv"),
    }
    with (output_episode_dir / "unified_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    env.close()
    return summary


def assert_same_split(scene_summaries: list[dict[str, Any]]) -> None:
    if len(scene_summaries) < 2:
        return
    reference_train = scene_summaries[0]["train_camera_names"]
    reference_validation = scene_summaries[0]["validation_camera_names"]
    for summary in scene_summaries[1:]:
        if summary["train_camera_names"] != reference_train:
            raise RuntimeError(f"Training split mismatch for scene {summary['scene']}")
        if summary["validation_camera_names"] != reference_validation:
            raise RuntimeError(f"Validation split mismatch for scene {summary['scene']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed 3DGS and agentic 3DGS with identical train/eval subsets.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "agentic_gs_phase1" / "configs" / "phase1_eval.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--scenes", nargs="+", default=["hotdog", "drums"])
    parser.add_argument("--methods", nargs="+", choices=["baseline", "agentic"], default=["baseline", "agentic"])
    parser.add_argument("--eval-subsets", nargs="+", choices=["validation", "test", "train"], default=["validation"])
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs" / "agentic_gs_phase1_unified_eval")
    parser.add_argument("--max-episode-iterations", type=int, default=None)
    parser.add_argument("--validation-cameras", type=int, default=None)
    parser.add_argument("--max-render-views", type=int, default=0)
    parser.add_argument("--skip-render-images", action="store_true")
    parser.add_argument("--force-full-iterations", action="store_true")
    parser.add_argument("--stochastic-policy", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Unified 3DGS evaluation requires CUDA.")

    config = load_json(args.config)
    if args.max_episode_iterations is not None:
        config["max_episode_iterations"] = int(args.max_episode_iterations)
        config.setdefault("optimization", {})["position_lr_max_steps"] = int(args.max_episode_iterations)
    if args.validation_cameras is not None:
        config["validation_cameras"] = int(args.validation_cameras)
    config["output_root"] = str(args.output_root)

    device = torch.device("cuda")
    policy = None
    if "agentic" in args.methods:
        if args.checkpoint is None:
            raise ValueError("--checkpoint is required when method 'agentic' is requested.")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        policy_config = merge_config(checkpoint.get("config", {}), config)
        obs_dim = int(checkpoint.get("obs_dim", len(AgenticGSEnv.observation_names)))
        policy = make_policy(policy_config, obs_dim, device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()

    run_name = args.run_name or time.strftime("unified_%Y%m%d_%H%M%S")
    output_root = resolve_path(args.output_root)
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    all_summaries = []
    for scene_index, scene in enumerate(args.scenes):
        scene_summaries = []
        for method in args.methods:
            print(f"[{scene}] running {method}")
            summary = run_method(
                method=method,
                scene=scene,
                scene_index=scene_index,
                config=config,
                run_dir=run_dir / scene,
                policy=policy,
                device=device,
                eval_subsets=list(args.eval_subsets),
                max_render_views=int(args.max_render_views),
                save_render_images=not bool(args.skip_render_images),
                deterministic=not args.stochastic_policy,
                force_full_iterations=bool(args.force_full_iterations),
            )
            scene_summaries.append(summary)
            all_summaries.append(summary)
            print(
                f"[{scene}] {summary['method']} iter={summary['final_iteration']} "
                f"time={summary['train_wall_seconds']:.2f}s gaussians={summary['final_gaussians']}"
            )
        assert_same_split(scene_summaries)
        shared_split = {
            "scene": scene,
            "train_camera_names": scene_summaries[0]["train_camera_names"],
            "validation_camera_names": scene_summaries[0]["validation_camera_names"],
            "methods": [summary["method"] for summary in scene_summaries],
        }
        with (run_dir / scene / "shared_split.json").open("w", encoding="utf-8") as f:
            json.dump(shared_split, f, indent=2)

    with (run_dir / "unified_eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"Unified evaluation directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
