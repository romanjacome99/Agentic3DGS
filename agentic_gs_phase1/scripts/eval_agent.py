from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.policies import ActorCritic


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fixed Phase 1 3DGS control policy.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-scenes", nargs="+", default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-episode-iterations", type=int, default=None)
    parser.add_argument("--save-time-budgets", nargs="*", type=float, default=[])
    parser.add_argument("--stochastic", action="store_true")
    return parser.parse_args()


def make_policy(config: dict[str, Any], obs_dim: int, device: torch.device) -> ActorCritic:
    policy_cfg = config.get("policy", {})
    return ActorCritic(
        obs_dim=obs_dim,
        hidden_width=int(policy_cfg.get("hidden_width", 256)),
        hidden_layers=int(policy_cfg.get("hidden_layers", 3)),
        activation=str(policy_cfg.get("activation", "gelu")),
    ).to(device)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS RL environment requires CUDA.")
    device = torch.device("cuda")
    ckpt = torch.load(args.checkpoint, map_location=device)
    config = dict(ckpt.get("config", {}))
    if args.config is not None:
        config.update(load_json(args.config))
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.max_episode_iterations is not None:
        config["max_episode_iterations"] = int(args.max_episode_iterations)

    eval_scenes = args.eval_scenes or list(config.get("eval_scenes", ["hotdog", "drums"]))
    obs_dim = int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names)))
    policy = make_policy(config, obs_dim, device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    run_name = args.run_name or time.strftime("eval_%Y%m%d_%H%M%S")
    output_root = Path(config.get("output_root", "outputs/agentic_gs_phase1"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    env = AgenticGSEnv(config, run_dir=run_dir, seed=int(config.get("seed", 0)) + 1000)
    summaries = []
    for episode_id, scene in enumerate(eval_scenes):
        obs = env.reset(scene, episode_id=episode_id)
        done = False
        total_reward = 0.0
        last_info = None
        pending_budgets = sorted(float(value) for value in args.save_time_budgets if float(value) > 0.0)
        while not done:
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _, _ = policy.act(obs_tensor, deterministic=not args.stochastic)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            last_info = info
            while pending_budgets and env.elapsed_seconds >= pending_budgets[0]:
                env.save_model()
                pending_budgets.pop(0)
        env.save_model()
        summaries.append(
            {
                "scene": scene,
                "total_reward": total_reward,
                "final_iteration": env.iteration,
                "final_validation": last_info["validation"] if last_info else {},
                "model_path": str(env.model_path),
                "test_camera_accessed_for_reward_or_state": False,
            }
        )
        print(
            f"[{scene}] iter={env.iteration} reward={total_reward:.4f} "
            f"val_psnr={summaries[-1]['final_validation'].get('psnr', 0.0):.3f}"
        )

    with (run_dir / "eval_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)
    env.close()
    print(f"Eval directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
