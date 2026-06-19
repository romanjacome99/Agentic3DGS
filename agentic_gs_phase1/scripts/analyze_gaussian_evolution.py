"""Replay an agent episode (and a fixed-schedule baseline) and record the raw
per-Gaussian attribute distributions at each block, to visualize how the
agentic controller reshapes the Gaussian population over training iterations.

Records, per block: count, opacity quantiles + near-transparent fractions,
scale quantiles (relative to scene extent), and anisotropy quantiles.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action
from agentic_gs_phase1.policies import ActorCritic


@torch.no_grad()
def gaussian_stats(env) -> dict:
    g = env.gaussians
    op = g.get_opacity.detach().flatten().float()
    sc = g.get_scaling.detach().float()
    max_axis = sc.max(dim=1).values
    min_axis = torch.clamp(sc.min(dim=1).values, min=1e-9)
    aniso = max_axis / min_axis
    extent = max(1e-6, float(env.scene.cameras_extent))

    def q(t, v):
        return float(torch.quantile(t, v).item()) if t.numel() else 0.0

    return {
        "count": int(g.get_xyz.shape[0]),
        "op_q10": round(q(op, 0.10), 4), "op_q50": round(q(op, 0.50), 4), "op_q90": round(q(op, 0.90), 4),
        "op_mean": round(float(op.mean().item()) if op.numel() else 0.0, 4),
        "op_lt01": round(float((op < 0.01).float().mean().item()) if op.numel() else 0.0, 4),
        "op_lt10": round(float((op < 0.10).float().mean().item()) if op.numel() else 0.0, 4),
        "sc_q50": round(q(max_axis, 0.50) / extent, 5), "sc_q90": round(q(max_axis, 0.90) / extent, 5),
        "aniso_q50": round(q(aniso, 0.50), 3), "aniso_q90": round(q(aniso, 0.90), 3),
    }


def run_episode(env, scene, policy, device, episode_id, use_policy, force_full=False):
    obs = env.reset(scene, episode_id=episode_id)
    rec = [{**gaussian_stats(env), "iter": 0, "psnr": float(env.last_validation["psnr"]), "added": 0, "pruned": 0}]
    done, info = False, {}
    while not done:
        if use_policy:
            ot = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, _, _ = policy.act(ot, deterministic=True)
            if force_full:
                action["discrete"]["stop"] = DISCRETE_ACTIONS["stop"].index("continue")
        else:
            action = default_action()
        obs, _, done, info = env.step(action)
        s = gaussian_stats(env)
        s["iter"] = int(info["iteration"])
        s["psnr"] = round(float(info["validation"]["psnr"]), 3)
        s["added"] = int(info["block_stats"]["gaussians_added"])
        s["pruned"] = int(info["block_stats"]["gaussians_pruned"])
        rec.append(s)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "agentic_gs_phase1" / "configs" / "phase1_eval.json")
    ap.add_argument("--scenes", nargs="+", default=["hotdog", "drums", "materials"])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-iter", type=int, default=0, help="override max episode iterations (0 = config default)")
    ap.add_argument("--force-full", action="store_true", help="disable the policy stop action (run to max-iter)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.max_iter > 0:
        config["max_episode_iterations"] = int(args.max_iter)
        config.setdefault("optimization", {})["position_lr_max_steps"] = int(args.max_iter)
    ckpt = torch.load(args.checkpoint, map_location=device)
    obs_dim = int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names)))
    pcfg = config.get("policy", {})
    policy = ActorCritic(obs_dim, int(pcfg.get("hidden_width", 256)), int(pcfg.get("hidden_layers", 3)),
                         str(pcfg.get("activation", "gelu"))).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()

    out = {}
    env = AgenticGSEnv(config, run_dir=PROJECT_ROOT / "outputs" / "_gauss_evo_tmp", seed=int(config.get("seed", 0)))
    for scene in args.scenes:
        out[scene] = {}
        print(f"[{scene}] agent episode ...", flush=True)
        out[scene]["agentic"] = run_episode(env, scene, policy, device, 80000, use_policy=True, force_full=args.force_full)
        print(f"[{scene}] baseline episode ...", flush=True)
        out[scene]["baseline"] = run_episode(env, scene, policy, device, 80001, use_policy=False)
        a, b = out[scene]["agentic"], out[scene]["baseline"]
        print(f"[{scene}] agent {len(a)} blocks -> it{a[-1]['iter']} {a[-1]['count']//1000}kG op50={a[-1]['op_q50']:.2f} | "
              f"baseline {len(b)} blocks -> it{b[-1]['iter']} {b[-1]['count']//1000}kG op50={b[-1]['op_q50']:.2f}", flush=True)
    env.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
