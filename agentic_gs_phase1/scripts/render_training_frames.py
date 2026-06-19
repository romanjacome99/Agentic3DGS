"""Train a scene under a given method (agentic / baseline) and render a fixed
test view at every block, recording the reconstructed frame and its PSNR.

Produces a frames/ folder of PNGs plus a manifest.json, for building a
training-progress animation (baseline vs agentic).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
GS_DIR = PROJECT_ROOT / "gaussian-splatting"
if str(GS_DIR) not in sys.path:
    sys.path.insert(0, str(GS_DIR))

from gaussian_renderer import render  # noqa: E402
from utils.image_utils import psnr as psnr_fn  # noqa: E402

from agentic_gs_phase1.envs import AgenticGSEnv  # noqa: E402
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, decode_action, default_action  # noqa: E402
from agentic_gs_phase1.policies import ActorCritic  # noqa: E402

CONTINUE = DISCRETE_ACTIONS["stop"].index("continue")


def to_png(img_tensor, size):
    arr = (img_tensor.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if size and im.size[0] != size:
        im = im.resize((size, size), Image.LANCZOS)
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--method", required=True, choices=["agentic", "baseline"])
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=PROJECT_ROOT / "agentic_gs_phase1" / "configs" / "phase1_eval.json")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--frame-size", type=int, default=400)
    ap.add_argument("--max-iter", type=int, default=7000)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    device = torch.device("cuda")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    config["max_episode_iterations"] = int(args.max_iter)
    config.setdefault("optimization", {})["position_lr_max_steps"] = int(args.max_iter)

    policy = None
    if args.method == "agentic":
        ckpt = torch.load(args.checkpoint, map_location=device)
        obs_dim = int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names)))
        pcfg = config.get("policy", {})
        policy = ActorCritic(obs_dim, int(pcfg.get("hidden_width", 256)), int(pcfg.get("hidden_layers", 3)),
                             str(pcfg.get("activation", "gelu"))).to(device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        policy.eval()

    frames_dir = args.output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    env = AgenticGSEnv(config, run_dir=args.output_dir / "_run", seed=int(config.get("seed", 0)))
    obs = env.reset(args.scene, episode_id=0)

    test_cams = list(env.scene.getTestCameras())
    cam = test_cams[args.view_index % len(test_cams)]
    gt = cam.original_image.to(device).clamp(0, 1)
    to_png(gt, args.frame_size).save(args.output_dir / "gt.png")

    @torch.no_grad()
    def capture(iteration):
        img = torch.clamp(render(cam, env.gaussians, env.pipe, env.background)["render"], 0.0, 1.0)
        p = float(psnr_fn(img.unsqueeze(0), gt.unsqueeze(0)).mean().item())
        fn = f"f_{iteration:06d}.png"
        to_png(img, args.frame_size).save(frames_dir / fn)
        return {"iter": int(iteration), "psnr": round(p, 3), "count": int(env.gaussians.get_xyz.shape[0]), "file": fn}

    def record_action(action):
        """Decode the action the policy actually took into a compact json record."""
        d = decode_action(action).as_dict()
        return {
            "block_steps": int(d["block_steps"]),
            "densify_mode": d["densify_mode"],
            "densification_interval": int(d["densification_interval"]),
            "prune_mode": d["prune_mode"],
            "opacity_reset": d["opacity_reset"],
            "stop": d["stop"],
            "densify_threshold_mult": round(float(d["densify_threshold_mult"]), 3),
            "prune_opacity_threshold": round(float(d["prune_opacity_threshold"]), 4),
            "position_lr_mult": round(float(d["position_lr_mult"]), 3),
            "feature_lr_mult": round(float(d["feature_lr_mult"]), 3),
            "opacity_lr_mult": round(float(d["opacity_lr_mult"]), 3),
            "scaling_lr_mult": round(float(d["scaling_lr_mult"]), 3),
            "rotation_lr_mult": round(float(d["rotation_lr_mult"]), 3),
        }

    frames = [capture(0)]
    done = False
    while not done:
        if args.method == "agentic":
            action, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=device), deterministic=True)
            action["discrete"]["stop"] = CONTINUE  # disable stop head
        else:
            action = default_action()
        act_rec = record_action(action)
        obs, _, done, info = env.step(action)
        fr = capture(int(info["iteration"]))
        fr["act"] = act_rec
        frames.append(fr)
        if int(info["iteration"]) >= int(args.max_iter):
            break
        if len(frames) % 10 == 0:
            print(f"[{args.scene}/{args.method}] it={frames[-1]['iter']} psnr={frames[-1]['psnr']:.2f}", flush=True)

    env.close()
    manifest = {"scene": args.scene, "method": args.method, "view_index": args.view_index,
                "frame_size": args.frame_size, "gt": "gt.png", "frames": frames}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[{args.scene}/{args.method}] {len(frames)} frames -> it{frames[-1]['iter']} "
          f"psnr={frames[-1]['psnr']:.2f}  wrote {args.output_dir/'manifest.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
