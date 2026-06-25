"""Regression smoke for the trainer-backend refactor: run a few env steps on a
synthetic scene with the selected backend (default 3dgs) using default actions."""
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import default_action

ap = argparse.ArgumentParser()
ap.add_argument("--backend", default="3dgs")
ap.add_argument("--scene", default="hotdog")
ap.add_argument("--config", default=str(ROOT / "agentic_gs_phase1" / "configs" / "phase1_eval.json"))
ap.add_argument("--steps", type=int, default=3)
args = ap.parse_args()

cfg = json.loads(Path(args.config).read_text())
cfg["trainer_backend"] = args.backend
cfg["max_episode_iterations"] = 600

env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs" / "_smoke_backend", seed=0)
print(f"[backend] {env.backend.name} ({env.backend.display}) repo={env.backend.repo_dir.name}", flush=True)
obs = env.reset(args.scene, episode_id=0)
print(f"[reset] init_gaussians={env.gaussians.get_xyz.shape[0]} train={len(list(env.scene.getTrainCameras()))} test={len(list(env.scene.getTestCameras()))}", flush=True)
done = False
i = 0
while not done and i < args.steps:
    obs, r, done, info = env.step(default_action())
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[step {i}] iter={info['iteration']} gaussians={env.gaussians.get_xyz.shape[0]} reward={r:.3f} val_psnr={info.get('validation_psnr', float('nan')):.2f} vram={vram:.1f}GB", flush=True)
    i += 1
# Direct test-view PSNR (backend-agnostic), independent of periodic validation.
cam = list(env.scene.getTestCameras())[0]
img = env.backend.render_image(cam, env.gaussians, env.pipe, env.background, use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
gt = cam.original_image.to("cuda").clamp(0, 1)
p = float(env.backend.psnr(img.unsqueeze(0), gt.unsqueeze(0)).mean().item())
print(f"[final] iter={env.iteration} gaussians={env.gaussians.get_xyz.shape[0]} test_view0_psnr={p:.2f}", flush=True)
env.close()
print("SMOKE OK", flush=True)
