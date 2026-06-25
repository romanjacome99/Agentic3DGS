"""Finite-difference gradient check on the REAL scene through the env backend.
Compares analytic autograd grad to central finite differences for the color (DC)
and opacity params of the most-active Gaussians on a real training view."""
import argparse, json, sys
from pathlib import Path
import torch
ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv

ap = argparse.ArgumentParser(); ap.add_argument("--backend", default="fastergs")
ap.add_argument("--scene", default="hotdog")
ap.add_argument("--config", default=str(ROOT / "agentic_gs_phase1" / "configs" / "phase1_eval.json"))
a = ap.parse_args()
cfg = json.loads(Path(a.config).read_text())
cfg["trainer_backend"] = a.backend; cfg["max_episode_iterations"] = 600
cfg["fastergs_acknowledge_grad_bug"] = True
env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs" / "_gce", seed=0)
env.reset(a.scene, episode_id=0)
g = env.gaussians
cam = list(env.scene.getTrainCameras())[0]
gt = cam.original_image.cuda().clamp(0, 1)

def loss_now():
    out = env.backend.render_training(cam, g, env.pipe, env.background, use_trained_exp=False)
    return env.backend.l1_loss(out["image"], gt)

L = loss_now(); L.backward()
print(f"[{a.backend}] loss={float(L):.5f}", flush=True)
eps = 2e-3
for name in ["_features_dc", "_opacity", "_xyz", "_scaling", "_rotation"]:
    p = getattr(g, name)
    gflat = p.grad.reshape(-1)
    idx = torch.topk(gflat.abs(), k=8).indices
    ana, num = [], []
    base = p.detach().clone()
    for i in idx.tolist():
        flat = base.reshape(-1).clone()
        o = flat[i].item()
        flat[i] = o + eps; p.data = flat.reshape(p.shape)
        with torch.no_grad(): lp = float(loss_now())
        flat[i] = o - eps; p.data = flat.reshape(p.shape)
        with torch.no_grad(): lm = float(loss_now())
        p.data = base.clone()
        num.append((lp - lm) / (2 * eps)); ana.append(gflat[i].item())
    na, nu = torch.tensor(ana), torch.tensor(num)
    cos = float((na @ nu) / (na.norm() * nu.norm() + 1e-12))
    print(f"  {name:13s} cos(ana,num)={cos:+.4f}  ana={[round(x,4) for x in ana[:4]]}  num={[round(x,4) for x in num[:4]]}", flush=True)
env.close(); print("DONE", flush=True)
