"""Measure deployment + training efficiency metrics at the 30k reference, for the agent
(acceleration policy, stop disabled -> runs full horizon) or the fixed-schedule baseline.

Captures, for one (backend, method) run:
  training : peak VRAM (GB), wall-clock training seconds, policy inference overhead
             (total act() seconds, ms/block, % of training time)
  model    : Gaussian count, model size on disk (MB, from save_ply)
  quality  : test PSNR / SSIM / LPIPS(vgg) over the held-out views
  render   : FPS (synchronized), peak render VRAM (GB)

Writes metrics.json under --out. Run in the conda env matching the backend.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action, observation_names_for
from agentic_gs_phase1.policies import ActorCritic

CONT = DISCRETE_ACTIONS["stop"].index("continue")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["agentic", "baseline"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--render-views", type=int, default=40)
    ap.add_argument("--lpips-net", default="vgg")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    cfg = json.loads(Path(args.config).read_text())
    cfg["fastergs_acknowledge_grad_bug"] = True
    cfg["budget_conditioned"] = False
    cfg["save_episode_models"] = False
    cfg["max_episode_iterations"] = int(args.max_iter)
    cfg.setdefault("optimization", {})["position_lr_max_steps"] = int(args.max_iter)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    policy = None
    if args.method == "agentic":
        ckpt = torch.load(args.checkpoint, map_location=dev)
        pc = cfg.get("policy", {})
        obs_dim = int(ckpt.get("obs_dim", len(observation_names_for(cfg))))
        policy = ActorCritic(obs_dim, int(pc.get("hidden_width", 256)), int(pc.get("hidden_layers", 3)),
                             str(pc.get("activation", "gelu")), config=cfg).to(dev)
        policy.load_state_dict(ckpt["policy_state_dict"]); policy.eval()

    env = AgenticGSEnv(cfg, run_dir=out / "_run", seed=0)
    obs = env.reset(args.scene, episode_id=0)

    # ---------------- training loop (measure VRAM + inference overhead) ----------------
    torch.cuda.reset_peak_memory_stats()
    infer_s, n_blocks = 0.0, 0
    done = False
    while not done and env.iteration < args.max_iter:
        if args.method == "agentic":
            t0 = time.perf_counter()
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            a["discrete"]["stop"] = CONT
            torch.cuda.synchronize(); infer_s += time.perf_counter() - t0
        else:
            a = default_action()
        obs, _, done, info = env.step(a)
        n_blocks += 1
        if env.iteration >= args.max_iter:
            break
    train_peak_gb = torch.cuda.max_memory_reserved() / 1024**3
    train_seconds = float(env.training_seconds)
    n_gauss = int(env.gaussians.get_xyz.shape[0])

    # ---------------- model size on disk ----------------
    ply = out / "final_model.ply"
    env.gaussians.save_ply(str(ply))
    size_mb = ply.stat().st_size / 1024**2

    # ---------------- quality + render FPS + render VRAM ----------------
    import lpips
    lp = lpips.LPIPS(net=args.lpips_net).to(dev).eval()
    cams = list(env.scene.getTestCameras())
    cams = cams[::max(1, len(cams) // args.render_views)][:args.render_views]

    def render(c):
        return env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                        use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)

    with torch.no_grad():
        render(cams[0]); torch.cuda.synchronize()          # warm-up
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for c in cams:
            _ = render(c)
        torch.cuda.synchronize()
        render_s = time.perf_counter() - t0
    fps = len(cams) / render_s
    render_peak_gb = torch.cuda.max_memory_reserved() / 1024**3

    ps, ss, lps = [], [], []
    with torch.no_grad():
        for c in cams:
            im = render(c); gt = c.original_image.cuda().clamp(0, 1)
            ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
            ss.append(float(env.backend.ssim(im, gt)))
            lps.append(float(lp(im.unsqueeze(0) * 2 - 1, gt.unsqueeze(0) * 2 - 1).mean()))
    env.close()

    m = {
        "backend": cfg.get("trainer_backend"), "method": args.method, "scene": args.scene,
        "max_iter": args.max_iter, "final_iter": env.iteration if False else args.max_iter,
        "gaussians_k": round(n_gauss / 1e3, 1), "model_size_mb": round(size_mb, 1),
        "psnr": round(float(np.mean(ps)), 3), "ssim": round(float(np.mean(ss)), 4),
        "lpips": round(float(np.mean(lps)), 4),
        "train_peak_vram_gb": round(train_peak_gb, 2), "render_peak_vram_gb": round(render_peak_gb, 2),
        "render_fps": round(fps, 1), "train_seconds": round(train_seconds, 1),
        "n_blocks": n_blocks,
        "infer_total_s": round(infer_s, 3),
        "infer_ms_per_block": round(1000 * infer_s / max(1, n_blocks), 2),
        "infer_pct_of_train": round(100 * infer_s / max(1e-6, train_seconds), 3),
        "render_views": len(cams), "lpips_net": args.lpips_net,
    }
    (out / "metrics.json").write_text(json.dumps(m, indent=1))
    print(json.dumps(m, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
