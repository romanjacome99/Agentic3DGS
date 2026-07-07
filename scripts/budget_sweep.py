"""Budget-sweep evaluation: run a budget-conditioned policy at several fixed
wall-clock budgets on a held-out scene and record how its behavior changes
(final PSNR/SSIM, Gaussian count, iterations reached, whether it self-stopped,
and the per-block action means). Answers: does the agent adapt to its budget?

Run in the env matching the backend (fgs_cu128 / env_pytorch_3dgs).
"""
import argparse, csv, json, sys
from pathlib import Path
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action
from agentic_gs_phase1.policies import ActorCritic

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--method", default="agentic", choices=["agentic", "baseline"])
ap.add_argument("--scene", required=True)
ap.add_argument("--budgets", type=float, nargs="+", default=[30, 60, 120, 300])
ap.add_argument("--unlimited", action="store_true", help="also run an 'unlimited' budget (ends at the 30k iteration cap)")
ap.add_argument("--views", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
cfg["fastergs_acknowledge_grad_bug"] = True
cfg["budget_conditioned"] = True
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
from agentic_gs_phase1.envs.spaces import observation_names_for
UNLIMITED = 10_000_000.0  # budget so large the episode ends at the 30k iteration cap
budgets = list(args.budgets) + ([UNLIMITED] if args.unlimited else [])

policy = None
if args.method == "agentic":
    ckpt = torch.load(args.checkpoint, map_location=dev)
    pcfg = cfg.get("policy", {})
    obs_dim = int(ckpt.get("obs_dim", len(observation_names_for(cfg))))
    policy = ActorCritic(obs_dim, int(pcfg.get("hidden_width", 256)), int(pcfg.get("hidden_layers", 3)),
                         str(pcfg.get("activation", "gelu"))).to(dev)
    policy.load_state_dict(ckpt["policy_state_dict"]); policy.eval()


def test_metrics(env, k):
    cams = list(env.scene.getTestCameras()); cams = cams[::max(1, len(cams)//k)][:k]
    ps, ss = [], []
    with torch.no_grad():
        for c in cams:
            im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                          use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
            gt = c.original_image.cuda().clamp(0, 1)
            ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
            ss.append(float(env.backend.ssim(im, gt)))
    return sum(ps)/len(ps), sum(ss)/len(ss)


DISC = ["densify_mode", "prune_mode", "opacity_reset", "block_steps", "densification_interval"]
CONT = ["densify_threshold_mult", "prune_opacity_threshold", "position_lr_mult", "feature_lr_mult",
        "opacity_lr_mult", "scaling_lr_mult", "rotation_lr_mult"]


def action_profile(acts):
    """Summarise the action set the agent used across an episode's blocks."""
    n = max(1, len(acts))
    prof = {"blocks": len(acts)}
    for f in DISC:
        counts = {}
        for a in acts:
            counts[str(a[f])] = counts.get(str(a[f]), 0) + 1
        prof[f] = {k: round(v / n, 3) for k, v in counts.items()}  # fraction of blocks per value
        prof[f + "_mode"] = max(counts, key=counts.get) if counts else None  # dominant value
    for f in CONT:
        vals = [float(a[f]) for a in acts]
        prof[f] = round(sum(vals) / n, 4) if vals else 0.0
    return prof


env = AgenticGSEnv(cfg, run_dir=OUT / "_run", seed=int(cfg.get("seed", 0)))
rows, profiles = [], {}
for B in budgets:
    env.budget_override = float(B)
    obs = env.reset(args.scene, episode_id=0)
    done, stopped, n_dens_on = False, False, 0
    n_blocks = 0
    acts = []
    while not done:
        if args.method == "agentic":
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            if DISCRETE_ACTIONS["stop"][int(a["discrete"]["stop"])] == "stop":
                stopped = True
        else:
            a = default_action()  # fixed 3DGS schedule (budget-terminated trainer baseline)
        obs, _, done, info = env.step(a)
        n_blocks += 1
        acts.append(info["action"])
        if info["action"]["densify_mode"] != "off":
            n_dens_on += 1
    p, s = test_metrics(env, args.views)
    row = {"budget_s": B, "final_iter": env.iteration, "train_seconds": round(env.training_seconds, 1),
           "self_stopped": stopped, "blocks": n_blocks, "densify_on_blocks": n_dens_on,
           "gaussians": int(env.gaussians.get_xyz.shape[0]), "test_psnr": round(p, 3), "test_ssim": round(s, 4)}
    rows.append(row)
    profiles[str(int(B))] = action_profile(acts)
    print(f"[B={B:5.0f}s] iter={row['final_iter']:5d} t={row['train_seconds']:6.1f}s stopped={stopped} "
          f"N={row['gaussians']:7d} psnr={p:6.2f} ssim={s:.3f} (blocks={n_blocks}, densify_on={n_dens_on})", flush=True)
env.close()
with open(OUT / "budget_sweep.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader()
    for r in rows: w.writerow(r)
(OUT / "action_profiles.json").write_text(json.dumps(profiles, indent=1))
print("wrote", OUT / "budget_sweep.csv", "+ action_profiles.json")
print("DONE")
