"""Multi-seed budget sweep for a budget-conditioned agent.

Runs every budget under several random seeds (default 5), reports per-budget
mean +/- std of the outcome metrics, and saves the FULL per-block action set for
every (seed, budget) episode.

Outputs under --out:
  runs.csv                  : one row per (seed, budget) with final metrics.
  summary.csv               : per budget, mean & std across seeds (psnr/ssim/N/iter/densify_on).
  blocks/seed{S}_B{B}.csv   : per-block action set (iter, time, gaussians, added, pruned,
                              + all decoded discrete/continuous actions) for that episode.

Run in the env matching the backend (fgs_cu128 / env_pytorch_3dgs).
"""
import argparse, csv, json, random, statistics, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action, observation_names_for
from agentic_gs_phase1.policies import ActorCritic

ACTION_FIELDS = ["block_steps", "densify_mode", "densification_interval", "prune_mode", "opacity_reset",
                 "densify_threshold_mult", "prune_opacity_threshold", "position_lr_mult", "feature_lr_mult",
                 "opacity_lr_mult", "scaling_lr_mult", "rotation_lr_mult"]

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--method", default="agentic", choices=["agentic", "baseline"])
ap.add_argument("--scene", required=True)
ap.add_argument("--budgets", type=float, nargs="+", default=[30, 60, 120, 300, 360])
ap.add_argument("--unlimited", action="store_true")
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
ap.add_argument("--views", type=int, default=12)
ap.add_argument("--out", required=True)
args = ap.parse_args()

dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
cfg["fastergs_acknowledge_grad_bug"] = True
cfg["budget_conditioned"] = True
cfg["save_episode_models"] = False
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
(OUT / "blocks").mkdir(exist_ok=True)
UNLIMITED = 10_000_000.0
budgets = list(args.budgets) + ([UNLIMITED] if args.unlimited else [])
CONT = DISCRETE_ACTIONS["stop"].index("continue")

policy = None
if args.method == "agentic":
    ckpt = torch.load(args.checkpoint, map_location=dev)
    pc = cfg.get("policy", {})
    obs_dim = int(ckpt.get("obs_dim", len(observation_names_for(cfg))))
    policy = ActorCritic(obs_dim, int(pc.get("hidden_width", 256)), int(pc.get("hidden_layers", 3)),
                         str(pc.get("activation", "gelu")), config=cfg).to(dev)
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


def blabel(B):
    return "unlimited" if B >= UNLIMITED else str(int(B))


env = AgenticGSEnv(cfg, run_dir=OUT / "_run", seed=args.seeds[0])
runs = []
for seed in args.seeds:
    env.rng = random.Random(seed); env.np_rng = np.random.default_rng(seed); env.seed = seed
    torch.manual_seed(seed)
    for B in budgets:
        env.budget_override = float(B)
        obs = env.reset(args.scene, episode_id=seed)
        done, n_dens_on, n_blocks = False, 0, 0
        blk_rows = []
        while not done:
            if args.method == "agentic":
                a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
                a["discrete"]["stop"] = CONT
            else:
                a = default_action()
            obs, _, done, info = env.step(a)
            n_blocks += 1
            act = info["action"]
            if act["densify_mode"] != "off":
                n_dens_on += 1
            bs = info["block_stats"]
            row = {"block": info["block_index"], "iter": info["iteration"],
                   "t": round(env.training_seconds, 3),
                   "gaussians": int(bs.get("gaussian_count", env.gaussians.get_xyz.shape[0])),
                   "added": int(bs.get("gaussians_added", 0)), "pruned": int(bs.get("gaussians_pruned", 0))}
            for f in ACTION_FIELDS:
                v = act.get(f)
                row[f] = round(v, 4) if isinstance(v, float) else v
            blk_rows.append(row)
        p, s = test_metrics(env, args.views)
        runs.append({"seed": seed, "budget_s": B, "budget": blabel(B), "final_iter": env.iteration,
                     "train_seconds": round(env.training_seconds, 1),
                     "gaussians": int(env.gaussians.get_xyz.shape[0]),
                     "densify_on_blocks": n_dens_on, "blocks": n_blocks,
                     "test_psnr": round(p, 3), "test_ssim": round(s, 4)})
        # save the per-block action set
        with open(OUT / "blocks" / f"seed{seed}_B{int(B)}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(blk_rows[0].keys())); w.writeheader()
            for r in blk_rows: w.writerow(r)
        print(f"[seed {seed} | B={blabel(B):>9}] iter={env.iteration:5d} t={env.training_seconds:6.1f}s "
              f"N={runs[-1]['gaussians']:7d} psnr={p:6.2f} ssim={s:.3f} densify_on={n_dens_on}/{n_blocks}", flush=True)
env.close()

with open(OUT / "runs.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(runs[0].keys())); w.writeheader()
    for r in runs: w.writerow(r)

# per-budget mean +/- std across seeds
def agg(vals):
    return (round(statistics.mean(vals), 4), round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0)

metrics = ["test_psnr", "test_ssim", "gaussians", "final_iter", "densify_on_blocks", "train_seconds"]
summary = []
for B in budgets:
    grp = [r for r in runs if r["budget_s"] == B]
    row = {"budget": blabel(B), "n_seeds": len(grp)}
    for m in metrics:
        mu, sd = agg([r[m] for r in grp])
        row[m + "_mean"] = mu; row[m + "_std"] = sd
    summary.append(row)
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys())); w.writeheader()
    for r in summary: w.writerow(r)

print("\n=== per-budget mean +/- std (across %d seeds) ===" % len(args.seeds))
for r in summary:
    print(f"  B={r['budget']:>9}: PSNR {r['test_psnr_mean']:.2f}+/-{r['test_psnr_std']:.2f} dB | "
          f"SSIM {r['test_ssim_mean']:.3f}+/-{r['test_ssim_std']:.3f} | "
          f"N {r['gaussians_mean']/1000:.0f}k+/-{r['gaussians_std']/1000:.0f}k | "
          f"iter {r['final_iter_mean']:.0f} | densify_on {r['densify_on_blocks_mean']:.0f}")
print("wrote", OUT / "runs.csv", "+ summary.csv + blocks/ (per-block action sets)")
print("DONE")
