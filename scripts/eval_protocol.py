"""Time-to-target PSNR evaluation protocol (backend-aware, FasterGS-clean).

For a scene, runs baseline (fixed 3DGS schedule) and/or agentic (policy), force-full,
and saves rich artifacts:
  - <method>_blocks.csv : per-block trajectory — iter, training wall-seconds,
        Gaussian count / added / pruned, block_seconds, per-block validation
        psnr/ssim, reward, and the full decoded ACTION (discrete + continuous),
        plus the policy's stop-intent (what it wanted before force-full).
  - curve.csv           : sampled multi-view TEST psnr/ssim + Gaussian count.
  - time_to_target.csv  : time each method first reaches each PSNR target + speedups.
  - summary.json        : final per-method stats + run metadata.

Run in the env matching the backend (fgs_cu128 for fastergs, env_pytorch_3dgs for 3dgs).
"""
import argparse, csv, json, sys, time
from pathlib import Path
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action
from agentic_gs_phase1.policies import ActorCritic

ACTION_FIELDS = ["block_steps", "densify_mode", "densification_interval", "prune_mode",
                 "opacity_reset", "stop", "densify_threshold_mult", "prune_opacity_threshold",
                 "position_lr_mult", "feature_lr_mult", "opacity_lr_mult", "scaling_lr_mult",
                 "rotation_lr_mult"]
CONT = DISCRETE_ACTIONS["stop"].index("continue")

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--checkpoint", default=None)
ap.add_argument("--scene", required=True)
ap.add_argument("--max-iter", type=int, default=7000)
ap.add_argument("--views", type=int, default=12)
ap.add_argument("--targets", type=float, nargs="+", default=[16, 17, 18, 19, 20])
ap.add_argument("--methods", nargs="+", default=["baseline", "agentic"])
ap.add_argument("--out", required=True)
args = ap.parse_args()

SAMPLES = [s for s in [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000, 10000, 15000, 30000] if s <= args.max_iter]
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
cfg["max_episode_iterations"] = args.max_iter
cfg["fastergs_acknowledge_grad_bug"] = True
cfg.setdefault("safety", {})["min_iterations_before_stop"] = args.max_iter
# Force-full: stretch the position-LR schedule to the full budget so 30k runs
# decay properly (otherwise LR hits its floor at the original ~7k max-steps).
cfg.setdefault("optimization", {})["position_lr_max_steps"] = args.max_iter
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
ckpt = torch.load(args.checkpoint, map_location=dev) if args.checkpoint else None


def make_policy():
    p = cfg.get("policy", {})
    m = ActorCritic(int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names))),
                    int(p.get("hidden_width", 256)), int(p.get("hidden_layers", 3)),
                    str(p.get("activation", "gelu"))).to(dev)
    m.load_state_dict(ckpt["policy_state_dict"]); m.eval(); return m


def test_metrics(env, k):
    cams = list(env.scene.getTestCameras())
    cams = cams[::max(1, len(cams) // k)][:k]
    ps, ss = [], []
    with torch.no_grad():
        for c in cams:
            im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                          use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
            gt = c.original_image.cuda().clamp(0, 1)
            ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
            ss.append(float(env.backend.ssim(im, gt)))
    return sum(ps) / len(ps), sum(ss) / len(ss)


def run(method):
    env = AgenticGSEnv(cfg, run_dir=OUT / ("_run_" + method), seed=int(cfg.get("seed", 0)))
    obs = env.reset(args.scene, episode_id=0)
    policy = make_policy() if method == "agentic" else None
    blocks, curve, si, done = [], [], 0, False
    while not done:
        if method == "agentic":
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            stop_intent = DISCRETE_ACTIONS["stop"][int(a["discrete"]["stop"])]
            a["discrete"]["stop"] = CONT  # force-full to trace the whole curve
        else:
            a = default_action(); stop_intent = "continue"
        obs, reward, done, info = env.step(a)
        bs, act, val = info["block_stats"], info["action"], info["validation"]
        row = {"method": method, "block": info["block_index"], "iter": info["iteration"],
               "t": round(env.training_seconds, 3),
               "gaussians": int(bs.get("gaussian_count", env.gaussians.get_xyz.shape[0])),
               "added": int(bs.get("gaussians_added", 0)), "pruned": int(bs.get("gaussians_pruned", 0)),
               "block_seconds": round(float(bs.get("block_seconds", 0.0)), 4),
               "val_psnr": round(float(val.get("psnr", 0.0)), 3), "val_ssim": round(float(val.get("ssim", 0.0)), 4),
               "reward": round(float(reward), 4), "stop_intent": stop_intent}
        for f in ACTION_FIELDS:
            v = act.get(f)
            row[f] = round(v, 4) if isinstance(v, float) else v
        blocks.append(row)
        it = info["iteration"]
        while si < len(SAMPLES) and it >= SAMPLES[si]:
            p, s = test_metrics(env, args.views)
            curve.append({"method": method, "iter": it, "t": round(env.training_seconds, 2),
                          "test_psnr": round(p, 3), "test_ssim": round(s, 4),
                          "N": int(env.gaussians.get_xyz.shape[0])})
            print(f"[{method}] it={it:5d} t={env.training_seconds:7.1f}s psnr={p:6.2f} ssim={s:.3f} N={env.gaussians.get_xyz.shape[0]}", flush=True)
            si += 1
        if it >= args.max_iter:
            break
    env.close()
    # per-method block trajectory
    with open(OUT / f"{method}_blocks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(blocks[0].keys())); w.writeheader()
        for r in blocks: w.writerow(r)
    return blocks, curve


all_curve, summary = [], {"scene": args.scene, "config": args.config, "checkpoint": args.checkpoint,
                          "max_iter": args.max_iter, "backend": cfg.get("trainer_backend", "3dgs"), "methods": {}}
for m in args.methods:
    blk, cur = run(m)
    all_curve += cur
    last = cur[-1] if cur else {}
    summary["methods"][m] = {"final": last, "blocks": len(blk),
                             "natural_stop_iter": next((b["iter"] for b in blk if b.get("stop_intent") == "stop"), None)}

with open(OUT / "curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "iter", "t", "test_psnr", "test_ssim", "N"]); w.writeheader()
    for r in all_curve: w.writerow(r)

def time_to(m, tgt):
    c = sorted((r["t"], r["test_psnr"]) for r in all_curve if r["method"] == m)
    for (t0, p0), (t1, p1) in zip(c, c[1:]):
        if p0 <= tgt <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return None

ttt_rows = []
print(f"\n=== time-to-target ({args.scene}, backend={summary['backend']}) ===")
for tg in args.targets:
    tb, ta = time_to("baseline", tg), time_to("agentic", tg)
    su = round(tb / ta, 3) if (tb and ta) else None
    ttt_rows.append({"target_psnr": tg, "baseline_s": tb and round(tb, 2), "agentic_s": ta and round(ta, 2), "speedup": su})
    print(f"  {tg} dB:  baseline={tb and round(tb,1)}s  agent={ta and round(ta,1)}s  speedup={su}x")
with open(OUT / "time_to_target.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["target_psnr", "baseline_s", "agentic_s", "speedup"]); w.writeheader()
    for r in ttt_rows: w.writerow(r)
summary["time_to_target"] = ttt_rows
(OUT / "summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nwrote {OUT}/  (blocks.csv per method, curve.csv, time_to_target.csv, summary.json)")
print("DONE")
