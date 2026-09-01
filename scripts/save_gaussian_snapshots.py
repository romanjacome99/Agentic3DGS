"""Save Gaussian point-cloud snapshots (.ply) along a training episode, so we can
analyze HOW the population evolves under the agent vs. the fixed-schedule baseline,
in both the acceleration and budget-conditioned regimes.

For a chosen (mode, method, backend) it drives one episode block-by-block and, the
first time the wall-clock training time crosses each requested threshold, dumps the
full Gaussian state (xyz, opacity, scale, rotation, SH) via gaussians.save_ply. It
also snapshots the initial point cloud (before any optimization) and the final state.

Outputs under --out:
  ply/snap_init.ply, ply/snap_t{T}s.ply, ply/snap_final.ply   full Gaussian states
  snapshots.csv   one row per snapshot: tag, trigger_s, iter, time_s, gaussians, psnr, ssim, ply
  blocks.csv      per-block action set + stats (iter, time, N, added, pruned, densify_mode, mults, reward)
  meta.json       mode/method/backend/scene/budget/checkpoint

Modes:
  --mode acceleration : config as-is (budget-agnostic policy, obs dim 45); the agent may
                        use its stop head. Episode runs until --horizon seconds (or the
                        agent stops, or --max-iter).
  --mode budget       : config budget_conditioned; sets the observed budget = --budget and
                        the horizon = --budget; stop head disabled (budget is the terminator).

Run in the conda env matching the backend (fgs_cu128 / env_pytorch_3dgs).
"""
from __future__ import annotations
import argparse, csv, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS, default_action, observation_names_for
from agentic_gs_phase1.policies import ActorCritic

ACTION_FIELDS = ["block_steps", "densify_mode", "densification_interval", "prune_mode", "opacity_reset",
                 "densify_threshold_mult", "prune_opacity_threshold", "position_lr_mult", "feature_lr_mult",
                 "opacity_lr_mult", "scaling_lr_mult", "rotation_lr_mult"]
CONTINUE = DISCRETE_ACTIONS["stop"].index("continue")
OFF = DISCRETE_ACTIONS["densify_mode"].index("off")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=["agentic", "baseline"])
    ap.add_argument("--mode", required=True, choices=["acceleration", "budget"])
    ap.add_argument("--checkpoint", default=None, help="policy checkpoint (required for --method agentic)")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--budget", type=float, default=120.0, help="budget mode: observed budget & horizon (s)")
    ap.add_argument("--horizon", type=float, default=None, help="acceleration mode: max wall-clock (s); default=max snap time")
    ap.add_argument("--snap-times", type=float, nargs="+", default=[5, 15, 30, 60, 120],
                    help="wall-clock thresholds (s) at which to dump a .ply")
    ap.add_argument("--views", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-iter", type=int, default=30000)
    ap.add_argument("--baseline-densify-until", type=int, default=-1,
                    help="baseline: freeze densification at/after this iter (-1 = never freeze; matches paper baseline)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required.")
    dev = torch.device("cuda")
    cfg = json.loads(Path(args.config).read_text())
    cfg["fastergs_acknowledge_grad_bug"] = True
    cfg["save_episode_models"] = False
    cfg["budget_conditioned"] = (args.mode == "budget")

    snap_times = sorted(set(float(t) for t in args.snap_times))
    horizon = args.budget if args.mode == "budget" else (args.horizon if args.horizon else max(snap_times))
    snap_times = [t for t in snap_times if t <= horizon + 1e-6]

    out = Path(args.out); (out / "ply").mkdir(parents=True, exist_ok=True)

    policy = None
    if args.method == "agentic":
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for --method agentic")
        ckpt = torch.load(args.checkpoint, map_location=dev)
        pc = cfg.get("policy", {})
        obs_dim = int(ckpt.get("obs_dim", len(observation_names_for(cfg))))
        policy = ActorCritic(obs_dim, int(pc.get("hidden_width", 256)), int(pc.get("hidden_layers", 3)),
                             str(pc.get("activation", "gelu"))).to(dev)
        policy.load_state_dict(ckpt["policy_state_dict"]); policy.eval()
        print(f"[policy] obs_dim={obs_dim} from {args.checkpoint}")

    env = AgenticGSEnv(cfg, run_dir=out / "_run", seed=args.seed)
    if args.mode == "budget":
        env.budget_override = float(args.budget)
    obs = env.reset(args.scene, episode_id=args.seed)

    def test_metrics():
        cams = list(env.scene.getTestCameras())
        cams = cams[::max(1, len(cams) // args.views)][:args.views]
        ps, ss = [], []
        with torch.no_grad():
            for c in cams:
                im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                              use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
                gt = c.original_image.cuda().clamp(0, 1)
                ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
                ss.append(float(env.backend.ssim(im, gt)))
        return sum(ps) / len(ps), sum(ss) / len(ss)

    snaps, blocks, saved = [], [], set()

    def save_snapshot(tag, trigger_s):
        ply = out / "ply" / f"snap_{tag}.ply"
        env.gaussians.save_ply(str(ply))
        p, s = test_metrics()
        rec = {"tag": tag, "trigger_s": trigger_s, "iter": env.iteration,
               "time_s": round(env.training_seconds, 2),
               "gaussians": int(env.gaussians.get_xyz.shape[0]),
               "psnr": round(p, 3), "ssim": round(s, 4), "ply": ply.name}
        snaps.append(rec)
        print(f"[snap {tag:>8}] it={rec['iter']:5d} t={rec['time_s']:6.1f}s N={rec['gaussians']:7d} "
              f"psnr={p:5.2f} ssim={s:.3f}", flush=True)

    # initial point cloud (before any optimization)
    save_snapshot("init", 0.0)

    done = False
    while not done and env.training_seconds < horizon and env.iteration < args.max_iter:
        if args.method == "agentic":
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            # Force-full to the horizon in BOTH modes so the agent is traced at every
            # snapshot checkpoint (incl. those past its natural early-stop), matching
            # eval_protocol.py's force-full curve and the baseline's coverage.
            a["discrete"]["stop"] = CONTINUE
        else:
            a = default_action()
            if args.baseline_densify_until >= 0 and env.iteration >= args.baseline_densify_until:
                a["discrete"]["densify_mode"] = OFF
        obs, reward, done, info = env.step(a)
        act, bs = info["action"], info["block_stats"]
        brow = {"block": info["block_index"], "iter": info["iteration"], "t": round(env.training_seconds, 3),
                "gaussians": int(bs.get("gaussian_count", env.gaussians.get_xyz.shape[0])),
                "added": int(bs.get("gaussians_added", 0)), "pruned": int(bs.get("gaussians_pruned", 0)),
                "reward": round(float(reward), 4)}
        for f in ACTION_FIELDS:
            v = act.get(f)
            brow[f] = round(v, 4) if isinstance(v, float) else v
        blocks.append(brow)
        # save any thresholds we just crossed
        for t in snap_times:
            if t not in saved and env.training_seconds >= t:
                saved.add(t)
                save_snapshot(f"t{int(t)}s", t)

    # final state
    save_snapshot("final", round(env.training_seconds, 2))
    env.close()

    with open(out / "snapshots.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(snaps[0].keys())); w.writeheader()
        for r in snaps: w.writerow(r)
    with open(out / "blocks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(blocks[0].keys())); w.writeheader()
        for r in blocks: w.writerow(r)
    (out / "meta.json").write_text(json.dumps({
        "mode": args.mode, "method": args.method, "scene": args.scene, "backend": cfg.get("trainer_backend"),
        "budget": args.budget if args.mode == "budget" else None, "horizon_s": horizon,
        "checkpoint": args.checkpoint, "seed": args.seed, "snap_times": snap_times,
        "baseline_densify_until": args.baseline_densify_until}, indent=1))
    print(f"[done] {len(snaps)} snapshots, {len(blocks)} blocks -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
