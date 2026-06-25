"""Self-contained time-to-target eval through the env + FasterGS backend.

Baseline (fixed 3DGS schedule) vs agentic (policy checkpoint), force-full, on a
held-out scene. Records (iteration, training wall-seconds, multi-view test PSNR,
Gaussian count) and computes time-to-target speedups. FasterGS-clean (no
hardcoded official rasterizer). Run in the fgs_cu128 env.
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
ap.add_argument("--config", default=str(ROOT / "configs" / "real_fastergs_train.json"))
ap.add_argument("--checkpoint", default=str(ROOT / "outputs/agentic_rl_real/ppo_real_fastergs_v1/checkpoints/best.pth"))
ap.add_argument("--scene", default="train")
ap.add_argument("--max-iter", type=int, default=7000)
ap.add_argument("--views", type=int, default=12)
ap.add_argument("--out", default=str(ROOT / "outputs/agentic_rl_real/ppo_real_fastergs_v1/ttp_eval"))
args = ap.parse_args()

SAMPLES = [s for s in [200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 7000] if s <= args.max_iter]
dev = torch.device("cuda")
cfg = json.loads(Path(args.config).read_text())
cfg["max_episode_iterations"] = args.max_iter
cfg["fastergs_acknowledge_grad_bug"] = True
cfg.setdefault("safety", {})["min_iterations_before_stop"] = args.max_iter
CONT = DISCRETE_ACTIONS["stop"].index("continue")

ckpt = torch.load(args.checkpoint, map_location=dev)

def make_policy():
    p = cfg.get("policy", {})
    m = ActorCritic(int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names))),
                    int(p.get("hidden_width", 256)), int(p.get("hidden_layers", 3)),
                    str(p.get("activation", "gelu"))).to(dev)
    m.load_state_dict(ckpt["policy_state_dict"]); m.eval(); return m

def test_psnr(env, k):
    cams = list(env.scene.getTestCameras())
    step = max(1, len(cams) // k)
    cams = cams[::step][:k]
    ps = []
    with torch.no_grad():
        for c in cams:
            im = env.backend.render_image(c, env.gaussians, env.pipe, env.background,
                                          use_trained_exp=env.dataset.train_test_exp).clamp(0, 1)
            gt = c.original_image.cuda().clamp(0, 1)
            ps.append(float(env.backend.psnr(im.unsqueeze(0), gt.unsqueeze(0)).mean()))
    return sum(ps) / len(ps)

def run(method):
    env = AgenticGSEnv(cfg, run_dir=Path(args.out) / ("_run_" + method), seed=0)
    obs = env.reset(args.scene, episode_id=0)
    policy = make_policy() if method == "agentic" else None
    rows, si, done = [], 0, False
    while not done:
        if method == "agentic":
            a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
            a["discrete"]["stop"] = CONT
        else:
            a = default_action()
        obs, _, done, info = env.step(a)
        it = int(info["iteration"])
        while si < len(SAMPLES) and it >= SAMPLES[si]:
            p = test_psnr(env, args.views)
            n = int(env.gaussians.get_xyz.shape[0])
            rows.append({"method": method, "iter": it, "t": round(env.training_seconds, 2), "psnr": round(p, 3), "N": n})
            print(f"[{method}] it={it:5d} t={env.training_seconds:6.1f}s psnr={p:6.2f} N={n}", flush=True)
            si += 1
        if it >= args.max_iter:
            break
    env.close()
    return rows

allrows = run("baseline") + run("agentic")
outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
with open(outdir / "curve.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["method", "iter", "t", "psnr", "N"]); w.writeheader()
    for r in allrows: w.writerow(r)

def curve(m): return sorted((r["t"], r["psnr"]) for r in allrows if r["method"] == m)
def time_to(m, tgt):
    c = curve(m)
    for (t0, p0), (t1, p1) in zip(c, c[1:]):
        if p0 <= tgt <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return None

print(f"\n=== time-to-target on held-out '{args.scene}' (FasterGS) ===")
for tg in [16, 17, 18, 19, 20]:
    tb, ta = time_to("baseline", tg), time_to("agentic", tg)
    su = (tb / ta) if (tb and ta) else None
    print(f"  {tg} dB:  baseline={tb and round(tb,1)}s  agent={ta and round(ta,1)}s  speedup={su and round(su,2)}x")
ab = max((r for r in allrows if r["method"] == "baseline"), key=lambda r: r["iter"])
aa = max((r for r in allrows if r["method"] == "agentic"), key=lambda r: r["iter"])
print(f"  @{args.max_iter}it: baseline {ab['psnr']}dB/{ab['N']//1000}k/{ab['t']}s  vs  agent {aa['psnr']}dB/{aa['N']//1000}k/{aa['t']}s")
print(f"wrote {outdir/'curve.csv'}")
print("DONE")
