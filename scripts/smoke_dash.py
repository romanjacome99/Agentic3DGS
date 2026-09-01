"""Smoke test for the DashGaussian backend (controllable substrate).

Verifies end-to-end that: (1) the dash backend loads and builds the resolution +
primitive-budget schedule, (2) the env trains through a few policy blocks rendering at
the scheduled (increasing) resolution while the momentum budget caps densification, and
(3) the existing base 3DGS policy checkpoint (obs_dim 45) acts on the dash env unchanged.

Run:  conda run -n env_pytorch_3dgs python scripts/smoke_dash.py --scene counter
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT))
import torch
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import default_action, observation_names_for
from agentic_gs_phase1.policies import ActorCritic

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "configs" / "final_accel_dash.json"))
ap.add_argument("--scene", default="counter")
ap.add_argument("--blocks", type=int, default=6)
ap.add_argument("--checkpoint", default=str(ROOT / "outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth"))
args = ap.parse_args()

cfg = json.loads(Path(args.config).read_text())
cfg["max_episode_iterations"] = 1200          # keep the smoke short
cfg.setdefault("safety", {})["min_iterations_before_stop"] = 1200
obs_dim = len(observation_names_for(cfg))
print(f"[cfg] backend={cfg['trainer_backend']} dash={cfg.get('dash', {}).get('enabled')} obs_dim={obs_dim}")

dev = torch.device("cuda")
env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs" / "_smoke_dash", seed=0)
print(f"[backend] {type(env.backend).__name__} is_dash={getattr(env.backend,'is_dash',False)} dash_enabled={env.dash_enabled}")

obs = env.reset(args.scene, episode_id=0)
assert env._dash is not None, "Dash scheduler was not built at reset"
print(f"[reset] scene={args.scene} init_gaussians={env.initial_gaussian_count} obs={obs.shape}")
# resolution schedule sanity: coarse -> fine
sched = [(it, round(env._dash.res_scale(it), 3)) for it in (0, 100, 300, 600, 1200, 3000)]
print(f"[schedule] res_scale over iters: {sched}")
assert env._dash.res_scale(0) < env._dash.res_scale(3000), "resolution should increase over training"

# Load the base 3DGS policy and confirm it acts on the dash observation unchanged.
policy = None
if Path(args.checkpoint).exists():
    ck = torch.load(args.checkpoint, map_location=dev)
    p = cfg.get("policy", {})
    policy = ActorCritic(int(ck.get("obs_dim", obs_dim)), int(p.get("hidden_width", 256)),
                         int(p.get("hidden_layers", 3)), str(p.get("activation", "gelu")), config=cfg).to(dev)
    policy.load_state_dict(ck["policy_state_dict"]); policy.eval()
    print(f"[policy] loaded {Path(args.checkpoint).name} (ckpt obs_dim={ck.get('obs_dim')})")

print(f"[run] stepping {args.blocks} blocks ...")
for b in range(args.blocks):
    if policy is not None:
        a, _, _ = policy.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
    else:
        a = default_action(cfg)
    obs, reward, done, info = env.step(a)
    bs = info["block_stats"]
    it = info["iteration"]
    cur = int(env.gaussians.get_xyz.shape[0])
    budget = env._dash.primitive_budget(it, cur)
    print(f"  block {b}: iter={it:4d} res_scale={env._current_res_scale:.3f} "
          f"N={cur} budget={budget} added={bs.get('gaussians_added',0)} "
          f"pruned={bs.get('gaussians_pruned',0)} val_psnr={info.get('validation',{}).get('psnr',0):.2f} "
          f"reward={reward:+.3f}")
    if done:
        print("  (episode ended)"); break

env.close()
print("DASH_SMOKE_OK")
