"""Quick smoke test: load the Mip360 bicycle scene under the eval config and run
a few agentic blocks to confirm COLMAP loading, images_4, policy, and VRAM."""
import json, sys
from pathlib import Path
import torch

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "gaussian-splatting"))
from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.policies import ActorCritic
from agentic_gs_phase1.envs.spaces import DISCRETE_ACTIONS

cfg = json.loads((ROOT / "configs" / "real_bicycle_eval.json").read_text())
cfg["max_episode_iterations"] = 600          # tiny smoke run
dev = torch.device("cuda")
ckpt = torch.load(ROOT / "outputs/agentic_gs_phase1/ppo_accel_500u_v8_resume144/checkpoints/best.pth", map_location=dev)
pol = ActorCritic(int(ckpt.get("obs_dim", len(AgenticGSEnv.observation_names))), 256, 3, "gelu").to(dev)
pol.load_state_dict(ckpt["policy_state_dict"]); pol.eval()

env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs/_smoke_bicycle", seed=17)
obs = env.reset("bicycle", episode_id=0)
print(f"[load] train_cams={len(list(env.scene.getTrainCameras()))} test_cams={len(list(env.scene.getTestCameras()))} init_gaussians={env.gaussians.get_xyz.shape[0]}", flush=True)
CONT = DISCRETE_ACTIONS["stop"].index("continue")
done = False; step = 0
while not done and step < 4:
    a, _, _ = pol.act(torch.as_tensor(obs, dtype=torch.float32, device=dev), deterministic=True)
    a["discrete"]["stop"] = CONT
    obs, r, done, info = env.step(a)
    vram = torch.cuda.max_memory_allocated() / 1e9
    print(f"[step {step}] iter={info['iteration']} gaussians={env.gaussians.get_xyz.shape[0]} reward={r:.3f} vram={vram:.1f}GB", flush=True)
    step += 1
env.close()
print("SMOKE OK", flush=True)
