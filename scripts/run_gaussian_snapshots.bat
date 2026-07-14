@echo off
REM Save Gaussian point-cloud snapshots (.ply) along training for the agent and the
REM fixed-schedule baseline, in BOTH the acceleration and budget-conditioned regimes,
REM on BOTH backends. Snapshots at a shared set of wall-clock times so the evolution is
REM directly comparable across cases. Run when the GPU is free.
cd /d C:\Roman\3DGS_PROPOSAL
set "OUT=outputs\gaussian_evolution"
set "SNAPS=5 15 30 60 120"
set "AVIEWS=--views 12 --scene train --seed 0"

set "ADG=outputs\agentic_rl_real\ppo_real_3dgs_v1\checkpoints\best.pth"
set "AFG=outputs\agentic_rl_real\ppo_real_fastergs_v1\checkpoints\best.pth"
set "BDG=outputs\agentic_rl_budget\ppo_budget_3dgs_v2\checkpoints\best.pth"
set "BFG=outputs\agentic_rl_budget\ppo_budget_fastergs_v2\checkpoints\best.pth"

REM ---------------- 3DGS backend ----------------
echo === 3DGS acceleration AGENT ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\save_gaussian_snapshots.py --config configs\real_3dgs_train.json  --mode acceleration --method agentic  --checkpoint %ADG% --horizon 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\3dgs_accel_agent
echo === 3DGS acceleration BASELINE ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\save_gaussian_snapshots.py --config configs\real_3dgs_train.json  --mode acceleration --method baseline --horizon 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\3dgs_accel_baseline
echo === 3DGS budget AGENT (B=120) ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\save_gaussian_snapshots.py --config configs\real_3dgs_budget.json --mode budget --method agentic  --checkpoint %BDG% --budget 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\3dgs_budget_agent
echo === 3DGS budget BASELINE (B=120) ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\save_gaussian_snapshots.py --config configs\real_3dgs_budget.json --mode budget --method baseline --budget 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\3dgs_budget_baseline

REM ---------------- FasterGS backend ----------------
echo === FasterGS acceleration AGENT ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\save_gaussian_snapshots.py --config configs\real_fastergs_train.json  --mode acceleration --method agentic  --checkpoint %AFG% --horizon 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\fastergs_accel_agent
echo === FasterGS acceleration BASELINE ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\save_gaussian_snapshots.py --config configs\real_fastergs_train.json  --mode acceleration --method baseline --horizon 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\fastergs_accel_baseline
echo === FasterGS budget AGENT (B=120) ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\save_gaussian_snapshots.py --config configs\real_fastergs_budget.json --mode budget --method agentic  --checkpoint %BFG% --budget 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\fastergs_budget_agent
echo === FasterGS budget BASELINE (B=120) ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\save_gaussian_snapshots.py --config configs\real_fastergs_budget.json --mode budget --method baseline --budget 120 --snap-times %SNAPS% %AVIEWS% --out %OUT%\fastergs_budget_baseline

echo === ALL_DONE ===
