@echo off
REM Re-run the two AGENT combos to capture per-budget action profiles
REM (action_profiles.json). Deterministic -> the sweep values are unchanged.
cd /d C:\Roman\3DGS_PROPOSAL
set "OUT=outputs\agentic_rl_budget\ultimate"
set "B=--budgets 30 60 120 300 360 --unlimited --scene train --views 12"

echo === FasterGS AGENT (with actions) ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\budget_sweep.py --config configs\real_fastergs_budget.json --method agentic --checkpoint outputs\agentic_rl_budget\ppo_budget_fastergs_v2\checkpoints\best.pth %B% --out %OUT%\fastergs_agent
echo EXIT1=%ERRORLEVEL%

echo === 3DGS AGENT (with actions) ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\budget_sweep.py --config configs\real_3dgs_budget.json --method agentic --checkpoint outputs\agentic_rl_budget\ppo_budget_3dgs_v2\checkpoints\best.pth %B% --out %OUT%\3dgs_agent
echo EXIT2=%ERRORLEVEL%
echo === ALL_DONE ===
