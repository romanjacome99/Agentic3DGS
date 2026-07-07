@echo off
REM "Ultimate" budget evaluation on held-out train: budget-agent vs baseline
REM trainer, both backends, across budgets 30/60/120/300/360s + unlimited(30k).
cd /d C:\Roman\3DGS_PROPOSAL
set "OUT=outputs\agentic_rl_budget\ultimate"
set "B=--budgets 30 60 120 300 360 --unlimited --scene train --views 12"

echo === 1/4 FasterGS AGENT ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\budget_sweep.py --config configs\real_fastergs_budget.json --method agentic --checkpoint outputs\agentic_rl_budget\ppo_budget_fastergs_v2\checkpoints\best.pth %B% --out %OUT%\fastergs_agent
echo EXIT1=%ERRORLEVEL%

echo === 2/4 FasterGS BASELINE ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\budget_sweep.py --config configs\real_fastergs_budget.json --method baseline %B% --out %OUT%\fastergs_baseline
echo EXIT2=%ERRORLEVEL%

echo === 3/4 3DGS AGENT ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\budget_sweep.py --config configs\real_3dgs_budget.json --method agentic --checkpoint outputs\agentic_rl_budget\ppo_budget_3dgs_v2\checkpoints\best.pth %B% --out %OUT%\3dgs_agent
echo EXIT3=%ERRORLEVEL%

echo === 4/4 3DGS BASELINE ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\budget_sweep.py --config configs\real_3dgs_budget.json --method baseline %B% --out %OUT%\3dgs_baseline
echo EXIT4=%ERRORLEVEL%
echo === ALL_DONE ===
