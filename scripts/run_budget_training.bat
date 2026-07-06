@echo off
REM Train both budget-conditioned agents (FasterGS + 3DGS) on the real-scene pool.
REM Each samples a wall-clock training budget per episode (20-300s) and adapts.
cd /d C:\Roman\3DGS_PROPOSAL

echo === FasterGS budget-conditioned (fgs_cu128) ===
call conda run --no-capture-output -n fgs_cu128 python -u agentic_gs_phase1\scripts\train_agent.py ^
  --config configs\real_fastergs_budget.json ^
  --run-name ppo_budget_fastergs_v1 ^
  --output-root outputs\agentic_rl_budget ^
  --max-updates 150
echo FGS_EXIT=%ERRORLEVEL%

echo === 3DGS budget-conditioned (env_pytorch_3dgs) ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u agentic_gs_phase1\scripts\train_agent.py ^
  --config configs\real_3dgs_budget.json ^
  --run-name ppo_budget_3dgs_v1 ^
  --output-root outputs\agentic_rl_budget ^
  --max-updates 150
echo TDG_EXIT=%ERRORLEVEL%
echo === ALL_DONE ===
