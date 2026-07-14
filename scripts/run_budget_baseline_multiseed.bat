@echo off
REM Matched 10-seed FIXED-SCHEDULE BASELINE budget sweep for both backends, so the
REM per-budget PSNR/SSIM comparison in the paper has baseline mean/std (not a single run).
cd /d C:\Roman\3DGS_PROPOSAL
set "OUT=outputs\agentic_rl_budget\multiseed10_baseline"
set "ARGS=--budgets 30 60 120 300 360 --unlimited --seeds 0 1 2 3 4 5 6 7 8 9 --scene train --views 12"

echo === FasterGS BASELINE (10 seeds) ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\budget_sweep_multi.py --config configs\real_fastergs_budget.json --method baseline %ARGS% --out %OUT%\fastergs_baseline
echo EXIT1=%ERRORLEVEL%

echo === 3DGS BASELINE (10 seeds) ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\budget_sweep_multi.py --config configs\real_3dgs_budget.json --method baseline %ARGS% --out %OUT%\3dgs_baseline
echo EXIT2=%ERRORLEVEL%
echo === ALL_DONE ===
