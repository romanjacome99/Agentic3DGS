@echo off
REM Run the time-to-target protocol force-full to the 30k reference depth for the
REM matched agent pair on both held-out scenes. Saves actions/Gaussians/curves.
cd /d C:\Roman\3DGS_PROPOSAL
set "FGS=outputs\agentic_rl_real\ppo_real_fastergs_v1\checkpoints\best.pth"
set "TDG=outputs\agentic_rl_real\ppo_real_3dgs_v1\checkpoints\best.pth"

echo === 1/4 FasterGS train 30k ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\eval_protocol.py --config configs\real_fastergs_train.json --checkpoint %FGS% --scene train --max-iter 30000 --targets 16 17 18 19 20 21 --out outputs\agentic_rl_real\protocol_fastergs_train_30k
echo EXIT_1=%ERRORLEVEL%

echo === 2/4 3DGS train 30k ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\eval_protocol.py --config configs\real_3dgs_train.json --checkpoint %TDG% --scene train --max-iter 30000 --targets 16 17 18 19 20 21 --out outputs\agentic_rl_real\protocol_3dgs_train_30k
echo EXIT_2=%ERRORLEVEL%

echo === 3/4 FasterGS bicycle 30k ===
call conda run --no-capture-output -n fgs_cu128 python -u scripts\eval_protocol.py --config configs\real_bicycle_eval_fastergs.json --checkpoint %FGS% --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 --out outputs\agentic_rl_real\protocol_fastergs_bicycle_30k
echo EXIT_3=%ERRORLEVEL%

echo === 4/4 3DGS bicycle 30k ===
call conda run --no-capture-output -n env_pytorch_3dgs python -u scripts\eval_protocol.py --config configs\real_bicycle_eval_3dgs.json --checkpoint %TDG% --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 --out outputs\agentic_rl_real\protocol_3dgs_bicycle_30k
echo EXIT_4=%ERRORLEVEL%

echo === ALL_DONE ===
