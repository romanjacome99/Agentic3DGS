@echo off
REM Train the agentic-3DGS policy on real COLMAP scenes using the FasterGS backend
REM (CUDA 12.8 env). Resumable: checkpoints saved every update under the run dir.
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"
call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"

cd /d C:\Roman\3DGS_PROPOSAL
python -u agentic_gs_phase1\scripts\train_agent.py ^
  --config configs\real_fastergs_train.json ^
  --run-name ppo_real_fastergs_v1 ^
  --output-root outputs\agentic_rl_real ^
  --max-updates 150
echo TRAIN_EXIT=%ERRORLEVEL%
echo ALL_DONE
