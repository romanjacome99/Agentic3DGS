@echo off
REM Fork train.py on a COLMAP scene (truck) under cu128 - the authors' target domain.
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"
call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"

cd /d C:\Roman\3DGS_PROPOSAL\gaussian-splatting-fastergs
python train.py ^
  -s C:\Roman\3DGS_PROPOSAL\archive\real_scenes\tandt\truck ^
  -m C:\Roman\3DGS_PROPOSAL\outputs\_fork_train_truck ^
  --eval -r 4 --iterations 2000 --test_iterations 1000 2000 --save_iterations 2000
echo TRAIN_EXIT=%ERRORLEVEL%
echo ALL_DONE
