@echo off
REM Run the FasterGS fork's OWN train.py on hotdog under cu128, to test whether the
REM authors' standard-codebase integration trains correctly (with full densification).
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"
call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"

cd /d C:\Roman\3DGS_PROPOSAL\gaussian-splatting-fastergs
python train.py ^
  -s C:\Roman\3DGS_PROPOSAL\archive\nerf_synthetic\hotdog ^
  -m C:\Roman\3DGS_PROPOSAL\outputs\_fork_train_hotdog ^
  --eval -w --iterations 3000 --test_iterations 1000 3000 --save_iterations 3000 ^
  --densify_until_iter 3000
echo TRAIN_EXIT=%ERRORLEVEL%
echo ALL_DONE
