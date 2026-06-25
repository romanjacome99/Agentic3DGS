@echo off
REM Build the official diff_gaussian_rasterization in fgs_cu128. The FasterGS fork
REM hard-imports it at module top (for its original non-faster render path), even
REM though FasterGS-Agent only uses faster_render. Same build recipe as the others.
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"

call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
call "C:\Users\User\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "DISTUTILS_USE_SDK=1"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"
set "LIB=%LIB%;%CUDA_ENV%\Library\lib"

cd /d C:\Roman\3DGS_PROPOSAL
echo === build diff_gaussian_rasterization ===
python -m pip install ".\gaussian-splatting\submodules\diff-gaussian-rasterization" --no-build-isolation --force-reinstall --no-deps
echo PIP_EXIT=%ERRORLEVEL%
echo ALL_DONE
