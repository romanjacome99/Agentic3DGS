@echo off
REM Build FasterGSCudaBackend inside the fgs_cu128 env (CUDA 12.8 + torch cu128)
REM to test whether the backward-gradient bug is CUDA-version specific.
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"

call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
call "C:\Users\User\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "DISTUTILS_USE_SDK=1"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
REM Force torch to use the env's CUDA 12.8 toolkit, NOT the system v11.8 on PATH.
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"
REM Make the linker find cudart.lib (CUDA 12.8 toolkit libs from the conda env).
set "LIB=%LIB%;%CUDA_ENV%\Library\lib;%CUDA_ENV%\Library\lib\x64;%CUDA_ENV%\lib\x64"

cd /d C:\Roman\3DGS_PROPOSAL
echo === which nvcc ===
where nvcc
call conda run -n fgs_cu128 nvcc --version
echo === force clean rebuild ===
rmdir /s /q faster-gaussian-splatting\FasterGSCudaBackend\build 2>nul
echo === build ===
python -m pip install ".\faster-gaussian-splatting\FasterGSCudaBackend" --no-build-isolation --force-reinstall --no-deps
echo PIP_EXIT=%ERRORLEVEL%
python -c "import FasterGSCudaBackend; print('IMPORT OK')"
echo IMPORT_EXIT=%ERRORLEVEL%
echo ALL_DONE
