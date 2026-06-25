@echo off
REM Build/install the FasterGSCudaBackend PyTorch extension in env_pytorch_3dgs.
REM Order matters: activate conda first, then vcvars64 LAST so the MSVC/Windows-SDK
REM INCLUDE/LIB/PATH win over the env's (broken) vs2017 activate.d shim.
REM CUDA 12.1 nvcc does not officially support MSVC 14.44, so allow it explicitly.

call "C:\Users\User\anaconda3\Scripts\activate.bat" env_pytorch_3dgs
call "C:\Users\User\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat"

set "DISTUTILS_USE_SDK=1"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
REM torch's build does not add a CUDA lib dir (and CUDA_PATH points at a mismatched
REM system toolkit), so the linker can't find cudart.lib. Add the env's CUDA libs
REM (12.1, matching torch) to the linker search path (link.exe reads %LIB%).
set "LIB=%LIB%;C:\Users\User\anaconda3\envs\env_pytorch_3dgs\Library\lib;C:\Users\User\anaconda3\envs\env_pytorch_3dgs\lib\x64"

cd /d C:\Roman\3DGS_PROPOSAL
echo === which cl ===
where cl
echo === which nvcc ===
where nvcc
echo === building FasterGSCudaBackend ===
python -m pip install ".\faster-gaussian-splatting\FasterGSCudaBackend" --no-build-isolation
echo PIP_EXIT=%ERRORLEVEL%
echo === import check ===
python -c "import FasterGSCudaBackend, torch; print('IMPORT OK', FasterGSCudaBackend.__file__)"
echo IMPORT_EXIT=%ERRORLEVEL%
echo ALL_DONE
