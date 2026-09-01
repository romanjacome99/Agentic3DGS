# Environment setup (reproducibility)

Two conda envs drive the experiments; each backend runs in its own env.

| Env | Backend | Used by |
|-----|---------|---------|
| `env_pytorch_3dgs` | official 3DGS (CUDA 12.1) | `final_*_3dgs` configs |
| `fgs_cu128` | Faster-GS (CUDA 12.8) | `final_*_fastergs` configs |

> This file preserves the recipe from the now-removed `scripts/*cu128*.bat` /
> `scripts/build_*.bat` helpers (deleted 2026-07-23 in the script cleanup). The
> envs already exist on the training machine; this is the rebuild recipe.

## `fgs_cu128` (Faster-GS, CUDA 12.8)

```bat
:: 1. create env + CUDA 12.8 toolkit + torch cu128
conda create -y -n fgs_cu128 python=3.11
conda install -y -n fgs_cu128 -c nvidia "cuda-toolkit=12.8"
conda run -n fgs_cu128 python -m pip install numpy
conda run -n fgs_cu128 python -m pip install torch --index-url https://download.pytorch.org/whl/cu128

:: 2. activate env, THEN vcvars64 LAST (MSVC/SDK INCLUDE/LIB must win over the
::    env's vs2017 activate.d shim). Point CUDA_HOME at the env's 12.8 toolkit.
call "C:\Users\User\anaconda3\Scripts\activate.bat" fgs_cu128
call "C:\Users\User\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_ENV=C:\Users\User\anaconda3\envs\fgs_cu128"
set "DISTUTILS_USE_SDK=1"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "CUDA_HOME=%CUDA_ENV%\Library"
set "CUDA_PATH=%CUDA_ENV%\Library"
set "PATH=%CUDA_ENV%\Library\bin;%PATH%"
set "LIB=%LIB%;%CUDA_ENV%\Library\lib;%CUDA_ENV%\Library\lib\x64;%CUDA_ENV%\lib\x64"

:: 3. pipeline deps + CUDA extensions
python -m pip install numpy plyfile Pillow tqdm opencv-python
python -m pip install ".\gaussian-splatting\submodules\simple-knn"              --no-build-isolation --force-reinstall --no-deps
python -m pip install ".\gaussian-splatting\submodules\diff-gaussian-rasterization" --no-build-isolation --force-reinstall --no-deps
python -m pip install ".\faster-gaussian-splatting\FasterGSCudaBackend"          --no-build-isolation --force-reinstall --no-deps
python -c "import simple_knn._C, FasterGSCudaBackend; print('OK')"
```

**Torch fragility (see memory `fgs-env-torch-fragility`):** pip installs in this
env can clobber the cu128 torch build (e.g. `lpips` pulled a CPU torch and broke
the rasterizer). Always install extra deps with `--no-deps`; repair with
`pip install torch==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128`.

## `env_pytorch_3dgs` (official 3DGS, CUDA 12.1)

```bat
call "C:\Users\User\anaconda3\Scripts\activate.bat" env_pytorch_3dgs
call "C:\Users\User\VSBuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "DISTUTILS_USE_SDK=1"
set "NVCC_PREPEND_FLAGS=-allow-unsupported-compiler"
set "LIB=%LIB%;C:\Users\User\anaconda3\envs\env_pytorch_3dgs\Library\lib;C:\Users\User\anaconda3\envs\env_pytorch_3dgs\lib\x64"
python -m pip install ".\faster-gaussian-splatting\FasterGSCudaBackend" --no-build-isolation
python -c "import FasterGSCudaBackend, torch; print('IMPORT OK')"
```
