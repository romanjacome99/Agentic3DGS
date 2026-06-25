@echo off
REM Install the remaining pipeline deps + build simple_knn in fgs_cu128 so the
REM full FasterGS-Agent (env + fork scene/model) runs end-to-end under CUDA 12.8.
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
echo === python deps ===
python -m pip install numpy plyfile Pillow tqdm opencv-python
echo DEPS_EXIT=%ERRORLEVEL%
echo === build simple_knn (CUDA ext) ===
python -m pip install ".\gaussian-splatting\submodules\simple-knn" --no-build-isolation --force-reinstall --no-deps
echo SIMPLEKNN_EXIT=%ERRORLEVEL%
python -c "import simple_knn._C; print('simple_knn OK')"
echo SKNN_IMPORT_EXIT=%ERRORLEVEL%
echo ALL_DONE
