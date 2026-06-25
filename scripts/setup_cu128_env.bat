@echo off
REM Create an isolated env with CUDA 12.8 toolkit + PyTorch cu128 to test whether
REM the FasterGS rasterizer's backward gradients are correct under 12.8.
echo === create env ===
call conda create -y -n fgs_cu128 python=3.11
echo CREATE_EXIT=%ERRORLEVEL%
echo === install CUDA 12.8 toolkit (nvcc + dev libs) from nvidia channel ===
call conda install -y -n fgs_cu128 -c nvidia "cuda-toolkit=12.8"
echo TOOLKIT_EXIT=%ERRORLEVEL%
echo === install torch cu128 + numpy ===
call conda run -n fgs_cu128 python -m pip install numpy
call conda run -n fgs_cu128 python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
echo TORCH_EXIT=%ERRORLEVEL%
echo === versions ===
call conda run -n fgs_cu128 python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
call conda run -n fgs_cu128 nvcc --version
echo SETUP_DONE
