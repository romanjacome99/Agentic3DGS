$ErrorActionPreference="Continue"
Set-Location C:\Roman\3DGS_PROPOSAL
& conda run --no-capture-output -n env_pytorch_3dgs python -m pip install .\faster-gaussian-splatting\FasterGSCudaBackend --no-build-isolation -v
Write-Output "=== PIP EXIT: $LASTEXITCODE ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -c "import FasterGSCudaBackend; print('IMPORT OK', FasterGSCudaBackend.__file__)"
Write-Output "=== IMPORT EXIT: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
