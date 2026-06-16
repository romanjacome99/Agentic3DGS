# Extended time-to-target sweeps for the v7 acceleration policy:
#   1. hotdog grid extended to 7000 iters (reuses existing 100-5000 samples)
#   2. plot hotdog
#   3. drums full grid (training scene; lower PSNR ceiling -> lower targets)
#   4. plot drums
# Runs sequentially on the single GPU; plotting is CPU-only.

$ErrorActionPreference = "Continue"
$ckpt   = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_150u_v7_tau45\checkpoints\best.pth"
$cfg    = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\configs\phase1_eval.json"
$ttp    = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot   = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$grid   = @("100","250","500","750","1000","1500","2000","2500","3000","4000","5000","6000","7000")
$hotReport = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\hotdog_time_to_target_v7_best"
$drmReport = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\drums_time_to_target_v7_best"

Write-Output "=== STAGE 1: hotdog extended grid (to 7000) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
    --checkpoint $ckpt --config $cfg --scene hotdog `
    --run-prefix unified_v7b_hotdog_tt --report-dir $hotReport `
    --sample-iterations @grid `
    --targets 25 27 29 31 33 35 `
    --run-missing --skip-render-images --validation-cameras 12
Write-Output "=== STAGE 1 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 2: plot hotdog ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $hotReport
Write-Output "=== STAGE 2 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 3: drums full grid ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
    --checkpoint $ckpt --config $cfg --scene drums `
    --run-prefix unified_v7_drums_tt --report-dir $drmReport `
    --sample-iterations @grid `
    --targets 18 20 22 24 `
    --run-missing --skip-render-images --validation-cameras 12
Write-Output "=== STAGE 3 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 4: plot drums ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $drmReport
Write-Output "=== STAGE 4 exit: $LASTEXITCODE ==="

Write-Output "=== ALL DONE ==="
