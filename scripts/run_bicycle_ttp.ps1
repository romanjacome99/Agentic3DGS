# Mip-NeRF 360 bicycle (images_4): time-to-target PSNR protocol, force-full to 30k.
# Same protocol as the other scenes. Produces sampled_test_curve.csv + speedup tables.
$ErrorActionPreference = "Continue"
$base = "C:\Roman\3DGS_PROPOSAL"
$ckpt = "$base\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "$base\configs\real_bicycle_eval.json"
$ttp  = "$base\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot = "$base\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$rep  = "$base\outputs\agentic_gs_phase1_reports\bicycle_time_to_target_v8_forcefull"

Write-Output "=== STAGE 1: bicycle force-full time-to-target (to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
    --checkpoint $ckpt --config $cfg --scene bicycle `
    --run-prefix unified_bicycleff_tt --report-dir $rep `
    --sample-iterations 1000 2000 3000 5000 7000 10000 15000 30000 `
    --targets 20 22 23 24 25 `
    --force-full-iterations --run-missing --skip-render-images --validation-cameras 12
Write-Output "=== STAGE 1 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 2: plot ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $rep
Write-Output "=== STAGE 2 exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
