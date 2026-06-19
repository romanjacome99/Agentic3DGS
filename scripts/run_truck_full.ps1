# Full real-scene (Truck) evaluation to 30k: force-full time-to-target + Gaussian evolution.
$ErrorActionPreference = "Continue"
$ckpt = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "C:\Roman\3DGS_PROPOSAL\configs\real_truck_eval.json"
$ttp  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$evo  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\analyze_gaussian_evolution.py"
$rep  = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\truck_time_to_target_v8_forcefull"

Write-Output "=== STAGE 1: Truck force-full time-to-target (to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
    --checkpoint $ckpt --config $cfg --scene truck `
    --run-prefix unified_truckff_tt --report-dir $rep `
    --sample-iterations 1000 2000 3000 5000 7000 10000 15000 30000 `
    --targets 20 22 24 25 26 `
    --force-full-iterations --run-missing --skip-render-images --validation-cameras 12
Write-Output "=== STAGE 1 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 2: plot ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $rep
Write-Output "=== STAGE 2 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 3: Gaussian evolution (force-full, to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $evo `
    --checkpoint $ckpt --config $cfg --scenes truck --max-iter 30000 --force-full `
    --output "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\gaussian_evolution_truck_forcefull.json"
Write-Output "=== STAGE 3 exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
