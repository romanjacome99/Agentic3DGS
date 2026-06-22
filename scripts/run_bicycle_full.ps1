# Full Mip-NeRF 360 bicycle evaluation (images_4) to 30k force-full:
# time-to-target speedup, plots, reconstruction frames (+actions), Gaussian
# evolution, and scrubber assets. Mirrors the Truck pipeline.
$ErrorActionPreference = "Continue"
$base = "C:\Roman\3DGS_PROPOSAL"
$ckpt = "$base\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "$base\configs\real_bicycle_eval.json"
$ttp  = "$base\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot = "$base\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$evo  = "$base\agentic_gs_phase1\scripts\analyze_gaussian_evolution.py"
$rend = "$base\agentic_gs_phase1\scripts\render_training_frames.py"
$build= "$base\agentic_gs_phase1\scripts\build_scrubber_assets.py"
$rep  = "$base\outputs\agentic_gs_phase1_reports\bicycle_time_to_target_v8_forcefull"
$anim = "$base\outputs\agentic_gs_phase1_reports\training_anim"

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

Write-Output "=== STAGE 3: render bicycle baseline frames (to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
    --scene bicycle --method baseline --config $cfg `
    --output-dir "$anim\bicycle_baseline" --view-index 0 --frame-size 400 --max-iter 30000
Write-Output "=== STAGE 3 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 4: render bicycle agentic frames (to 30k, +actions) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
    --scene bicycle --method agentic --checkpoint $ckpt --config $cfg `
    --output-dir "$anim\bicycle_agentic" --view-index 0 --frame-size 400 --max-iter 30000
Write-Output "=== STAGE 4 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 5: Gaussian evolution (force-full, to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $evo `
    --checkpoint $ckpt --config $cfg --scenes bicycle --max-iter 30000 --force-full `
    --output "$base\outputs\agentic_gs_phase1_reports\gaussian_evolution_bicycle_forcefull.json"
Write-Output "=== STAGE 5 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 6: rebuild scrubber assets (incl. bicycle) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $build
Write-Output "=== STAGE 6 exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
