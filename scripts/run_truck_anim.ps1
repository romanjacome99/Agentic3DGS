# Render Truck (real scene) reconstruction frames to 30k force-full for baseline
# and agentic, then rebuild the interactive scrubber assets.
$ErrorActionPreference = "Continue"
$base  = "C:\Roman\3DGS_PROPOSAL"
$ckpt  = "$base\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg   = "$base\configs\real_truck_eval.json"
$rend  = "$base\agentic_gs_phase1\scripts\render_training_frames.py"
$build = "$base\agentic_gs_phase1\scripts\build_scrubber_assets.py"
$anim  = "$base\outputs\agentic_gs_phase1_reports\training_anim"

Write-Output "=== STAGE 1: render Truck baseline frames (to 30k) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
    --scene truck --method baseline --config $cfg `
    --output-dir "$anim\truck_baseline" --view-index 0 --frame-size 400 --max-iter 30000
Write-Output "=== STAGE 1 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 2: render Truck agentic frames (to 30k, force-full) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
    --scene truck --method agentic --checkpoint $ckpt --config $cfg `
    --output-dir "$anim\truck_agentic" --view-index 0 --frame-size 400 --max-iter 30000
Write-Output "=== STAGE 2 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 3: rebuild scrubber assets (incl. truck) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $build
Write-Output "=== STAGE 3 exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
