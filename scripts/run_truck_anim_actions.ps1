# Re-render ONLY the agentic Truck trajectory (now logging the per-block action
# the policy took), then rebuild the scrubber assets so each budget frame carries
# the agent's discrete + continuous action. Baseline frames are reused as-is.
$ErrorActionPreference = "Continue"
$base  = "C:\Roman\3DGS_PROPOSAL"
$ckpt  = "$base\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg   = "$base\configs\real_truck_eval.json"
$rend  = "$base\agentic_gs_phase1\scripts\render_training_frames.py"
$build = "$base\agentic_gs_phase1\scripts\build_scrubber_assets.py"
$anim  = "$base\outputs\agentic_gs_phase1_reports\training_anim"

Write-Output "=== STAGE 1: re-render Truck agentic frames (to 30k, force-full, +actions) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
    --scene truck --method agentic --checkpoint $ckpt --config $cfg `
    --output-dir "$anim\truck_agentic" --view-index 0 --frame-size 400 --max-iter 30000
Write-Output "=== STAGE 1 exit: $LASTEXITCODE ==="

Write-Output "=== STAGE 2: rebuild scrubber assets (incl. truck actions) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $build
Write-Output "=== STAGE 2 exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
