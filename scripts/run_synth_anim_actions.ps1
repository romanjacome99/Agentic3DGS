# Re-render the synthetic agentic trajectories (hotdog/drums/materials) to 7k with
# per-block action logging, so the scrubber action bar shows for those scenes too.
# Baseline frames are reused. Rebuild scrubber assets at the end.
$ErrorActionPreference = "Continue"
$base  = "C:\Roman\3DGS_PROPOSAL"
$ckpt  = "$base\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$rend  = "$base\agentic_gs_phase1\scripts\render_training_frames.py"
$build = "$base\agentic_gs_phase1\scripts\build_scrubber_assets.py"
$anim  = "$base\outputs\agentic_gs_phase1_reports\training_anim"

foreach ($scene in @("hotdog","drums","materials")) {
  Write-Output "=== render $scene agentic (to 7k, +actions) ==="
  & conda run --no-capture-output -n env_pytorch_3dgs python -u $rend `
      --scene $scene --method agentic --checkpoint $ckpt `
      --output-dir "$anim\${scene}_agentic" --view-index 0 --frame-size 400 --max-iter 7000
  Write-Output "=== $scene exit: $LASTEXITCODE ==="
}

Write-Output "=== rebuild scrubber assets (all scenes, +actions) ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $build
Write-Output "=== rebuild exit: $LASTEXITCODE ==="
Write-Output "=== ALL DONE ==="
