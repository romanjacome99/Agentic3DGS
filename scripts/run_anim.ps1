# Render baseline + agentic training frames for 3 scenes, then assemble animations.
$ErrorActionPreference = "Continue"
$ckpt = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\configs\phase1_eval.json"
$rdr  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\render_training_frames.py"
$mk   = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\make_animation.py"
$root = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\training_anim"
$scenes = "hotdog","drums","materials"

foreach ($s in $scenes) {
  Write-Output "=== RENDER $s baseline ==="
  & conda run --no-capture-output -n env_pytorch_3dgs python -u $rdr --scene $s --method baseline `
      --config $cfg --output-dir "$root\${s}_baseline" --max-iter 7000 --frame-size 400 --view-index 0
  Write-Output "=== RENDER $s agentic ==="
  & conda run --no-capture-output -n env_pytorch_3dgs python -u $rdr --scene $s --method agentic --checkpoint $ckpt `
      --config $cfg --output-dir "$root\${s}_agentic" --max-iter 7000 --frame-size 400 --view-index 0
}

Write-Output "=== ASSEMBLE animations ==="
& conda run --no-capture-output -n env_pytorch_3dgs python -u $mk --root $root --out-dir "$root\animations" --fps 3 --size 400
Write-Output "=== ALL DONE ==="
