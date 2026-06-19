# v8 (update-194) time-to-target sweeps for drums + materials, with plots.
$ErrorActionPreference = "Continue"
$ckpt = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\configs\phase1_eval.json"
$ttp  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$grid = "100","250","500","750","1000","1500","2000","2500","3000","4000","5000","6000","7000"

function Run-Scene($scene, $targets, $prefix, $report) {
    Write-Output "=== STAGE: $scene sweep ==="
    & conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
        --checkpoint $ckpt --config $cfg --scene $scene `
        --run-prefix $prefix --report-dir $report `
        --sample-iterations @grid --targets @targets `
        --run-missing --skip-render-images --validation-cameras 12
    Write-Output "=== $scene TT exit: $LASTEXITCODE ==="
    & conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $report
    Write-Output "=== $scene plot exit: $LASTEXITCODE ==="
}

Run-Scene "drums"     @("18","20","22","24")        "unified_v8_drums_tt"     "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\drums_time_to_target_v8_best"
Run-Scene "materials" @("20","22","24","26","28")   "unified_v8_materials_tt" "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\materials_time_to_target_v8_best"
Write-Output "=== ALL DONE ==="
