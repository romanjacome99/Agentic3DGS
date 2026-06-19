# v8 force-full (stop head OFF) extended time-to-target for drums + materials.
$ErrorActionPreference = "Continue"
$ckpt = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_500u_v8_resume144\checkpoints\best.pth"
$cfg  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\configs\phase1_eval.json"
$ttp  = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\time_to_target_protocol.py"
$plot = "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\plot_time_to_target_artifacts.py"
$grid = "1000","2000","3000","5000","7000","10000","15000","30000"

function Run-Scene($scene, $targets, $prefix, $report) {
    Write-Output "=== STAGE: $scene force-full sweep ==="
    & conda run --no-capture-output -n env_pytorch_3dgs python -u $ttp `
        --checkpoint $ckpt --config $cfg --scene $scene `
        --run-prefix $prefix --report-dir $report `
        --sample-iterations @grid --targets @targets `
        --force-full-iterations --run-missing --skip-render-images --validation-cameras 12
    Write-Output "=== $scene TT exit: $LASTEXITCODE ==="
    & conda run --no-capture-output -n env_pytorch_3dgs python -u $plot --report-dir $report
    Write-Output "=== $scene plot exit: $LASTEXITCODE ==="
}

Run-Scene "drums"     @("22","24","26")        "unified_v8ff_drums_tt"     "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\drums_time_to_target_v8_forcefull"
Run-Scene "materials" @("24","26","28")        "unified_v8ff_materials_tt" "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\materials_time_to_target_v8_forcefull"
Write-Output "=== ALL DONE ==="
