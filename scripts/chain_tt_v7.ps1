# Waits for the v7 PPO training process to exit, then runs the hotdog
# time-to-target protocol with the v7 best.pth checkpoint.
param(
    [int]$TrainPid = 12128
)
try {
    Wait-Process -Id $TrainPid -ErrorAction Stop
} catch {
    # Process already exited - proceed.
}
$log = "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\tt_v7_hotdog.log"
& conda run --no-capture-output -n env_pytorch_3dgs python -u "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\scripts\time_to_target_protocol.py" `
    --checkpoint "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1\ppo_accel_150u_v7_tau45\checkpoints\best.pth" `
    --config "C:\Roman\3DGS_PROPOSAL\agentic_gs_phase1\configs\phase1_eval.json" `
    --scene hotdog `
    --run-prefix unified_v7_best_hotdog_tt `
    --report-dir "C:\Roman\3DGS_PROPOSAL\outputs\agentic_gs_phase1_reports\hotdog_time_to_target_v7_best" `
    --run-missing --skip-render-images --validation-cameras 12 *> $log
