#!/usr/bin/env bash
# Gaussian-count-relaxed reward retrains, acceleration mode, augmented pool.
# Tests whether relaxing the count penalty lets the policy densify enough to
# accelerate high-frequency scenes. Sequential (single GPU), correct env each.
set -e
cd /c/Roman/3DGS_PROPOSAL
FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
D3_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
TA="agentic_gs_phase1.scripts.train_agent"

echo "########## 1/3 FasterGS relaxed ##########"
"$FGS_PY" -u -m $TA --config agentic_gs_phase1/configs/real_fastergs_accel_aug_relaxed.json --run-name real_fastergs_accel_aug_relaxed
echo "########## 2/3 Dash relaxed (lr=1e-3) ##########"
"$D3_PY" -u -m $TA --config agentic_gs_phase1/configs/real_dash_accel_aug_relaxed.json --run-name real_dash_accel_aug_relaxed
echo "########## 3/3 3DGS relaxed (first 3dgs accel-aug) ##########"
"$D3_PY" -u -m $TA --config agentic_gs_phase1/configs/real_3dgs_accel_aug_relaxed.json --run-name real_3dgs_accel_aug_relaxed
echo "ALL RELAXED TRAININGS DONE"
