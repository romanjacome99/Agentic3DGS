#!/usr/bin/env bash
# Decisive test: do the RELAXED-reward agents accelerate bicycle (full-res, training setting)?
set -e
cd /c/Roman/3DGS_PROPOSAL
FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
D3_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
T="16 17 18 19 20 21"
echo "=== FGS relaxed / bicycle ==="
"$FGS_PY" scripts/eval_protocol.py --config configs/final_accel_fastergs.json \
  --checkpoint outputs/agentic_rl_real/real_fastergs_accel_aug_relaxed/checkpoints/best.pth \
  --scene bicycle --max-iter 30000 --targets $T --out outputs/agentic_rl_real/protocol_relaxed_fastergs_bicycle
echo "=== Dash relaxed / bicycle ==="
"$D3_PY" scripts/eval_protocol.py --config configs/final_accel_dash.json \
  --checkpoint outputs/agentic_rl_real/real_dash_accel_aug_relaxed/checkpoints/best.pth \
  --scene bicycle --max-iter 30000 --targets $T --out outputs/agentic_rl_real/protocol_relaxed_dash_bicycle
echo "=== 3dgs relaxed / bicycle ==="
"$D3_PY" scripts/eval_protocol.py --config configs/final_accel_3dgs.json \
  --checkpoint outputs/agentic_gs_phase1/real_3dgs_accel_aug_relaxed/checkpoints/best.pth \
  --scene bicycle --max-iter 30000 --targets $T --out outputs/agentic_rl_real/protocol_relaxed_3dgs_bicycle
echo "RELAXED BICYCLE EVAL DONE"
