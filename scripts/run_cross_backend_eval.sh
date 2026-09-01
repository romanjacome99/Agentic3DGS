#!/usr/bin/env bash
# Cross-backend validation (3dgs <-> dash, both 45-dim base policies), scene train.
# agent = transferred policy; baseline = the TARGET backend's fixed schedule.
set -e
cd /c/Roman/3DGS_PROPOSAL
PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
T="16 17 18 19 20 21"
C3=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
CD=outputs/agentic_rl_real/final_accel_dash/checkpoints/selected_accel.pth
echo "### 3DGS policy -> DASH backend"
"$PY" scripts/eval_protocol.py --config configs/final_accel_dash.json --checkpoint "$C3" \
  --scene train --max-iter 30000 --targets $T --out outputs/agentic_rl_real/protocol_cross_3dgsPolicy_on_dash
echo "### DASH policy -> 3DGS backend"
"$PY" scripts/eval_protocol.py --config configs/final_accel_3dgs.json --checkpoint "$CD" \
  --scene train --max-iter 30000 --targets $T --out outputs/agentic_rl_real/protocol_cross_dashPolicy_on_3dgs
echo "CROSS-BACKEND EVAL DONE"
