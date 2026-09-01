#!/usr/bin/env bash
# Fill gaps: NON-AUGMENTED accel agents (selected_accel.pth) at FULL-RES on
# train + bicycle, targets 16-21, to match the augmented/relaxed full-res evals.
# Missing cells only (fastergs/dash train full-res already exist).
set -e
cd /c/Roman/3DGS_PROPOSAL
FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
D3_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
T="16 17 18 19 20 21"
ep(){ local py="$1" cfg="$2" ck="$3" scene="$4" out="$5"
  echo "=== $out ($scene) ==="
  "$py" scripts/eval_protocol.py --config "$cfg" --checkpoint "$ck" --scene "$scene" --max-iter 30000 --targets $T --out "$out"; }
C3=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
CF=outputs/agentic_rl_real/final_accel_fastergs/checkpoints/selected_accel.pth
CD=outputs/agentic_rl_real/final_accel_dash/checkpoints/selected_accel.pth
ep "$D3_PY" configs/final_accel_3dgs.json     "$C3" train   outputs/agentic_rl_real/protocol_nonaug_3dgs_train_fr
ep "$D3_PY" configs/final_accel_3dgs.json     "$C3" bicycle outputs/agentic_rl_real/protocol_nonaug_3dgs_bicycle_fr
ep "$FGS_PY" configs/final_accel_fastergs.json "$CF" bicycle outputs/agentic_rl_real/protocol_nonaug_fastergs_bicycle_fr
ep "$D3_PY" configs/final_accel_dash.json     "$CD" bicycle outputs/agentic_rl_real/protocol_nonaug_dash_bicycle_fr
echo "NONAUG FULLRES EVAL DONE"
