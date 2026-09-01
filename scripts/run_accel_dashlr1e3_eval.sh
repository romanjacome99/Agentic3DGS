#!/usr/bin/env bash
# Time-to-target eval for the Dash lr=1e-3 augmented accel agent (train + bicycle).
set -e
cd /c/Roman/3DGS_PROPOSAL
PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
CKPT="outputs/agentic_rl_real/real_dash_accel_aug_lr1e3/checkpoints/best.pth"
run(){ local cfg="$1" scene="$2" out="$3"; shift 3
  echo "=== $out (scene=$scene targets=$*) ==="
  "$PY" scripts/eval_protocol.py --config "$cfg" --checkpoint "$CKPT" --scene "$scene" --max-iter 30000 --targets "$@" --out "$out"; }
run configs/final_accel_dash.json         train   outputs/agentic_rl_real/protocol_dashlr1e3_train   16 17 18 19 20 21
run configs/final_accel_dash_bicycle.json bicycle outputs/agentic_rl_real/protocol_dashlr1e3_bicycle 20 21 22 23 24 25
echo "DASH lr=1e-3 EVAL DONE"
