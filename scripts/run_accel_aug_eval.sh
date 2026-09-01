#!/usr/bin/env bash
# Time-to-target acceleration eval for the AUGMENTED accel agents (FasterGS + Dash)
# on held-out test scenes (train, bicycle). 3dgs skipped (no accel-aug agent).
# Each eval_protocol.py run does baseline + agentic (force-full) and writes
# time_to_target.csv (speedups per target PSNR). Runs sequentially (single GPU),
# each backend in its matching conda env.
set -e
cd /c/Roman/3DGS_PROPOSAL

FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
DASH_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
MAXITER=30000
FGS_CKPT="outputs/agentic_rl_real/real_fastergs_accel_aug/checkpoints/best.pth"
DASH_CKPT="outputs/agentic_rl_real/real_dash_accel_aug/checkpoints/best.pth"

run() {  # $1=python $2=config $3=scene $4=out  $5..=targets
  local py="$1" cfg="$2" scene="$3" out="$4"; shift 4
  echo "=================================================================="
  echo ">>> $out  (scene=$scene, targets=$*)"
  echo "=================================================================="
  "$py" scripts/eval_protocol.py --config "$cfg" --checkpoint "$CKPT" \
    --scene "$scene" --max-iter "$MAXITER" --targets "$@" \
    --out "$out"
}

# ---- FasterGS (env fgs_cu128) ----
CKPT="$FGS_CKPT"
run "$FGS_PY" configs/final_accel_fastergs.json         train   outputs/agentic_rl_real/protocol_aug_fastergs_train   16 17 18 19 20 21
run "$FGS_PY" configs/final_accel_fastergs_bicycle.json bicycle outputs/agentic_rl_real/protocol_aug_fastergs_bicycle 20 21 22 23 24 25

# ---- Dash (env env_pytorch_3dgs) ----
CKPT="$DASH_CKPT"
run "$DASH_PY" configs/final_accel_dash.json         train   outputs/agentic_rl_real/protocol_aug_dash_train   16 17 18 19 20 21
run "$DASH_PY" configs/final_accel_dash_bicycle.json bicycle outputs/agentic_rl_real/protocol_aug_dash_bicycle 20 21 22 23 24 25

echo "ALL ACCEL-AUG EVALS DONE"
