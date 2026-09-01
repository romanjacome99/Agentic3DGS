#!/usr/bin/env bash
# Same time-to-target protocol as run_accel_aug_eval.sh but for the NON-AUGMENTED
# accel agents (final_accel_{fastergs,dash}/selected_accel.pth). Lets us compare
# augmented-agent speedup vs non-augmented-agent speedup (each vs its own fresh
# baseline). Scenes train + bicycle, same targets/configs. Sequential, per-backend env.
set -e
cd /c/Roman/3DGS_PROPOSAL

FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
DASH_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
MAXITER=30000
FGS_CKPT="outputs/agentic_rl_real/final_accel_fastergs/checkpoints/selected_accel.pth"
DASH_CKPT="outputs/agentic_rl_real/final_accel_dash/checkpoints/selected_accel.pth"

run() {
  local py="$1" cfg="$2" scene="$3" out="$4"; shift 4
  echo "=================================================================="
  echo ">>> $out  (scene=$scene, targets=$*)"
  echo "=================================================================="
  "$py" scripts/eval_protocol.py --config "$cfg" --checkpoint "$CKPT" \
    --scene "$scene" --max-iter "$MAXITER" --targets "$@" --out "$out"
}

# ---- FasterGS (env fgs_cu128) ----
CKPT="$FGS_CKPT"
run "$FGS_PY" configs/final_accel_fastergs.json         train   outputs/agentic_rl_real/protocol_nonaug_fastergs_train   16 17 18 19 20 21
run "$FGS_PY" configs/final_accel_fastergs_bicycle.json bicycle outputs/agentic_rl_real/protocol_nonaug_fastergs_bicycle 20 21 22 23 24 25

# ---- Dash (env env_pytorch_3dgs) ----
CKPT="$DASH_CKPT"
run "$DASH_PY" configs/final_accel_dash.json         train   outputs/agentic_rl_real/protocol_nonaug_dash_train   16 17 18 19 20 21
run "$DASH_PY" configs/final_accel_dash_bicycle.json bicycle outputs/agentic_rl_real/protocol_nonaug_dash_bicycle 20 21 22 23 24 25

echo "ALL NONAUG ACCEL EVALS DONE"
