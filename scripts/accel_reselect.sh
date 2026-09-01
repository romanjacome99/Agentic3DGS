#!/usr/bin/env bash
# Compaction-aware acceleration checkpoint investigation (10-scene). The accel_auc
# probe picks by EARLY quality-vs-time and can select checkpoints that balloon at
# 30k (3dgs u279 -> 1.54M). Run candidate accel checkpoints to 30k (agentic-only,
# force-full) and record @30k Gaussian count + quality + time-to-target curve, to
# find a fast-AND-compact operating point. Speedups computed later vs the existing
# baseline curve (protocol_*_train_30k/curve.csv), which is unchanged.
set -u
cd /c/Roman/3DGS_PROPOSAL
one() { # env backend update
  local env="$1" bk="$2" u="$3"
  local ck=outputs/agentic_rl_real/final_accel_${bk}/checkpoints/policy_update_${u}.pth
  local out=outputs/_accel_reselect/${bk}_u${u}
  [ -f "$out/curve.csv" ] && { echo "SKIP ${bk}_u${u}"; return 0; }
  echo ">>> accel_reselect ${bk}_u${u}"
  conda run --no-capture-output -n "$env" python scripts/eval_protocol.py \
    --config configs/final_accel_${bk}.json --checkpoint "$ck" --scene train \
    --max-iter 30000 --methods agentic --targets 16 17 18 19 20 21 --out "$out"
  echo "    exit=$?"
}
for u in 0019 0199 0289; do one env_pytorch_3dgs 3dgs $u; done
for u in 0009 0109;      do one fgs_cu128        fastergs $u; done
echo "=== ACCEL_RESELECT_DONE ==="
