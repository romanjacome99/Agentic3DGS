#!/usr/bin/env bash
# Acceleration time-to-target eval on the held-out unbounded scene `stump` (Mip-NeRF360),
# added as extra rows to Table 3. Reuses the compaction-aware accel checkpoints
# (selected_accel.pth: 3dgs u289 / fastergs u99). Sequential (wall-clock is measured).
set -u
cd /c/Roman/3DGS_PROPOSAL
one() { # env backend
  local env="$1" bk="$2"
  local ck=outputs/agentic_rl_real/final_accel_${bk}/checkpoints/selected_accel.pth
  local out=outputs/agentic_rl_real/protocol_${bk}_stump_30k
  echo ">>> stump ${bk}"
  conda run --no-capture-output -n "$env" python scripts/eval_protocol.py \
    --config configs/final_accel_${bk}_stump.json --checkpoint "$ck" \
    --scene stump --max-iter 30000 --targets 16 17 18 19 20 21 --out "$out"
  echo "    exit=$?"
}
one env_pytorch_3dgs 3dgs
one fgs_cu128        fastergs
echo "=== EVAL_STUMP_DONE ==="
