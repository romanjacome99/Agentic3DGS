#!/usr/bin/env bash
# In-distribution acceleration check: run the AUGMENTED accel agents on the
# TRAINING-SET unbounded Mip-360 scenes (garden, flowers, treehill), at the SAME
# setting bicycle used (mip360 / images_4, targets 20-25) so it's comparable to
# the held-out bicycle result. FasterGS aug (lr3e-4) + Dash aug (lr1e-3).
set -e
cd /c/Roman/3DGS_PROPOSAL
FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
DASH_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
FGS_CKPT="outputs/agentic_rl_real/real_fastergs_accel_aug/checkpoints/best.pth"
DASH_CKPT="outputs/agentic_rl_real/real_dash_accel_aug_lr1e3/checkpoints/best.pth"
T="20 21 22 23 24 25"
for s in garden flowers treehill; do
  echo "=== FGS $s ==="
  "$FGS_PY" scripts/eval_protocol.py --config configs/final_accel_fastergs_mip360.json \
    --checkpoint "$FGS_CKPT" --scene "$s" --max-iter 30000 --targets $T \
    --out "outputs/agentic_rl_real/protocol_augcheck_fastergs_$s"
done
for s in garden flowers treehill; do
  echo "=== DASH $s ==="
  "$DASH_PY" scripts/eval_protocol.py --config configs/final_accel_dash_mip360.json \
    --checkpoint "$DASH_CKPT" --scene "$s" --max-iter 30000 --targets $T \
    --out "outputs/agentic_rl_real/protocol_augcheck_dash_$s"
done
echo "MIP360 AUG-CHECK DONE"
