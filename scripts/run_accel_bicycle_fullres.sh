#!/usr/bin/env bash
# Test bicycle in the TRAINING-DATA setting: full-res rl/images (not images_4).
# base configs already point at archive_root=rl, images=images. Same augmented
# agents. Targets 16-21 to match the (full-res) 'train' scene table.
set -e
cd /c/Roman/3DGS_PROPOSAL
FGS_PY="$HOME/anaconda3/envs/fgs_cu128/python.exe"
DASH_PY="$HOME/anaconda3/envs/env_pytorch_3dgs/python.exe"
FGS_CKPT="outputs/agentic_rl_real/real_fastergs_accel_aug/checkpoints/best.pth"
DASH_CKPT="outputs/agentic_rl_real/real_dash_accel_aug_lr1e3/checkpoints/best.pth"
T="16 17 18 19 20 21"
echo "=== FGS bicycle (full-res) ==="
"$FGS_PY" scripts/eval_protocol.py --config configs/final_accel_fastergs.json \
  --checkpoint "$FGS_CKPT" --scene bicycle --max-iter 30000 --targets $T \
  --out outputs/agentic_rl_real/protocol_augcheck_fastergs_bicycle_fullres
echo "=== DASH bicycle (full-res) ==="
"$DASH_PY" scripts/eval_protocol.py --config configs/final_accel_dash.json \
  --checkpoint "$DASH_CKPT" --scene bicycle --max-iter 30000 --targets $T \
  --out outputs/agentic_rl_real/protocol_augcheck_dash_bicycle_fullres
echo "BICYCLE FULLRES DONE"
