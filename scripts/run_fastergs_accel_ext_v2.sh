#!/usr/bin/env bash
# Acceleration-mode FasterGS-ext v2: AA action removed, save_episode_models=false (fixes the
# disk-full crash of the previous accel-ext run), acceleration reward unchanged (time-value
# already front-loads quality). 300 PPO updates, resumable.
cd /c/Roman/3DGS_PROPOSAL
name=ppo_real_fastergs_fullres_ext_v2; outroot=outputs/agentic_rl_real
ck=$outroot/$name/checkpoints
if [ -f "$ck/final.pth" ]; then echo "=== SKIP $name (final.pth) ==="; exit 0; fi
resume=""; [ -f "$ck/latest.pth" ] && resume="--checkpoint $ck/latest.pth" && echo "=== RESUME $name ==="
conda run --no-capture-output -n fgs_cu128 python -u agentic_gs_phase1/scripts/train_agent.py \
  --config configs/real_fastergs_train_fullres_ext_v2.json --run-name $name \
  --output-root $outroot --max-updates 300 $resume
echo "=== EXIT $name = $? ==="
echo "=== ACCEL_EXT_V2_DONE ==="
