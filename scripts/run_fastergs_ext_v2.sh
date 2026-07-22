#!/usr/bin/env bash
# Retrain the FasterGS-ext BUDGET policy v2: AA action removed + backend-aware
# quality/compute reward term. 300 PPO updates, resumable.
cd /c/Roman/3DGS_PROPOSAL
name=ppo_budget_fastergs_fullres_ext_v2; outroot=outputs/agentic_rl_budget
ck=$outroot/$name/checkpoints
if [ -f "$ck/final.pth" ]; then echo "=== SKIP $name (final.pth) ==="; exit 0; fi
resume=""; [ -f "$ck/latest.pth" ] && resume="--checkpoint $ck/latest.pth" && echo "=== RESUME $name ==="
conda run --no-capture-output -n fgs_cu128 python -u agentic_gs_phase1/scripts/train_agent.py \
  --config configs/real_fastergs_budget_fullres_ext_v2.json --run-name $name \
  --output-root $outroot --max-updates 300 $resume
echo "=== EXIT $name = $? ==="
echo "=== V2_DONE ==="
