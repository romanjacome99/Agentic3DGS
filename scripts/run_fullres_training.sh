#!/usr/bin/env bash
# Sequentially train all 4 policies at FULL resolution (resolution=1). Runs are exclusive
# (GPU not shared) because the acceleration/budget rewards are wall-clock based. Resumable:
# re-running this skips finished runs (final.pth) and resumes in-progress ones (latest.pth).
cd /c/Roman/3DGS_PROPOSAL
run_one() {
  local env="$1" cfg="$2" name="$3" outroot="$4"
  local ckdir="$outroot/$name/checkpoints"
  if [ -f "$ckdir/final.pth" ]; then echo "=== SKIP $name (final.pth present) ==="; return 0; fi
  local resume=""
  if [ -f "$ckdir/latest.pth" ]; then resume="--checkpoint $ckdir/latest.pth"; echo "=== RESUME $name from latest.pth ==="; else echo "=== START $name (full-res) ==="; fi
  conda run --no-capture-output -n "$env" python -u agentic_gs_phase1/scripts/train_agent.py \
    --config "configs/$cfg" --run-name "$name" --output-root "$outroot" --max-updates 150 $resume
  echo "=== EXIT $name = $? ==="
}
run_one env_pytorch_3dgs real_3dgs_train_fullres.json      ppo_real_3dgs_fullres      outputs/agentic_rl_real
run_one fgs_cu128        real_fastergs_train_fullres.json  ppo_real_fastergs_fullres  outputs/agentic_rl_real
run_one env_pytorch_3dgs real_3dgs_budget_fullres.json     ppo_budget_3dgs_fullres    outputs/agentic_rl_budget
run_one fgs_cu128        real_fastergs_budget_fullres.json ppo_budget_fastergs_fullres outputs/agentic_rl_budget
echo "=== ALL_FULLRES_DONE ==="
