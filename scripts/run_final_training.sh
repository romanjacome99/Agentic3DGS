#!/usr/bin/env bash
# ==============================================================================
# CANONICAL FINAL TRAINING RUNNER  (uniform 8-scene pool, 300 PPO updates)
# ------------------------------------------------------------------------------
# Trains all four final policies sequentially (wall-clock rewards need exclusive
# GPU). Resumable: skips a run whose final.pth exists, resumes from latest.pth.
# Each backend runs in its own conda env.
#
#   accel_3dgs      -> outputs/agentic_rl_real/final_accel_3dgs
#   accel_fastergs  -> outputs/agentic_rl_real/final_accel_fastergs
#   accel_dash      -> outputs/agentic_rl_real/final_accel_dash      (DashGaussian)
#   budget_3dgs     -> outputs/agentic_rl_budget/final_budget_3dgs
#   budget_fastergs -> outputs/agentic_rl_budget/final_budget_fastergs
#   budget_dash     -> outputs/agentic_rl_budget/final_budget_dash   (DashGaussian)
#
# Usage:  bash scripts/run_final_training.sh                       # the original four
#         bash scripts/run_final_training.sh accel_dash budget_dash # the DashGaussian pair
#         bash scripts/run_final_training.sh accel_3dgs            # a single run
# ==============================================================================
set -u
cd /c/Roman/3DGS_PROPOSAL
UPDATES=300
TRAIN=agentic_gs_phase1/scripts/train_agent.py

run_one() {
  local key="$1" env="$2" cfg="$3" outroot="$4"
  local name="final_$key"
  local ckdir="$outroot/$name/checkpoints"
  if [ -f "$ckdir/final.pth" ]; then echo "=== SKIP $name (final.pth present) ==="; return 0; fi
  local resume=""
  if [ -f "$ckdir/latest.pth" ]; then resume="--checkpoint $ckdir/latest.pth"; echo "=== RESUME $name ==="; else echo "=== START $name ($UPDATES updates, env=$env) ==="; fi
  conda run --no-capture-output -n "$env" python -u "$TRAIN" \
    --config "configs/$cfg" --run-name "$name" --output-root "$outroot" \
    --max-updates $UPDATES $resume
  echo "=== EXIT $name = $? ==="
}

# key             env                cfg                          output_root
# (DashGaussian is the pragmatic 3dgs-based backend -> env_pytorch_3dgs, no CUDA build.)
declare -A ENVV=( [accel_3dgs]=env_pytorch_3dgs [budget_3dgs]=env_pytorch_3dgs \
                  [accel_fastergs]=fgs_cu128    [budget_fastergs]=fgs_cu128 \
                  [accel_dash]=env_pytorch_3dgs [budget_dash]=env_pytorch_3dgs )
declare -A CFGV=( [accel_3dgs]=final_accel_3dgs.json   [budget_3dgs]=final_budget_3dgs.json \
                  [accel_fastergs]=final_accel_fastergs.json [budget_fastergs]=final_budget_fastergs.json \
                  [accel_dash]=final_accel_dash.json  [budget_dash]=final_budget_dash.json )
declare -A OUTV=( [accel_3dgs]=outputs/agentic_rl_real   [budget_3dgs]=outputs/agentic_rl_budget \
                  [accel_fastergs]=outputs/agentic_rl_real [budget_fastergs]=outputs/agentic_rl_budget \
                  [accel_dash]=outputs/agentic_rl_real  [budget_dash]=outputs/agentic_rl_budget )

# Order: 3DGS pair first (env_pytorch_3dgs), then FasterGS pair (fgs_cu128).
if [ $# -gt 0 ]; then KEYS=("$@"); else KEYS=(accel_3dgs budget_3dgs accel_fastergs budget_fastergs); fi
for key in "${KEYS[@]}"; do
  run_one "$key" "${ENVV[$key]}" "${CFGV[$key]}" "${OUTV[$key]}"
done
echo "=== ALL_FINAL_TRAINING_DONE ==="
