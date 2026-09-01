#!/usr/bin/env bash
# Budget checkpoint re-selection (10-scene pool). The accel_auc probe under-selects
# on quality; re-evaluate high-quality candidates on a light budget sweep
# (30/120/300 s x 3 seeds) and pick the best quality-at-budget. Sequential (GPU).
# Baseline (fixed schedule on held-out train) is pool-independent -> reuse existing.
set -u
cd /c/Roman/3DGS_PROPOSAL
LIGHT="--budgets 30 120 300 --seeds 0 1 2 --scene train --views 12"
sweep() { # env backend update
  local env="$1" bk="$2" u="$3"
  local ck=outputs/agentic_rl_budget/final_budget_${bk}/checkpoints/policy_update_${u}.pth
  local out=outputs/agentic_rl_budget/reselect10/${bk}_u${u}
  [ -f "$out/summary.csv" ] && { echo "SKIP ${bk}_u${u}"; return 0; }
  echo ">>> reselect10 ${bk}_u${u}"
  conda run --no-capture-output -n "$env" python scripts/budget_sweep_multi.py \
    --config configs/final_budget_${bk}.json --method agentic --checkpoint "$ck" $LIGHT --out "$out"
  echo "    exit=$?"
}
for u in 0039 0049 0279; do sweep env_pytorch_3dgs 3dgs $u; done
for u in 0079 0089 0129 0049; do sweep fgs_cu128 fastergs $u; done
echo "=== RESELECT10_DONE ==="
