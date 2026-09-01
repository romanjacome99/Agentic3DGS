#!/usr/bin/env bash
# DashGaussian checkpoint re-selection (mirrors the 3dgs/fastergs reselection).
# accel: run candidates to 30k (agentic-only) on held-out train -> pick fast+compact.
# budget: light sweep (30/120/300 s x 3 seeds) over candidates -> pick quality-at-budget.
# All on env_pytorch_3dgs (dash = pragmatic 3dgs-based backend).
set -u
cd /c/Roman/3DGS_PROPOSAL
E=env_pytorch_3dgs

# --- acceleration candidates -> 30k on train --------------------------------
accel() {
  local u="$1"
  local ck=outputs/agentic_rl_real/final_accel_dash/checkpoints/policy_update_${u}.pth
  local out=outputs/_accel_reselect/dash_u${u}
  [ -f "$out/curve.csv" ] && { echo "SKIP accel u${u}"; return 0; }
  echo ">>> accel_reselect dash u${u}"
  conda run --no-capture-output -n $E python scripts/eval_protocol.py \
    --config configs/final_accel_dash.json --checkpoint "$ck" --scene train \
    --max-iter 30000 --methods agentic --targets 16 17 18 19 20 21 --out "$out"
  echo "    exit=$?"
}
for u in 0289 0009 0019; do accel $u; done

# --- budget candidates -> light sweep ---------------------------------------
budget() {
  local u="$1"
  local ck=outputs/agentic_rl_budget/final_budget_dash/checkpoints/policy_update_${u}.pth
  local out=outputs/agentic_rl_budget/reselect_dash/u${u}
  [ -f "$out/summary.csv" ] && { echo "SKIP budget u${u}"; return 0; }
  echo ">>> budget_reselect dash u${u}"
  conda run --no-capture-output -n $E python scripts/budget_sweep_multi.py \
    --config configs/final_budget_dash.json --method agentic --checkpoint "$ck" \
    --budgets 30 120 300 --seeds 0 1 2 --scene train --views 12 --out "$out"
  echo "    exit=$?"
}
for u in 0019 0069 0209; do budget $u; done
echo "=== RESELECT_DASH_DONE ==="
