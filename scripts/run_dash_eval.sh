#!/usr/bin/env bash
# CANONICAL DashGaussian EVAL (mirrors run_final_eval.sh, dash-only, env_pytorch_3dgs).
# Uses the re-selected checkpoints (selected_accel.pth / selected_budget.pth). Dash's
# own fixed-schedule DashGaussian trainer is the baseline (default action on the dash
# backend, which still applies the resolution + primitive-budget schedule).
#
# Stages: bash scripts/run_dash_eval.sh [A B C D E]   (default all)
set -u
cd /c/Roman/3DGS_PROPOSAL
E=env_pytorch_3dgs
CK_A=outputs/agentic_rl_real/final_accel_dash/checkpoints/selected_accel.pth
CK_B=outputs/agentic_rl_budget/final_budget_dash/checkpoints/selected_budget.pth
run(){ echo ">>> $*"; conda run --no-capture-output -n $E python "$@"; echo "   exit=$?"; }
STAGES="${*:-A B C D E}"; has(){ case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# A. accel time-to-target on held-out train -> tab:accel(dash) + AUC
if has A; then
  run scripts/eval_protocol.py --config configs/final_accel_dash.json --checkpoint $CK_A \
      --scene train --max-iter 30000 --targets 16 17 18 19 20 21 \
      --out outputs/agentic_rl_real/protocol_dash_train_30k
fi
# B. generalization: stump (targets 16-24) + bicycle (20-25)
if has B; then
  run scripts/eval_protocol.py --config configs/final_accel_dash_stump.json --checkpoint $CK_A \
      --scene stump --max-iter 30000 --targets 16 17 18 19 20 21 22 23 24 \
      --out outputs/agentic_rl_real/protocol_dash_stump_30k
  run scripts/eval_protocol.py --config configs/final_accel_dash_bicycle.json --checkpoint $CK_A \
      --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 \
      --out outputs/agentic_rl_real/protocol_dash_bicycle_30k
fi
# C. deployment metrics (agent + own-backend baseline) -> tab:efficiency(dash)
if has C; then
  run scripts/measure_metrics.py --config configs/final_accel_dash.json --method agentic  --checkpoint $CK_A --scene train --max-iter 30000 --out outputs/metrics/dash_agent
  run scripts/measure_metrics.py --config configs/final_accel_dash.json --method baseline                   --scene train --max-iter 30000 --out outputs/metrics/dash_baseline
fi
# D. budget sweep AGENT, 10 seeds -> tab:scaling(dash)
SWEEP="--budgets 30 60 120 300 360 --unlimited --seeds 0 1 2 3 4 5 6 7 8 9 --scene train --views 12"
if has D; then
  run scripts/budget_sweep_multi.py --config configs/final_budget_dash.json --method agentic  --checkpoint $CK_B $SWEEP --out outputs/agentic_rl_budget/multiseed10/dash_agent
fi
# E. budget sweep BASELINE (dash fixed schedule), 10 seeds -> tab:vsbaseline(dash)
if has E; then
  run scripts/budget_sweep_multi.py --config configs/final_budget_dash.json --method baseline $SWEEP --out outputs/agentic_rl_budget/multiseed10_baseline/dash_baseline
fi
echo "=== DASH_EVAL_DONE ($STAGES) ==="
