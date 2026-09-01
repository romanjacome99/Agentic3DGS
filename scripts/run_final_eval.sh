#!/usr/bin/env bash
# ==============================================================================
# CANONICAL FINAL EVALUATION RUNNER — reproduces every paper table & figure
# from the four final checkpoints trained by scripts/run_final_training.sh.
# ------------------------------------------------------------------------------
# Output dir names are kept IDENTICAL to what the figure/table generators expect
# (protocol_*_train_30k, multiseed10/*, outputs/metrics/*, gaussian_evolution/*)
# so make_*.py run unchanged. Each backend runs in its own conda env.
#
# Stages (run top-to-bottom; each stage skips if its output already exists):
#   A. time-to-target on held-out `train`   -> tab:accel, AUC, fig:action_sets(top)
#   B. time-to-target on held-out `bicycle` -> tab:generalization
#   C. deployment metrics (agent+baseline)   -> tab:efficiency
#   D. budget sweep agent (10 seeds)         -> tab:scaling, fig:action_sets(bottom)
#   E. budget sweep baseline (10 seeds)      -> tab:vsbaseline, fig:budget_vs_baseline
#   F. gaussian snapshots + renders          -> fig:gaussian_evolution
#   G. regenerate all figures + budget_vs_baseline.tex
#
# Usage: bash scripts/run_final_eval.sh [A B C D E F G]   (default: all)
# ==============================================================================
set -u
cd /c/Roman/3DGS_PROPOSAL
E3=env_pytorch_3dgs
EF=fgs_cu128
# Acceleration: compaction-aware selected checkpoints (3dgs u289 / fastergs u99),
# chosen by @30k quality+compactness, NOT the accel_auc probe's best.pth (which
# picked a checkpoint that balloons to 1.54M gauss by 30k). See PROGRESS sec 4b.
CK_A3=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
CK_AF=outputs/agentic_rl_real/final_accel_fastergs/checkpoints/selected_accel.pth
# Budget: quality-selected checkpoints (u39 / u219) chosen by quality-at-budget
# re-selection, NOT the accel_auc probe's best.pth (which over-compacts). See
# REPRODUCIBILITY_PROGRESS.txt sec 3b. selected_budget.pth = policy_update_00{39,219}.pth.
CK_B3=outputs/agentic_rl_budget/final_budget_3dgs/checkpoints/selected_budget.pth
CK_BF=outputs/agentic_rl_budget/final_budget_fastergs/checkpoints/selected_budget.pth
run() { echo ">>> $*"; conda run --no-capture-output -n "$1" python "${@:2}"; echo "    exit=$?"; }

STAGES="${*:-A B C D E F G}"
has() { case " $STAGES " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- A. Time-to-target on held-out `train` (30k), both backends -------------- #
if has A; then
  run $E3 scripts/eval_protocol.py --config configs/final_accel_3dgs.json     --checkpoint $CK_A3 \
      --scene train --max-iter 30000 --targets 16 17 18 19 20 21 \
      --out outputs/agentic_rl_real/protocol_3dgs_train_30k
  run $EF scripts/eval_protocol.py --config configs/final_accel_fastergs.json --checkpoint $CK_AF \
      --scene train --max-iter 30000 --targets 16 17 18 19 20 21 \
      --out outputs/agentic_rl_real/protocol_fastergs_train_30k
  run $E3 scripts/compute_auc.py   # reads both protocol_*_train_30k/curve.csv -> outputs/metrics/auc.json
fi

# --- B. Time-to-target on held-out `bicycle` (30k), both backends ----------- #
# NOTE: needs a bicycle-pointed config carrying the SAME backend + policy flags
# as the trained checkpoint (esp. fastergs_policy_ext for FasterGS). Generated
# in the eval phase as configs/final_accel_{3dgs,fastergs}_bicycle.json.
if has B; then
  run $E3 scripts/eval_protocol.py --config configs/final_accel_3dgs_bicycle.json     --checkpoint $CK_A3 \
      --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 \
      --out outputs/agentic_rl_real/protocol_3dgs_bicycle_30k
  run $EF scripts/eval_protocol.py --config configs/final_accel_fastergs_bicycle.json --checkpoint $CK_AF \
      --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 \
      --out outputs/agentic_rl_real/protocol_fastergs_bicycle_30k
fi

# --- C. Deployment efficiency metrics (agent + baseline), both backends ----- #
if has C; then
  run $E3 scripts/measure_metrics.py --config configs/final_accel_3dgs.json     --method agentic  --checkpoint $CK_A3 --scene train --max-iter 30000 --out outputs/metrics/3dgs_agent
  run $E3 scripts/measure_metrics.py --config configs/final_accel_3dgs.json     --method baseline                     --scene train --max-iter 30000 --out outputs/metrics/3dgs_baseline
  run $EF scripts/measure_metrics.py --config configs/final_accel_fastergs.json --method agentic  --checkpoint $CK_AF --scene train --max-iter 30000 --out outputs/metrics/fastergs_agent
  run $EF scripts/measure_metrics.py --config configs/final_accel_fastergs.json --method baseline                     --scene train --max-iter 30000 --out outputs/metrics/fastergs_baseline
fi

# --- D. Budget sweep, AGENT, 10 seeds, both backends ------------------------ #
SWEEP="--budgets 30 60 120 300 360 --unlimited --seeds 0 1 2 3 4 5 6 7 8 9 --scene train --views 12"
if has D; then
  run $E3 scripts/budget_sweep_multi.py --config configs/final_budget_3dgs.json     --method agentic --checkpoint $CK_B3 $SWEEP --out outputs/agentic_rl_budget/multiseed10/3dgs_agent
  run $EF scripts/budget_sweep_multi.py --config configs/final_budget_fastergs.json --method agentic --checkpoint $CK_BF $SWEEP --out outputs/agentic_rl_budget/multiseed10/fastergs_agent
fi

# --- E. Budget sweep, BASELINE (fixed schedule), 10 seeds, both backends ---- #
if has E; then
  run $E3 scripts/budget_sweep_multi.py --config configs/final_budget_3dgs.json     --method baseline $SWEEP --out outputs/agentic_rl_budget/multiseed10_baseline/3dgs_baseline
  run $EF scripts/budget_sweep_multi.py --config configs/final_budget_fastergs.json --method baseline $SWEEP --out outputs/agentic_rl_budget/multiseed10_baseline/fastergs_baseline
fi

# --- F. Gaussian snapshots + renders (3DGS acceleration case) --------------- #
if has F; then
  SNAP="--scene train --horizon 120 --snap-times 5 15 30 60 120"
  run $E3 scripts/save_gaussian_snapshots.py --config configs/final_accel_3dgs.json --mode acceleration \
      --method agentic  --checkpoint $CK_A3 $SNAP --out outputs/gaussian_evolution/3dgs_accel_agent
  run $E3 scripts/save_gaussian_snapshots.py --config configs/final_accel_3dgs.json --mode acceleration \
      --method baseline                       $SNAP --out outputs/gaussian_evolution/3dgs_accel_baseline
  run $E3 scripts/render_snapshots.py --config configs/final_accel_3dgs.json --scene train \
      --dirs outputs/gaussian_evolution/3dgs_accel_agent outputs/gaussian_evolution/3dgs_accel_baseline
fi

# --- G. Regenerate all figures + the vs-baseline table ---------------------- #
if has G; then
  run $E3 scripts/make_action_figs.py
  run $E3 scripts/make_budget_compare.py     # -> fig:budget_vs_baseline + tables/budget_vs_baseline.tex
  run $E3 scripts/make_paper_figs.py         # -> fig:densify_vs_budget
  run $E3 scripts/make_evolution_paper_fig.py
fi
echo "=== FINAL_EVAL_DONE ($STAGES) ==="
