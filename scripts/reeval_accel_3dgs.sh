#!/usr/bin/env bash
# Targeted re-eval of the 3DGS ACCELERATION side after adopting the compaction-aware
# checkpoint u289 (selected_accel.pth). FasterGS accel unchanged (u99 = best.pth), so
# only the 3dgs accel artifacts are regenerated + the shared figures.
set -u
cd /c/Roman/3DGS_PROPOSAL
E3=env_pytorch_3dgs
CK=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
run(){ echo ">>> $*"; conda run --no-capture-output -n $E3 python "$@"; echo "   exit=$?"; }

# 1. time-to-target on held-out train (tab:accel, AUC, action_sets top-left)
run scripts/eval_protocol.py --config configs/final_accel_3dgs.json --checkpoint $CK \
    --scene train --max-iter 30000 --targets 16 17 18 19 20 21 \
    --out outputs/agentic_rl_real/protocol_3dgs_train_30k
# 2. bicycle generalization (tab:generalization)
run scripts/eval_protocol.py --config configs/final_accel_3dgs_bicycle.json --checkpoint $CK \
    --scene bicycle --max-iter 30000 --targets 20 21 22 23 24 25 \
    --out outputs/agentic_rl_real/protocol_3dgs_bicycle_30k
# 3. deployment metrics, agent only (tab:efficiency); baseline unchanged
run scripts/measure_metrics.py --config configs/final_accel_3dgs.json --method agentic \
    --checkpoint $CK --scene train --max-iter 30000 --out outputs/metrics/3dgs_agent
# 4. AUC (reads both backends' curves)
run scripts/compute_auc.py
# 5. gaussian snapshots, agent only (baseline unchanged); then render + fig
run scripts/save_gaussian_snapshots.py --config configs/final_accel_3dgs.json --mode acceleration \
    --method agentic --checkpoint $CK --scene train --horizon 120 --snap-times 5 15 30 60 120 \
    --out outputs/gaussian_evolution/3dgs_accel_agent
run scripts/render_snapshots.py --config configs/final_accel_3dgs.json --scene train \
    --dirs outputs/gaussian_evolution/3dgs_accel_agent
# 6. regenerate the two accel-dependent figures
run scripts/make_action_figs.py
run scripts/make_evolution_paper_fig.py
echo "=== REEVAL_ACCEL_3DGS_DONE ==="
