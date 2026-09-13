#!/usr/bin/env bash
# Training-view ablation: time-to-target protocol (agent vs fixed schedule) with a reduced number of
# training cameras, 3DGS backend, frozen controller. No training. Sequential (timings are the measurement).
#
#   bash scripts/run_ablation_views.sh [scenes...]      default: train ignatius caterpillar barn
#
# Per scene and view count N in $VIEWS, writes configs/ablation_views/<scene>_v<N>.json (base config +
# train_camera_limit) and runs scripts/eval_protocol.py into outputs/agentic_rl_real/ablation_views_3dgs/<scene>_v<N>/.
# The full-view reference is the existing protocol run of each scene (see website/tools/build_ablation_views.py).
cd /c/Roman/3DGS_PROPOSAL || exit 1
PY="conda run -n env_pytorch_3dgs --no-capture-output python"
CK=outputs/agentic_rl_real/final_accel_3dgs/checkpoints/selected_accel.pth
VIEWS="${VIEWS:-100 50 25}"
MAXIT="${MAXIT:-30000}"
TARGETS="16 17 18 19 20 21 22 23 24 25 26 27 28"
SCENES="${*:-train ignatius caterpillar barn}"
mkdir -p configs/ablation_views outputs/agentic_rl_real/ablation_views_3dgs
for S in $SCENES; do
  if [ "$S" = "train" ]; then BASE=configs/final_accel_3dgs.json; else BASE=configs/final_accel_3dgs_tandt.json; fi
  for N in $VIEWS; do
    CFG=configs/ablation_views/${S}_v${N}.json
    python - "$BASE" "$CFG" "$N" <<'EOF'
import json, sys
base, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = json.load(open(base)); c["train_camera_limit"] = n; c["train_camera_limit_seed"] = 1234
c["_ablation"] = f"training-view ablation: {n} training views (farthest-point subset), base {base}"
json.dump(c, open(out, "w"), indent=1)
EOF
    OUT=outputs/agentic_rl_real/ablation_views_3dgs/${S}_v${N}
    if [ -f "$OUT/summary.json" ]; then echo "=== skip $S v$N (done)"; continue; fi
    echo "=== $(date +%T) $S views=$N"
    $PY scripts/eval_protocol.py --config "$CFG" --checkpoint $CK --scene "$S" --max-iter $MAXIT --views 12 \
        --targets $TARGETS --methods baseline agentic --out "$OUT"
    echo "=== exit $? $(date +%T)"
  done
done
echo "=== ABLATION DONE $(date +%T)"
