# Agentic GS Phase 1

This folder implements the first MVP for block-wise RL control of static 3DGS training.

The implementation keeps the working fixed baseline path intact and adds a separate PPO-controlled training loop. The policy acts once per block and controls:

- block length,
- densification mode and threshold multiplier,
- densification interval,
- pruning threshold and pruning mode,
- opacity reset trigger,
- learning-rate multipliers for position, feature/color, opacity, scaling, and rotation groups,
- early stop.

Rewards and observations use a held-out subset of training cameras. Dataset test cameras are not used for state or reward construction.

To keep the train-camera reward aligned with the test-camera objective:

- validation cameras are picked by farthest-point sampling over camera centers (`validation_selection: "coverage"`),
- reward quality blends mean and worst-view quality (`reward.min_view_weight`),
- every `selection.eval_every_updates` PPO updates a frozen probe episode runs on the held-out `selection.probe_scene` (not a train or eval scene) and is scored on its test cameras; the best score selects `checkpoints/best.pth` (see `probe_eval.csv`). Evaluate `best.pth`, not `final.pth`.

## Train

```powershell
conda activate env_pytorch_3dgs
python agentic_gs_phase1\scripts\train_agent.py --config agentic_gs_phase1\configs\phase1_mvp.json
```

Default policy training scenes:

```text
chair, drums, mic
```

## Evaluate

```powershell
python agentic_gs_phase1\scripts\eval_agent.py `
  --checkpoint outputs\agentic_gs_phase1\<run>\checkpoints\final.pth `
  --config agentic_gs_phase1\configs\phase1_eval.json
```

Default evaluation scenes:

```text
hotdog, drums
```

## Report Acceleration

The protocol is documented in `docs/evaluation_protocol.md`. After running fixed baselines and policy evaluation, aggregate wall-clock and quality tables with:

```powershell
python agentic_gs_phase1\scripts\report_acceleration.py `
  --baseline-root outputs\3dgs_baseline `
  --policy-root outputs\agentic_gs_phase1_eval\<eval-run> `
  --time-budgets 30 60 120 300 `
  --target-psnrs 20 25 30 `
  --output-dir outputs\agentic_gs_phase1_reports\<eval-run>
```

Plot wall-clock curves and policy decisions:

```powershell
python agentic_gs_phase1\scripts\plot_acceleration.py `
  --baseline-root outputs\3dgs_baseline `
  --policy-root outputs\agentic_gs_phase1_eval\<eval-run> `
  --output-dir outputs\agentic_gs_phase1_plots\<eval-run>
```

Run fixed 3DGS and the frozen agent through the same evaluation code and the same camera subset:

```powershell
python agentic_gs_phase1\scripts\run_unified_eval.py `
  --config agentic_gs_phase1\configs\phase1_eval.json `
  --checkpoint outputs\agentic_gs_phase1\<run>\checkpoints\final.pth `
  --scenes hotdog drums `
  --methods baseline agentic `
  --eval-subsets validation test `
  --run-name unified_hotdog_drums_7k
```

This writes `shared_split.json`, reconstructed views, `eval_metrics_<subset>.csv`, `unified_eval_summary.json`, and `action_sets.csv/json` for each method.

## Fast Debug Run

Use this to verify plumbing before spending time on RL:

```powershell
python agentic_gs_phase1\scripts\train_agent.py `
  --config agentic_gs_phase1\configs\phase1_mvp.json `
  --run-name smoke `
  --max-episode-iterations 200 `
  --rollout-steps 2 `
  --max-updates 1 `
  --train-scenes chair
```

Outputs are written under `outputs/agentic_gs_phase1/<run-name>/`. Each episode has `agentic_blocks.csv` and `agentic_episode_metadata.json`.
