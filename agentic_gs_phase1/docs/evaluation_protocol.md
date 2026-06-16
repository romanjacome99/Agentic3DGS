# Evaluation Protocol: Agentic 3DGS Training Acceleration

This protocol tests whether the learned policy accelerates 3DGS training compared with the fixed 3DGS schedule.

The policy may use held-out training cameras for observations and rewards. It must never use dataset test cameras during policy training, state construction, or reward computation. Dataset test cameras are reserved for offline reporting and for **checkpoint selection on a probe scene** (see below).

## Checkpoint Selection (reward/eval alignment)

The train-camera reward is a proxy; over-optimizing it widens the train-cam/test-cam gap (observed: 150 PPO updates scored 0.3–1.5 dB lower test PSNR than 60 updates while reward and PPO-health metrics kept improving). Three mitigations are part of the protocol:

1. **Probe-based selection.** During PPO training, every `selection.eval_every_updates` updates a frozen deterministic episode runs on `selection.probe_scene` and is scored on that scene's *test* cameras. The best probe test PSNR selects `checkpoints/best.pth` (logged in `probe_eval.csv`). The probe scene must not be a policy-training scene, and should not be an evaluation scene either (`lego` for the MVP split), so `hotdog`/`drums` remain untouched for final reporting. Test cameras still never enter reward or observations.
2. **Coverage validation cameras.** The held-out reward cameras are chosen by farthest-point sampling over camera centers (`validation_selection: "coverage"`) instead of uniform random, so the reward proxy spans the capture hemisphere like the test set.
3. **Worst-view robust quality.** Reward quality is `mean + min_view_weight * (min - mean)` over per-view `psnr + w_ssim * ssim`, preventing the policy from trading away poorly covered views.

Report results from `best.pth` (probe-selected), not `final.pth`; `final.pth` remains for diagnostics.

## 1. Experimental Question

The proposed policy is considered an accelerator if, on evaluation scenes, it satisfies at least one of:

- reaches the same test PSNR as fixed 3DGS in less training wall-clock time,
- reaches higher test PSNR at the same wall-clock budget,
- reaches comparable test PSNR with fewer Gaussians, lower peak VRAM, or smaller model size,
- improves the PSNR-versus-time Pareto curve.

## 2. Scene Split

Use the split below unless explicitly running an ablation.

| Role | Scenes |
|---|---|
| Policy training | `chair`, `drums`, `mic` |
| Policy evaluation | `hotdog`, `drums` |

`hotdog` tests scene generalization. `drums` tests whether the learned controller still helps on a scene seen during policy training.

## 3. Compared Methods

Report at least these methods:

| Method | Description |
|---|---|
| Fixed 3DGS 7k | Stock training schedule at 7,000 iterations |
| Fixed 3DGS 30k | Stock reference-quality schedule at 30,000 iterations |
| Agentic policy 7k budget | Frozen PPO policy, same maximum raw-iteration budget as 7k baseline |
| Optional fixed 3DGS 1k/3k/5k | Short-budget curve points |

The policy must be frozen during evaluation. PPO updates, reward optimization, and checkpoint selection using test metrics are not allowed.

## 4. Wall-Clock Definition

Primary wall-clock time is **training wall-clock**:

```text
start: immediately before scene initialization/training begins
stop: immediately after the requested training checkpoint is saved
included: optimizer steps, rendering during optimization, densification, pruning, opacity reset, policy inference, validation-subset evaluation used by the policy, CUDA synchronization
excluded: offline test rendering, offline metric computation, plotting/report generation
```

Also report these secondary timings separately:

- render wall-clock,
- metric-computation wall-clock,
- total end-to-end wall-clock = training + render + metrics.

Baseline runs now write `baseline_command_timings.json`. Policy runs write per-block elapsed time in `agentic_blocks.csv`.

## 5. Quality Metrics

Primary quality metrics:

- test PSNR,
- test SSIM,
- test LPIPS when available.

Online validation-subset PSNR/SSIM may be used for debugging and learning curves, but final acceleration claims should use dataset test cameras only.

## 6. Compute and Efficiency Metrics

Report these for every method and scene:

- training wall-clock seconds,
- time per raw optimizer iteration,
- time per policy block for agentic runs,
- validation-subset overhead for agentic runs,
- peak VRAM,
- final Gaussian count,
- Gaussian growth rate,
- model size on disk,
- render FPS or render seconds per test view,
- final checkpoint iteration.

Derived acceleration metrics:

```text
time_to_target_psnr(method, scene, target)
speedup_x = baseline_time_to_target / policy_time_to_target
time_saved_seconds = baseline_time_to_target - policy_time_to_target
psnr_at_budget(method, scene, budget_seconds)
gaussian_efficiency = test_psnr / final_gaussian_count
memory_efficiency = test_psnr / peak_vram_gb
```

Recommended fixed wall-clock budgets:

```text
30 s, 60 s, 120 s, 300 s
```

For longer runs, also include:

```text
600 s, 1200 s
```

## 7. Required Tables

### 7.1 Final Quality and Compute

| Scene | Method | Iterations | Train seconds | Test PSNR | Test SSIM | LPIPS | Gaussians | Peak VRAM GB | Model MB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|

### 7.2 Fixed-Budget Quality

| Scene | Method | 30s PSNR | 60s PSNR | 120s PSNR | 300s PSNR | Peak VRAM GB | Gaussians |
|---|---:|---:|---:|---:|---:|---:|---:|

### 7.3 Time-to-Target

| Scene | Target PSNR | Baseline seconds | Baseline status | Policy seconds | Policy status | Speedup x | Time saved | Claim ready |
|---|---:|---:|---|---:|---|---:|---:|---|

Use this as the primary table when speed is the main claim. The strongest acceleration result is not "same iterations but slightly faster"; it is:

```text
agentic GS reaches a target test PSNR in less training wall-clock time than fixed 3DGS.
```

Statuses must be reported with the time values:

- `interpolated_between_samples`: crossing is linearly interpolated between two measured full-test PSNR samples.
- `upper_bound_first_sample`: first measured sample already exceeds the target, so the true crossing happened earlier or equal to that time.
- `not_reached`: the target was not reached by the sampled curve.

When repeated budget runs produce timing reversals, compute crossings on the PSNR-versus-time Pareto envelope. A sample is dominated if an earlier or equal wall-clock sample from the same method has equal or higher test PSNR; keep it in the raw curve, but do not use it to make time-to-target worse.

Only use the row as a final speed claim when both methods have observed or interpolated crossings, not only upper bounds.

### 7.4 Policy Behavior

| Scene | Block | Iteration | Action | Validation PSNR | Reward | Gaussians | Time/block |
|---|---:|---:|---|---:|---:|---:|---:|

This table is mainly for interpretability and debugging.

## 8. Minimum Repetition

Use at least three random seeds per method and scene for claims. Report mean and standard deviation.

Recommended seeds:

```text
7, 17, 27
```

For quick internal debugging, one seed is acceptable, but label results as preliminary.

## 9. Commands

### 9.1 Train the Policy

```powershell
conda activate env_pytorch_3dgs

python agentic_gs_phase1\scripts\train_agent.py `
  --config agentic_gs_phase1\configs\phase1_mvp.json `
  --run-name ppo_chair_drums_mic_7k
```

### 9.2 Evaluate the Frozen Policy

Save checkpoints when wall-clock budgets are crossed:

```powershell
python agentic_gs_phase1\scripts\eval_agent.py `
  --checkpoint outputs\agentic_gs_phase1\ppo_chair_drums_mic_7k\checkpoints\final.pth `
  --config agentic_gs_phase1\configs\phase1_eval.json `
  --eval-scenes hotdog drums `
  --run-name eval_hotdog_drums `
  --save-time-budgets 30 60 120 300
```

### 9.3 Run Fixed Baselines

Run baseline at the same iteration budget as the policy:

```powershell
python scripts\run_3dgs_baseline.py `
  --scene hotdog `
  --iterations 7000 `
  --model-path outputs\3dgs_baseline\hotdog\fixed_7000_eval `
  --python C:\Users\User\anaconda3\envs\env_pytorch_3dgs\python.exe
```

Run the 30k reference:

```powershell
python scripts\run_3dgs_baseline.py `
  --scene hotdog `
  --iterations 30000 `
  --model-path outputs\3dgs_baseline\hotdog\fixed_30000_eval `
  --python C:\Users\User\anaconda3\envs\env_pytorch_3dgs\python.exe
```

Repeat for `drums`.

### 9.4 Offline Test Rendering and Metrics

For any baseline or policy model folder:

```powershell
python gaussian-splatting\render.py -m <model_dir> --skip_train
python gaussian-splatting\metrics.py -m <model_dir>
```

For a specific saved iteration:

```powershell
python gaussian-splatting\render.py -m <model_dir> --iteration <iteration> --skip_train
python gaussian-splatting\metrics.py -m <model_dir>
```

Offline rendering/metrics are not counted in primary training wall-clock, but should be reported separately when available.

### 9.5 Aggregate Reports

```powershell
python agentic_gs_phase1\scripts\report_acceleration.py `
  --baseline-root outputs\3dgs_baseline `
  --policy-root outputs\agentic_gs_phase1_eval\eval_hotdog_drums `
  --time-budgets 30 60 120 300 `
  --target-psnrs 20 25 30 `
  --output-dir outputs\agentic_gs_phase1_reports\eval_hotdog_drums
```

Generated files:

```text
run_summary.csv
time_budget_curve.csv
time_to_target.csv
speedup_summary.csv
```

### 9.6 Hotdog Time-to-Target Protocol

This command uses the requested PPO checkpoint and compares fixed 3DGS against agentic 3DGS on `hotdog` at target test PSNR values `25, 27, 29, 31, 33, 35`.

To reuse existing sampled runs and write the report:

```powershell
python agentic_gs_phase1\scripts\time_to_target_protocol.py `
  --checkpoint outputs\agentic_gs_phase1\ppo_5scene_fast_15k_long\checkpoints\latest.pth `
  --config agentic_gs_phase1\configs\phase1_fast.json `
  --scene hotdog `
  --sample-iterations 100 250 500 750 1000 1500 2000 2500 3000 4000 5000 `
  --targets 25 27 29 31 33 35
```

To run missing sample budgets first, add `--run-missing`:

```powershell
python agentic_gs_phase1\scripts\time_to_target_protocol.py `
  --checkpoint outputs\agentic_gs_phase1\ppo_5scene_fast_15k_long\checkpoints\latest.pth `
  --config agentic_gs_phase1\configs\phase1_fast.json `
  --scene hotdog `
  --sample-iterations 100 250 500 750 1000 1500 2000 2500 3000 4000 5000 `
  --targets 25 27 29 31 33 35 `
  --run-missing
```

Generated files:

```text
outputs/agentic_gs_phase1_reports/hotdog_time_to_target_ppo5scene_fast_latest/sampled_test_curve.csv
outputs/agentic_gs_phase1_reports/hotdog_time_to_target_ppo5scene_fast_latest/time_to_target_by_method.csv
outputs/agentic_gs_phase1_reports/hotdog_time_to_target_ppo5scene_fast_latest/time_to_target_comparison.csv
outputs/agentic_gs_phase1_reports/hotdog_time_to_target_ppo5scene_fast_latest/hotdog_time_to_target_protocol.md
```

## 10. Fairness Rules

- Use the same GPU, CUDA/PyTorch environment, resolution, background setting, and dataset files.
- Use `resolution: 1` and `white_background: true` for NeRF synthetic scenes.
- Do not tune the policy checkpoint using test PSNR.
- Do not include test cameras in policy observations or rewards.
- Warm up the GPU once before timed production runs, or discard the first timed run.
- Close unrelated GPU processes before running timed comparisons.
- Report failed runs and numerical failures.
- Report whether wall-clock times are measured command wall time or estimated from logged iteration timings.

## 11. Claim Template

Use this language only if supported by the tables:

```text
At a fixed 300 s training budget, the proposed policy improves test PSNR by X dB over fixed 3DGS on <scene>, while using Y% fewer/more Gaussians and Z GB peak VRAM.
```

or:

```text
The proposed policy reaches <target> dB test PSNR in A seconds, compared with B seconds for fixed 3DGS, giving a B/A speedup.
```

Avoid claiming acceleration from validation-subset PSNR alone. Use validation curves to explain policy behavior, and test metrics for the main result.
