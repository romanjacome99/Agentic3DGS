# Phase 1 RL Specification: Agentic 3D Gaussian Splatting Training

**Project:** Agentic Gaussian Splatting for fast replay-oriented scene encoding  
**Phase:** Phase 1 — static 3D Gaussian Splatting control  
**Goal:** Train an offline RL agent that controls the 3DGS training schedule to improve reconstruction quality per unit compute under fixed wall-clock, memory, and Gaussian-count budgets.

---

## 1. Phase 1 Scope

Phase 1 should focus on **static 3D Gaussian Splatting only**. The agent should not yet modify the representation to 2DGS, 4DGS, FreeTimeGS, or Temporal Gaussian Hierarchy. Those belong to later phases. The purpose of Phase 1 is to prove that a learned policy can outperform a fixed 3DGS training schedule by adapting the optimization process to the scene, camera layout, and current fitting state.

The target is a learned controller that decides, during training, how to allocate compute among:

- optimization steps,
- densification,
- pruning,
- opacity reset,
- learning-rate scaling,
- camera/view sampling,
- image-resolution scheduling,
- SH/color complexity scheduling,
- early stopping.

The agent should be trained **offline**. At deployment or evaluation time, the policy is fixed and simply controls the GS optimizer. No PPO update, reward computation, or ground-truth-dependent learning should happen at deployment.

---

## 2. Motivation and Design Principle

The Netflix call emphasizes practical GS streaming systems, especially for 3D spatial plus 1D temporal replay, where training time, bitrate, and memory footprint are major bottlenecks. Phase 1 addresses the first bottleneck directly: reducing training/encoding time without significantly compromising reconstruction quality. It also starts preparing for memory and bitrate constraints by logging and penalizing unnecessary Gaussian growth.

The design should mirror the Agentic Phase Retrieval idea: the agent is **not** an unconstrained algorithm generator. It is a closed-loop controller over a predefined, interpretable action library. In APR, the policy observes solver statistics and chooses among update templates, fidelity terms, priors, and continuous hyperparameters. In Phase 1 GS, the policy observes GS training statistics and chooses among training-schedule operations, densification/pruning controls, camera sampling modes, and optimizer hyperparameters.

The central hypothesis is:

> Different scenes, camera distributions, and optimization stages need different GS training decisions. A learned policy can use the current fitting state to spend compute only where it improves validation quality efficiently.

---

## 3. System Requirements

### 3.1 3DGS Environment Wrapper

Create an environment wrapper around the 3DGS training loop with an interface similar to:

```python
obs = env.reset(scene_id, camera_split, budget_config)
obs, reward, done, info = env.step(action)
```

Each `env.step(action)` should run a short block of low-level 3DGS optimization iterations. The action remains active for the entire block.

The wrapper must expose the following controls without restarting training:

- learning-rate multipliers for each optimizer group,
- densification enable/disable and threshold scaling,
- densification interval or trigger timing,
- pruning threshold and pruning mode,
- opacity reset trigger,
- camera sampling mode,
- resolution scale,
- DSSIM/L1 loss weighting,
- SH degree progression control,
- maximum Gaussian budget or soft Gaussian-count penalty,
- stop/continue decision.

### 3.2 Decision Granularity

Do not make one RL decision per raw 3DGS iteration. This would be too expensive and noisy. Instead, define one RL step as a **training block**.

Recommended initial block sizes:

- 25 iterations for debugging,
- 50 iterations for small scenes,
- 100 iterations for default Phase 1 training,
- 200 iterations for low-overhead evaluation.

The action can include the block size, but the first prototype may fix the block size to 100 iterations.

### 3.3 Training Budgets

Use both iteration and wall-clock budgets. The reward should prioritize wall-clock time because the application motivation is encoding latency.

Recommended Phase 1 budgets:

- debug: 1,000 to 2,000 raw 3DGS iterations,
- fast budget: 3,000 to 5,000 iterations,
- medium budget: 7,000 iterations,
- reference budget: 30,000 iterations fixed-schedule 3DGS.

For proposal-aligned reporting, always include quality at fixed wall-clock budgets such as 30 s, 60 s, 120 s, and 300 s, even if the internal training loop is iteration-based.

### 3.4 Hard Safety Constraints

The environment should enforce hard constraints regardless of the policy output:

- maximum number of Gaussians,
- maximum VRAM usage,
- valid learning-rate multiplier ranges,
- valid densification/pruning thresholds,
- no densification before enough gradient statistics are collected,
- rollback or terminate on NaN/Inf loss,
- rollback or clamp if a policy action causes sudden catastrophic validation drop,
- maximum number of opacity resets per episode,
- no test-camera access during training.

---

## 4. Observation Space

The policy should observe compact solver statistics, not raw images. Raw images are expensive and would encourage scene memorization. The state should summarize the current optimization condition in a way that transfers across scenes and camera layouts.

### 4.1 Required Scalar Features

| Feature group | Features |
|---|---|
| Progress | normalized iteration, elapsed wall-clock fraction, remaining budget fraction, current block index |
| Training loss | EMA L1 loss, EMA DSSIM loss, total loss, loss slope over recent blocks |
| Validation quality | held-out PSNR, SSIM, LPIPS if available, improvement over previous block |
| Compute | time per iteration, time per rendered view, peak VRAM, allocated VRAM, active resolution scale |
| Gaussian count | total Gaussians, Gaussian growth rate, percentage of budget used |
| Previous action | previous discrete choices, previous continuous controls, whether previous action improved validation |

### 4.2 Gaussian Population Features

Compute summary statistics over the current Gaussian set:

- opacity quantiles: min, 10%, 25%, median, 75%, 90%, max,
- percentage of near-transparent Gaussians,
- scale quantiles,
- anisotropy quantiles,
- screen-space radius quantiles,
- number of visible Gaussians per sampled camera,
- number of Gaussians pruned in the previous block,
- number of Gaussians added in the previous block,
- average SH/color parameter magnitude,
- SH degree currently active.

### 4.3 Densification and Gradient Features

Track the statistics already used by 3DGS densification:

- view-space gradient magnitude quantiles,
- percentage of Gaussians above densification threshold,
- average accumulated gradient per visible Gaussian,
- number of candidate Gaussians for split/clone,
- max radii statistics,
- visibility-filter ratio,
- residual concentration across cameras.

### 4.4 Camera and Sensing Features

To support generalization across camera positions, include camera-layout descriptors:

- number of training cameras,
- number of validation cameras,
- average camera distance to scene center,
- camera distance variance,
- angular coverage statistics,
- camera baseline statistics,
- per-camera recent loss or PSNR,
- entropy of camera sampling over the previous blocks,
- percentage of cameras not sampled recently,
- train/validation camera distribution mismatch descriptor.

These features are important because the policy should generalize to different sensing patterns, not memorize a fixed camera rig.

---

## 5. Reward Design

The reward should measure **quality improvement per compute**, with penalties for unstable or inefficient training.

### 5.1 Default Reward Components

Use a block-level reward after each RL step:

- positive reward for validation PSNR/SSIM improvement,
- optional negative reward for LPIPS increase,
- penalty for wall-clock time spent in the block,
- penalty for excessive Gaussian growth,
- penalty for peak VRAM above target,
- penalty for validation quality drop,
- small terminal reward for final validation quality,
- optional bonus for reaching a target quality early.

### 5.2 Reward Computation Rules

- Compute validation quality on held-out cameras only.
- Never use test cameras for reward or state construction.
- Use a small validation subset during RL training to reduce overhead.
- Use full validation evaluation only at episode end or at sparse checkpoints.
- Normalize rewards per scene to avoid policies overfitting to scenes with naturally high PSNR.
- Log all reward terms separately for debugging.

### 5.3 Suggested Initial Reward Terms

Start simple:

```text
reward = validation_quality_gain
         - time_penalty
         - gaussian_growth_penalty
         - vram_penalty
         - instability_penalty
         + terminal_quality_bonus
```

Then add more terms only if the policy learns undesirable behavior, such as excessive Gaussian growth or frequent opacity resets.

---

## 6. RL Algorithm Requirements

### 6.1 Algorithm

Use PPO first, following the APR design. PPO is appropriate because the action space is mixed discrete-continuous and the environment is expensive, stochastic, and non-differentiable with respect to schedule decisions.

Required components:

- actor-critic network,
- shared state encoder,
- categorical heads for discrete actions,
- bounded continuous heads for threshold and learning-rate controls,
- value head,
- entropy regularization,
- GAE advantage estimation,
- KL early stopping,
- action clipping,
- rollout buffer storing observations, actions, rewards, log-probabilities, values, and info metrics.

### 6.2 Policy Architecture

Recommended initial architecture:

- input: normalized observation vector,
- MLP encoder: 2 to 4 layers, width 128 to 512,
- activation: GELU, SiLU, or ReLU,
- categorical heads for action groups,
- Beta or squashed Gaussian heads for continuous controls,
- value head with scalar output.

The policy should factorize the action into interpretable groups. This makes action traces easy to analyze and reduces the difficulty of training.

### 6.3 Offline Training Protocol

Train the policy over many GS fitting episodes:

1. Sample a scene.
2. Sample a camera split and camera perturbation configuration.
3. Initialize 3DGS from the standard sparse point cloud or a controlled degraded initialization.
4. Run the policy-controlled training loop until budget or stop action.
5. Compute rewards using held-out validation views.
6. Update the policy with PPO after collecting a rollout batch.

Optional warm start:

- collect trajectories from the fixed 3DGS schedule,
- train the policy by behavior cloning to imitate safe default behavior,
- then fine-tune with PPO to improve compute allocation.

---

## 7. Generalization Requirements

The policy should be trained to generalize across both **data distribution** and **sensing distribution**.

### 7.1 Scene Distribution Randomization

Train across diverse static scenes:

- indoor scenes,
- outdoor scenes,
- object-centric scenes,
- forward-facing scenes,
- 360-degree scenes,
- different texture densities,
- different lighting conditions,
- different scene scales.

### 7.2 Sensing and Camera Randomization

At each episode, randomize the sensing setup:

- number of cameras,
- camera density,
- camera baselines,
- held-out replay trajectory,
- camera dropout,
- camera pose noise,
- focal length variation,
- image resolution,
- train/validation split,
- sparse versus dense view coverage.

The policy should observe camera-layout statistics, so it can adapt when the camera positions change.

### 7.3 Initialization Randomization

Randomize the starting point cloud quality:

- full SfM initialization,
- downsampled SfM points,
- noisy point positions,
- reduced point cloud density,
- missing sparse points in low-texture regions.

This helps the agent learn when aggressive densification is useful and when it creates redundant Gaussians.

---

## 8. Phase 1 Action Space

The action space should be implemented in two stages: a minimal MVP action space and an extended Phase 1 action space.

---

## 8.1 MVP Action Space

Implement this first. It is small enough to debug but already tests the core idea.

| Action group | Type | Choices or range | Purpose |
|---|---:|---|---|
| `block_steps` | discrete | 50, 100, 200 | Number of raw 3DGS iterations before the next policy decision. |
| `camera_sampler` | discrete | uniform, loss_weighted, coverage_balanced | Selects which training views are sampled during the block. |
| `densify_mode` | discrete | off, conservative, default, aggressive | Controls whether and how strongly densification is applied. |
| `densify_threshold_mult` | continuous | 0.25 to 4.0, log-scaled | Multiplies the default view-space gradient threshold. Lower means more densification. |
| `prune_opacity_threshold` | continuous | 0.001 to 0.02 | Controls low-opacity pruning strength. |
| `opacity_reset` | discrete | no_reset, reset_if_plateau, force_reset | Decides whether opacity should be reset in this block. |
| `position_lr_mult` | continuous | 0.25 to 4.0, log-scaled | Controls position learning speed. |
| `feature_lr_mult` | continuous | 0.25 to 2.0, log-scaled | Controls SH/color learning speed. |
| `opacity_lr_mult` | continuous | 0.25 to 2.0, log-scaled | Controls opacity learning speed. |
| `scale_rotation_lr_mult` | continuous | 0.25 to 2.0, log-scaled | Controls covariance-related learning speed. |
| `stop` | discrete | continue, stop | Allows early stopping when marginal improvement is low. |

### MVP Notes

- Keep `lambda_dssim` fixed initially.
- Keep SH degree progression fixed initially.
- Keep resolution fixed initially unless training is too slow.
- Use the stock 3DGS split/clone densification mechanism first.
- Log all candidate extended controls, but do not allow the policy to control them yet.

---

## 8.2 Extended Phase 1 Action Space

Add these after the MVP policy is stable.

| Action group | Type | Choices or range | Purpose |
|---|---:|---|---|
| `resolution_scale` | discrete | 0.25, 0.5, 1.0 | Low resolution early, full resolution later. |
| `loss_profile` | discrete | L1_heavy, balanced, DSSIM_heavy | Changes L1/DSSIM weighting during training. |
| `lambda_dssim` | continuous | 0.05 to 0.35 | Continuous DSSIM weight control. |
| `sh_policy` | discrete | hold, default_increase, accelerate, defer | Controls when SH degree increases. |
| `random_background` | discrete | off, on | Useful for robustness on alpha/background-sensitive scenes. |
| `densification_interval` | discrete | 50, 100, 200, 400 | Controls how frequently densification is attempted. |
| `densify_from_policy` | discrete | stats_only, allow_densify | Prevents premature densification before enough signal exists. |
| `max_new_gaussians_frac` | continuous | 0.01 to 0.25 | Caps newly added Gaussians per densification event. |
| `center_selection_mode` | discrete | stock_gradient, residual_weighted, coverage_underfit | Selects where new Gaussians should be proposed. |
| `prune_mode` | discrete | off, opacity_only, opacity_and_size, budgeted | Controls how pruning is applied. |
| `budget_pressure` | continuous | 0.0 to 1.0 | Controls how aggressively to enforce Gaussian-count budget. |
| `exposure_lr_mult` | continuous | 0.25 to 2.0 | For datasets using exposure compensation. |
| `rollback_request` | discrete | no, request_safe_rollback | Optional action for recovery from unstable regions. |

### Extended Notes

- `center_selection_mode = stock_gradient` should call the existing 3DGS densification logic.
- `residual_weighted` can use per-view residual heatmaps to bias clone/split candidates.
- `coverage_underfit` can favor regions visible from poorly reconstructed or under-sampled cameras.
- `budgeted` pruning should prune low-contribution Gaussians only when the model exceeds a soft budget.

---

## 9. Action Execution Rules

### 9.1 Densification Rules

For Phase 1 MVP, densification should remain compatible with the standard 3DGS implementation.

Policy controls:

- whether densification is active,
- threshold multiplier,
- interval,
- maximum new Gaussian fraction,
- conservative/default/aggressive mode.

Suggested mapping:

```text
conservative: high threshold, longer interval, small new-Gaussian cap
default: original 3DGS threshold and interval
aggressive: lower threshold, shorter interval, larger new-Gaussian cap
off: collect stats but do not split/clone
```

### 9.2 Pruning Rules

Pruning should be conservative early in training and stronger near the end or when memory pressure is high.

Policy controls:

- opacity threshold,
- size-aware pruning on/off,
- budget-aware pruning pressure,
- no-prune warmup period.

Avoid allowing the policy to prune too aggressively before a minimum number of iterations has passed.

### 9.3 Opacity Reset Rules

Opacity reset can improve recovery but can also destabilize reward. Treat it as a discrete action with safety guards.

Allowed modes:

- no reset,
- reset if validation plateau is detected,
- force reset, only if minimum interval since previous reset is satisfied.

Hard limits:

- no reset before enough training has occurred,
- no more than a small number of resets per episode,
- no repeated resets in adjacent blocks.

### 9.4 Learning-Rate Rules

The policy should output multipliers, not raw learning rates. Multipliers are applied to safe base schedules.

Control these groups:

- position,
- SH/features,
- opacity,
- scale,
- rotation,
- exposure if enabled.

Clamp all multipliers to safe ranges and use smoothing to avoid abrupt changes.

### 9.5 Camera Sampling Rules

Camera sampling is important for generalization to new camera positions.

Allowed modes:

- `uniform`: original-style random camera sampling,
- `loss_weighted`: sample cameras with high recent training residual,
- `coverage_balanced`: favor cameras or angular regions that have not been sampled recently,
- `validation_proxy_aware`: favor training cameras near poorly reconstructed validation/replay viewpoints, without using test cameras.

Do not let the agent directly sample validation cameras for optimization. Validation views are for reward and diagnostics only.

---

## 10. Suggested Configuration File

```yaml
phase: 1_static_3dgs
rl_algorithm: PPO
policy:
  encoder: mlp
  hidden_width: 256
  hidden_layers: 3
  activation: gelu
  continuous_head: beta
  entropy_weight: 0.003
  value_weight: 0.5
  target_kl: 0.03

environment:
  raw_optimizer: graphdeco_3dgs
  decision_block_steps_default: 100
  max_episode_iterations: 7000
  validation_interval_blocks: 1
  full_validation_interval_blocks: 10
  hard_max_gaussians: 3000000
  hard_max_vram_gb: 24
  rollback_on_nan: true
  no_test_camera_access: true

reward:
  use_validation_psnr_gain: true
  use_ssim_gain: true
  use_lpips_penalty: false
  time_penalty_weight: 0.02
  gaussian_growth_penalty_weight: 0.001
  vram_penalty_weight: 0.01
  validation_drop_penalty_weight: 0.1
  terminal_quality_weight: 0.1

actions_mvp:
  block_steps: [50, 100, 200]
  camera_sampler: [uniform, loss_weighted, coverage_balanced]
  densify_mode: [off, conservative, default, aggressive]
  densify_threshold_mult: [0.25, 4.0]
  prune_opacity_threshold: [0.001, 0.02]
  opacity_reset: [no_reset, reset_if_plateau, force_reset]
  position_lr_mult: [0.25, 4.0]
  feature_lr_mult: [0.25, 2.0]
  opacity_lr_mult: [0.25, 2.0]
  scale_rotation_lr_mult: [0.25, 2.0]
  stop: [continue, stop]

generalization_randomization:
  camera_count: true
  camera_dropout: true
  camera_pose_noise: true
  image_resolution: true
  sparse_point_density: true
  train_val_split: true
  scene_scale: true
```

---

## 11. Environment Step Pseudocode

```python
def step(action):
    # 1. Decode and clamp action
    controls = decode_action(action)
    controls = apply_safety_clamps(controls)

    # 2. Update optimizer and scheduler controls
    set_lr_multipliers(controls.lr_multipliers)
    set_camera_sampler(controls.camera_sampler)
    set_resolution_scale(controls.resolution_scale)
    set_loss_profile(controls.loss_profile)

    # 3. Run a block of low-level 3DGS iterations
    for i in range(controls.block_steps):
        camera = sample_training_camera()
        render_outputs = render(camera)
        loss = compute_training_loss(render_outputs, camera)
        loss.backward()
        collect_gradient_and_visibility_stats(render_outputs)

        if should_densify(i, controls):
            densify_and_prune_with_policy_controls(controls)

        if should_reset_opacity(i, controls):
            reset_opacity_safely(controls)

        optimizer_step()

        if numerical_failure_detected():
            rollback_or_terminate()
            break

    # 4. Evaluate compact validation subset
    validation_metrics = evaluate_validation_subset()

    # 5. Build next observation
    obs = build_observation(validation_metrics, gaussian_stats, compute_stats, previous_action=action)

    # 6. Compute reward
    reward = compute_block_reward(validation_metrics, compute_stats, gaussian_stats)

    # 7. Check stopping conditions
    done = budget_exhausted() or action.stop == "stop" or hard_failure()

    return obs, reward, done, info
```

---

## 12. Baselines

Compare Phase 1 against at least these baselines:

1. **Original fixed 3DGS schedule** using default training parameters.
2. **Short-budget fixed 3DGS** at 1k, 3k, 5k, and 7k iterations.
3. **Hand-tuned fast schedule** with reduced densification and fewer iterations.
4. **Greedy residual camera sampler** without RL.
5. **RL without densification actions** to test whether densification control matters.
6. **RL without camera sampling actions** to test whether sensing-aware control matters.
7. **RL without Gaussian-count penalty** to test whether the policy over-grows the model.

---

## 13. Evaluation Metrics

Report performance as Pareto curves, not only final quality.

Required metrics:

- PSNR at fixed wall-clock budgets,
- SSIM at fixed wall-clock budgets,
- LPIPS if available,
- time to reach target PSNR,
- final number of Gaussians,
- peak VRAM,
- model size on disk,
- training time,
- render FPS,
- action trace interpretability,
- stability rate across scenes,
- performance on unseen camera layouts.

Generalization metrics:

- train scenes versus unseen scenes,
- familiar camera density versus sparse camera density,
- original cameras versus perturbed cameras,
- dense trajectories versus replay-like held-out trajectories,
- high-quality SfM initialization versus degraded initialization.

---

## 14. Acceptance Criteria for Phase 1

Phase 1 should be considered successful if the MVP policy satisfies at least one of these criteria:

- reaches the same validation PSNR as fixed 3DGS using less wall-clock time,
- improves validation PSNR at the same fixed wall-clock budget,
- reduces Gaussian count or peak memory at matched quality,
- shows stable generalization to unseen scenes and camera layouts,
- produces interpretable action traces that differ across scene types and training stages.

A strong target is:

- comparable or better quality than fixed 3DGS at 7k iterations,
- 20% to 50% reduction in time-to-quality on at least some scene classes,
- no catastrophic failures on the validation suite,
- visible policy adaptation to camera sparsity and residual distribution.

---

## 15. Implementation Milestones

### Milestone 1 — Instrumented 3DGS Baseline

- Run default 3DGS training from script.
- Log losses, PSNR, SSIM, Gaussian count, VRAM, time per iteration.
- Save per-camera residual statistics.
- Save Gaussian opacity/scale/radius histograms.

### Milestone 2 — Environment Wrapper

- Convert training loop into reset/step API.
- Add block-level action execution.
- Add validation subset evaluation.
- Add reward computation.
- Add rollback on failure.

### Milestone 3 — Heuristic Policies

Before RL, implement heuristic controllers:

- fixed default schedule,
- conservative schedule,
- aggressive schedule,
- residual-weighted camera schedule,
- plateau-triggered opacity reset.

These provide debugging baselines and possible behavior-cloning data.

### Milestone 4 — PPO MVP

- Train PPO with MVP action space.
- Use small scenes and short budgets first.
- Confirm reward increases over episodes.
- Confirm policy does not collapse to a single unsafe action.
- Inspect action traces.

### Milestone 5 — Generalization Tests

- Test on unseen scenes.
- Test on sparse camera layouts.
- Test on perturbed camera poses.
- Test on degraded initial point clouds.
- Compare to fixed and heuristic schedules.

### Milestone 6 — Extended Actions

Only after MVP is stable:

- add resolution scheduling,
- add SH progression control,
- add budget-aware pruning,
- add residual-weighted center selection,
- add DSSIM weight control.

---

## 16. Risks and Mitigations

| Risk | Symptom | Mitigation |
|---|---|---|
| Reward too noisy | PPO unstable, no learning | Use validation EMA, larger block sizes, reward normalization. |
| Policy over-densifies | High quality but huge memory | Add Gaussian growth and VRAM penalties; hard cap Gaussian count. |
| Policy prunes too early | Irrecoverable quality loss | Add no-prune warmup and rollback. |
| Policy overfits to camera layout | Poor performance on new camera positions | Domain-randomize camera counts, baselines, dropout, and pose noise. |
| Validation overhead too high | RL training too slow | Evaluate subset every block; full validation sparsely. |
| Continuous controls unstable | Sudden divergence | Use multiplier smoothing and bounded distributions. |
| Action space too large | No clear learning signal | Start with MVP action space and add groups only after ablation. |

---

## 17. What Not to Include in Phase 1

Do not include these yet:

- dynamic 4D GS,
- temporal Gaussian duration or velocity,
- FreeTimeGS primitives,
- LongVolCap hierarchy,
- 2DGS/3DGS/4DGS representation switching,
- entropy coding or MPEG-style compression,
- learned raw-image policy inputs,
- test-camera training,
- online RL during deployment.

Phase 1 should prove the core control idea on static 3DGS first.

---

## 18. Reference Implementation Anchors

The original 3DGS implementation already exposes many variables that are natural Phase 1 controls:

- total iterations,
- position learning rate,
- feature/SH learning rate,
- opacity learning rate,
- scaling learning rate,
- rotation learning rate,
- percent dense,
- DSSIM weight,
- densification interval,
- opacity reset interval,
- densify-from iteration,
- densify-until iteration,
- densification gradient threshold,
- random background,
- optimizer type.

The default 3DGS training loop also already includes useful monitoring signals:

- training L1 loss,
- total loss,
- iteration time,
- validation PSNR,
- opacity histogram,
- total point count,
- view-space point gradients,
- visibility filters,
- screen-space radii.

Phase 1 should reuse these variables before adding new custom mechanisms.

---

## 19. Recommended MVP Deliverable

The first deliverable should be a reproducible experiment folder with:

```text
agentic_gs_phase1/
  envs/
    gs_env.py
    wrappers.py
  policies/
    ppo_policy.py
    action_heads.py
  configs/
    phase1_mvp.yaml
    phase1_eval.yaml
  scripts/
    train_agent.py
    eval_agent.py
    run_fixed_baselines.py
    plot_pareto_curves.py
  logs/
    README.md
  docs/
    action_space.md
    observation_space.md
```

Minimum experiment to run:

```bash
python scripts/run_fixed_baselines.py --config configs/phase1_eval.yaml
python scripts/train_agent.py --config configs/phase1_mvp.yaml
python scripts/eval_agent.py --checkpoint <policy_ckpt> --config configs/phase1_eval.yaml
python scripts/plot_pareto_curves.py --results_dir <results>
```

---

## 20. Summary

Phase 1 should demonstrate that a learned, offline-trained RL controller can make useful decisions during static 3DGS fitting. The key is to keep the action space structured and interpretable, use compact solver-state observations, reward quality improvement per compute, and train across randomized scenes and camera layouts. A successful Phase 1 will provide the foundation for later adding 4D temporal primitives, hierarchical memory, and compression-aware training.
