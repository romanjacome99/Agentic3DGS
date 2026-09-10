# Agentic3DGS

Reinforcement-learned control of 3D Gaussian Splatting (3DGS) training. A PPO policy
acts once per block of optimizer iterations and decides how to densify, prune, reset
opacity, schedule learning rates, and when to stop — optimized so that maximizing
return means **accelerating** reconstruction at a bounded model size.

## Headline result

The frozen best policy (**v7**) reaches every reachable test-PSNR target in
**1.24–1.53× less training wall-clock time** than the fixed 3DGS schedule, across
three independent scenes — including `materials`, which the policy never saw in
training, evaluation, or checkpoint selection — while holding the model at or below
the baseline Gaussian budget.

| Scene | Role | Speed-up (claim-ready targets) | Agent peak |
|---|---|---|---|
| hotdog | held-out | 1.24–1.37× (25–33 dB) | 34.1 dB / ~55k G |
| materials | held-out (never trained) | 1.29–1.53× (20–26 dB) | 26.2 dB / ~55k G |
| drums | training scene | 1.24–1.37× (18–24 dB) | 24.6 dB / ~77k G |

## Interactive demo website

`website/` is a static, dependency-free page (WebGL2 Gaussian-splat viewer + decision-log explorer)
built from the saved training snapshots and evaluation logs of the trained controllers. Preview it with
`python -m http.server` inside `website/`; `.github/workflows/pages.yml` deploys it to GitHub Pages
(set *Settings → Pages → Source: GitHub Actions* once). See `website/README.md`.

## Repository layout

```
agentic_gs_phase1/
  envs/          block-wise RL environment around the stock 3DGS optimizer
  policies/      actor-critic network + PPO update
  scripts/       train_agent.py, eval_agent.py, time_to_target_protocol.py, plotting
  configs/       phase1_mvp.json (v7 setup), phase1_eval.json, phase1_fast.json
  docs/          evaluation_protocol.md, agentic_gs_approach.tex / .pdf / .html
configs/         top-level baseline configs
scripts/         experiment driver scripts (PowerShell)
phase1_agentic_gs_rl_spec.md   full design spec
BASELINE_3DGS.md               fixed-baseline notes
```

The interactive results page is `agentic_gs_phase1/docs/agentic_gs_approach.html`
(self-contained; open in any browser — math via MathJax, charts via Chart.js).

## Not in version control

These are excluded by `.gitignore` (too large / reproducible / machine-specific):

- **`gaussian-splatting/`** — the upstream 3DGS implementation. Clone it into the
  project root:
  ```bash
  git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
  ```
- **`archive/` and `archive.zip`** — the NeRF-synthetic dataset (`chair`, `drums`,
  `ficus`, `hotdog`, `lego`, `materials`, `mic`, `ship`). Place the scenes under
  `archive/nerf_synthetic/<scene>/`.
- **`outputs/`** — all training runs, checkpoints (`*.pth`), rendered views, and
  generated plots.

## Setup

```bash
conda activate env_pytorch_3dgs   # PyTorch + CUDA + the 3DGS rasterizer
```

The environment requires CUDA; it wraps the unmodified 3DGS optimizer.

## Train

For a storage-limited DL3DV training pool, see [DL3DV subset setup](docs/DL3DV_SUBSET.md).
A reproducible 500-scene 480P selection and resumable downloader are included.
The full DL3DV release needs COLMAP preparation before the current backends can train on it.

```bash
python agentic_gs_phase1/scripts/train_agent.py \
  --config agentic_gs_phase1/configs/phase1_mvp.json \
  --run-name my_run --max-updates 150
```

Checkpoints are written to `outputs/agentic_gs_phase1/<run>/checkpoints/`. Select
`best.pth` (chosen by the acceleration-aware probe), not `final.pth`.

## Evaluate (time-to-target)

```bash
python agentic_gs_phase1/scripts/time_to_target_protocol.py \
  --checkpoint outputs/agentic_gs_phase1/<run>/checkpoints/best.pth \
  --config agentic_gs_phase1/configs/phase1_eval.json \
  --scene hotdog --run-prefix tt_hotdog \
  --report-dir outputs/agentic_gs_phase1_reports/hotdog_tt \
  --run-missing --skip-render-images --validation-cameras 12
```

`eval` config note: `safety.min_iterations_before_stop` must match the value used at
training time (3000), or the policy stops out-of-distribution and the curve is invalid.

Plot the report artifacts:

```bash
python agentic_gs_phase1/scripts/plot_time_to_target_artifacts.py \
  --report-dir outputs/agentic_gs_phase1_reports/hotdog_tt
```

See `agentic_gs_phase1/docs/evaluation_protocol.md` for the full protocol.
