# Baseline 3D Gaussian Splatting

This workspace vendors the official GraphDECO 3DGS implementation in `gaussian-splatting/` and adds only fixed-schedule baseline plumbing. No RL environment, policy, action space, or heuristic controller is implemented here.

## Dataset

The archive is a NeRF synthetic multi-view dataset:

```text
archive/nerf_synthetic/<scene>/
  transforms_train.json
  transforms_test.json
  train/*.png
  test/*.png
```

The stock 3DGS loader recognizes this format through `transforms_train.json`. Use `--white_background` for these synthetic scenes.

## Setup Check

Run this after activating the 3DGS environment:

```powershell
python scripts/check_3dgs_setup.py
```

If the CUDA extensions are missing, create the official environment:

```powershell
conda env create --file gaussian-splatting/environment.yml
conda activate gaussian_splatting
python scripts/check_3dgs_setup.py
```

On Windows, the CUDA extensions also need MSVC build tools available in the active shell.

## Run Baseline

Dry-run the exact commands:

```powershell
python scripts/run_3dgs_baseline.py --scene ship --iterations 30000 --dry-run
```

Run the default fixed 3DGS baseline:

```powershell
python scripts/run_3dgs_baseline.py --scene ship --iterations 30000
```

Useful short smoke run:

```powershell
python scripts/run_3dgs_baseline.py --scene ship --iterations 1000 --no-render --no-metrics
```

Outputs are written to:

```text
outputs/3dgs_baseline/<scene>/fixed_<iterations>/
```

The patched stock trainer writes:

```text
baseline_train_metrics.csv
baseline_eval_metrics.csv
baseline_histograms.jsonl
baseline_manifest.json
logs/train.log
```

If rendering and metrics are enabled, the original `render.py` and `metrics.py` also populate the standard `test/`, `results.json`, and `per_view.json` outputs.
