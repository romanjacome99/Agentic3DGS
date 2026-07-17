"""Training AUC: area under the test-PSNR / test-SSIM vs wall-clock-time curve, agent vs
fixed-schedule baseline, from the protocol curve.csv files. Higher AUC = reaches and holds
quality sooner. We integrate (trapezoid) over a common window [t0, T] both methods cover,
and report the raw area plus the time-averaged value (AUC / window)."""
import csv
from pathlib import Path
import numpy as np

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
PROT = ROOT / "outputs/agentic_rl_real"


def load(backend):
    d = {"agentic": ([], [], []), "baseline": ([], [], [])}
    for r in csv.DictReader(open(PROT / f"protocol_{backend}_train_30k/curve.csv")):
        t, p, s = float(r["t"]), float(r["test_psnr"]), float(r["test_ssim"])
        d[r["method"]][0].append(t); d[r["method"]][1].append(p); d[r["method"]][2].append(s)
    for m in d:
        o = np.argsort(d[m][0])
        d[m] = (np.array(d[m][0])[o], np.array(d[m][1])[o], np.array(d[m][2])[o])
    return d


def auc(backend):
    d = load(backend)
    t0 = max(d["agentic"][0][0], d["baseline"][0][0])
    T = min(d["agentic"][0][-1], d["baseline"][0][-1])
    grid = np.linspace(t0, T, 1000)
    res = {"backend": backend, "window_s": (round(t0, 1), round(T, 1))}
    for m in ("agentic", "baseline"):
        t, p, s = d[m]
        pg = np.interp(grid, t, p); sg = np.interp(grid, t, s)
        res[m] = {
            "auc_psnr_dBs": round(float(np.trapz(pg, grid)), 1),
            "auc_ssim_s": round(float(np.trapz(sg, grid)), 2),
            "mean_psnr": round(float(np.trapz(pg, grid) / (T - t0)), 3),
            "mean_ssim": round(float(np.trapz(sg, grid) / (T - t0)), 4),
        }
    a, b = res["agentic"], res["baseline"]
    res["gain_mean_psnr"] = round(a["mean_psnr"] - b["mean_psnr"], 3)
    res["auc_ratio_psnr"] = round(a["auc_psnr_dBs"] / b["auc_psnr_dBs"], 3)
    return res


out = {}
for backend in ("3dgs", "fastergs"):
    r = auc(backend)
    out[backend] = r
    w = r["window_s"]
    print(f"[{backend}] window {w[0]}-{w[1]}s | "
          f"mean PSNR agent {r['agentic']['mean_psnr']} vs base {r['baseline']['mean_psnr']} "
          f"(+{r['gain_mean_psnr']} dB) | AUC ratio {r['auc_ratio_psnr']}x | "
          f"mean SSIM agent {r['agentic']['mean_ssim']} vs base {r['baseline']['mean_ssim']}")
(ROOT / "outputs/metrics").mkdir(parents=True, exist_ok=True)
(ROOT / "outputs/metrics/auc.json").write_text(__import__("json").dumps(out, indent=1))
print("wrote outputs/metrics/auc.json")
