"""Combine the 3DGS and FasterGS time-to-target curves into one comparison:
baseline/agent x 3DGS/FasterGS, on a held-out scene.
"""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
ap = argparse.ArgumentParser()
ap.add_argument("--tdg-csv", default=str(ROOT / "outputs/agentic_rl_real/eval_3dgs_train/curve.csv"))
ap.add_argument("--fgs-csv", default=str(ROOT / "outputs/agentic_rl_real/ppo_real_fastergs_v1/ttp_eval/curve.csv"))
ap.add_argument("--out", default=str(ROOT / "outputs/agentic_rl_real/compare_train"))
ap.add_argument("--scene", default="train")
ap.add_argument("--targets", type=float, nargs="+", default=[16, 17, 18, 19])
args = ap.parse_args()
FGS = Path(args.fgs_csv); TDG = Path(args.tdg_csv)
OUT = Path(args.out); OUT.mkdir(parents=True, exist_ok=True)
TARGETS = args.targets

def load(p, tag):
    out = {}
    for r in csv.DictReader(open(p)):
        m = f"{tag}-{r['method']}"
        out.setdefault(m, []).append((float(r["t"]), float(r["psnr"]), float(r["N"])))
    for m in out:
        out[m].sort()
    return out

data = {}
data.update(load(TDG, "3DGS"))
data.update(load(FGS, "FasterGS"))

STYLE = {
    "3DGS-baseline":     ("#9aa6bf", "--", "Baseline 3DGS"),
    "3DGS-agentic":      ("#3a5da8", "-",  "Agent (v8) - 3DGS"),
    "FasterGS-baseline": ("#f0b27a", "--", "Baseline FasterGS"),
    "FasterGS-agentic":  ("#e0682f", "-",  "Agent - FasterGS"),
}

fig, ax = plt.subplots(1, 2, figsize=(13, 4.8))
for key, (c, ls, lbl) in STYLE.items():
    if key not in data:
        continue
    s = data[key]
    ax[0].plot([t for t, _, _ in s], [p for _, p, _ in s], ls, color=c, marker="o", ms=4, lw=2.2, label=lbl)
    ax[1].plot([n/1000 for _, _, n in s], [p for _, p, _ in s], ls, color=c, marker="o", ms=4, lw=2.2, label=lbl)
ax[0].set_xlabel("training wall-clock (s)"); ax[0].set_ylabel("held-out test PSNR (dB)")
ax[0].set_title(f"Time-to-target PSNR - held-out '{args.scene}'")
ax[1].set_xlabel("Gaussians (k)"); ax[1].set_ylabel("held-out test PSNR (dB)")
ax[1].set_title("PSNR vs model size")
for a in ax: a.grid(alpha=.3); a.legend(fontsize=9)
fig.tight_layout(); fig.savefig(OUT / "compare_curve.png", dpi=130)

def time_to(series, tgt):
    for (t0, p0, _), (t1, p1, _) in zip(series, series[1:]):
        if p0 <= tgt <= p1 and p1 > p0:
            return t0 + (t1 - t0) * (tgt - p0) / (p1 - p0)
    return None

print(f"\n=== time-to-target seconds (held-out '{args.scene}') ===")
hdr = f"{'target':>7} | " + " | ".join(f"{STYLE[k][2]:>20}" for k in STYLE if k in data)
print(hdr)
keys = [k for k in STYLE if k in data]
for tg in TARGETS:
    cells = []
    for k in keys:
        t = time_to(data[k], tg)
        cells.append(f"{(round(t,1) if t else '--'):>20}")
    print(f"{tg:>5} dB | " + " | ".join(cells))

print("\n=== speed-ups (baseline_time / method_time) ===")
for tg in TARGETS:
    def tt(k): return time_to(data[k], tg) if k in data else None
    b3, a3 = tt("3DGS-baseline"), tt("3DGS-agentic")
    bf, af = tt("FasterGS-baseline"), tt("FasterGS-agentic")
    def sp(num, den): return round(num/den, 2) if (num and den) else None
    print(f"{tg} dB:  agent/3DGS_base={sp(b3,a3)}x  agent/FGS_base={sp(bf,af)}x  "
          f"FGS_base/3DGS_base={sp(b3,bf)}x  FGS_agent/3DGS_agent={sp(a3,af)}x")
print("wrote", OUT / "compare_curve.png")
