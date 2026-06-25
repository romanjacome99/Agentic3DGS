import csv, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(r"C:\Roman\3DGS_PROPOSAL\outputs\agentic_rl_real\ppo_real_fastergs_v1\ttp_eval")
rows = list(csv.DictReader(open(d / "curve.csv")))
def series(m, x, y):
    r = sorted((float(z[x]), float(z[y])) for z in rows if z["method"] == m)
    return [a for a, _ in r], [b for _, b in r]

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for m, c, lbl in [("baseline", "#5b6b8c", "Fixed 3DGS"), ("agentic", "#e0682f", "FasterGS-Agent")]:
    t, p = series(m, "t", "psnr"); ax[0].plot(t, p, "-o", color=c, label=lbl, lw=2.2, ms=5)
    n, p2 = series(m, "N", "psnr"); ax[1].plot([x/1000 for x in n], p2, "-o", color=c, label=lbl, lw=2.2, ms=5)
ax[0].set_xlabel("training wall-clock (s)"); ax[0].set_ylabel("held-out test PSNR (dB)")
ax[0].set_title("Speed: PSNR vs time (held-out 'train', FasterGS)")
ax[1].set_xlabel("Gaussians (k)"); ax[1].set_ylabel("held-out test PSNR (dB)")
ax[1].set_title("Compactness: PSNR vs model size")
for a in ax: a.grid(alpha=.3); a.legend()
fig.tight_layout(); fig.savefig(d / "ttp_curve.png", dpi=130)
print("wrote", d / "ttp_curve.png")
