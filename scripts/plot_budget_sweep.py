"""Plot budget-scaling: how a budget-conditioned agent's quality / iterations /
Gaussians / densification change with the given wall-clock budget."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL\outputs\agentic_rl_budget")
RUNS = {"FasterGS-Agent": (ROOT / "sweep_v2_fastergs_train/budget_sweep.csv", "#e0682f"),
        "3DGS-Agent": (ROOT / "sweep_v2_3dgs_train/budget_sweep.csv", "#3a5da8")}
OUT = ROOT / "budget_scaling.png"

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
for name, (p, c) in RUNS.items():
    if not p.exists():
        continue
    r = list(csv.DictReader(open(p)))
    B = [float(x["budget_s"]) for x in r]
    ax[0].plot(B, [float(x["test_psnr"]) for x in r], "-o", color=c, lw=2.2, label=name)
    ax[1].plot(B, [int(x["final_iter"]) for x in r], "-o", color=c, lw=2.2, label=name)
    ax[2].plot(B, [int(x["gaussians"])/1000 for x in r], "-o", color=c, lw=2.2, label=name)
ax[0].set(xlabel="time budget (s)", ylabel="held-out test PSNR (dB)", title="Quality scales with budget")
ax[1].set(xlabel="time budget (s)", ylabel="iterations reached", title="Iterations scale with budget")
ax[2].set(xlabel="time budget (s)", ylabel="Gaussians (k)", title="Model size scales with budget")
for a in ax:
    a.grid(alpha=.3); a.legend(fontsize=9)
fig.suptitle("Budget-conditioned agents on held-out 'train' — behavior adapts to the sampled time budget", y=1.02)
fig.tight_layout(); fig.savefig(OUT, dpi=130, bbox_inches="tight")
print("wrote", OUT)
