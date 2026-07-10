"""Generate the densify-vs-budget figure (with error bars) for the paper."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
MS = ROOT / "outputs/agentic_rl_budget/multiseed10"
OUT = ROOT / "Paper/figures"; OUT.mkdir(parents=True, exist_ok=True)

def load(agent):
    xs, ym, ys = [], [], []
    for r in csv.DictReader(open(MS / agent / "summary.csv")):
        b = r["budget"]
        x = 420 if b == "unlimited" else float(b)
        xs.append(x); ym.append(float(r["densify_on_blocks_mean"])); ys.append(float(r["densify_on_blocks_std"]))
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    return [xs[i] for i in order], [ym[i] for i in order], [ys[i] for i in order]

fig, ax = plt.subplots(figsize=(5.0, 3.4))
for agent, c, lbl in [("3dgs_agent", "#3a5da8", "3DGS-Agent"), ("fastergs_agent", "#e0682f", "FasterGS-Agent")]:
    x, ym, ys = load(agent)
    ax.errorbar(x, ym, yerr=ys, fmt="-o", color=c, capsize=3, lw=2, ms=5, label=lbl)
ax.set_xlabel("wall-clock budget (s)"); ax.set_ylabel("blocks with densification active")
ax.set_xticks([30, 60, 120, 300, 360, 420]); ax.set_xticklabels(["30", "60", "120", "300", "360", r"$\infty$"])
ax.grid(alpha=.3); ax.legend()
fig.tight_layout(); fig.savefig(OUT / "densify_vs_budget.pdf", bbox_inches="tight")
print("wrote", OUT / "densify_vs_budget.pdf")
