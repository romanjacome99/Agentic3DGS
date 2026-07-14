"""Generate the learned action-set figure (2x2) for the paper method section.

Top row  = acceleration mode: the per-iteration schedule the agent runs on `train`.
Bottom   = budget-conditioned mode: the action set as a function of the queried budget
           (mean over 10 seeds).
Left col  = discrete densification decision; right col = continuous multipliers.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
ACC = ROOT / "outputs/agentic_rl_real"
MS = ROOT / "outputs/agentic_rl_budget/multiseed10"
OUT = ROOT / "Paper/figures"; OUT.mkdir(parents=True, exist_ok=True)

SEEDS = list(range(10))
BUDGETS = [30, 60, 120, 300, 360, 10000000]   # last = unlimited (30k cap)
BLABEL = ["30", "60", "120", "300", "360", r"$\infty$"]
BX = [30, 60, 120, 300, 360, 420]             # plot x for unlimited
C3, CF = "#3a5da8", "#e0682f"                  # 3DGS / FasterGS colours
CONT = [("position_lr_mult", "pos-lr", "#3a5da8"),
        ("opacity_lr_mult", "opac-lr", "#e0682f"),
        ("scaling_lr_mult", "scale-lr", "#3f9950"),
        ("densify_threshold_mult", "densify-thr", "#8a5cc0")]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


# ---------- acceleration: per-iteration schedule ----------
def accel(agent):
    r = rows(ACC / f"protocol_{agent}_train_30k/agentic_blocks.csv")
    it = np.array([float(x["iter"]) for x in r]) / 1000.0
    dens = np.array([0.0 if x["densify_mode"].strip() == "off" else 1.0 for x in r])
    cont = {k: np.array([float(x[k]) for x in r]) for k, _, _ in CONT}
    return it, dens, cont


# ---------- budget: action set vs budget (10-seed mean) ----------
def budget(agent):
    dfrac_m, dfrac_s = [], []
    cont_m = {k: [] for k, _, _ in CONT}
    for B in BUDGETS:
        per_seed_frac, per_seed_cont = [], {k: [] for k, _, _ in CONT}
        for s in SEEDS:
            p = MS / agent / "blocks" / f"seed{s}_B{B}.csv"
            if not p.exists():
                continue
            r = rows(p)
            if not r:
                continue
            per_seed_frac.append(np.mean([0.0 if x["densify_mode"].strip() == "off" else 1.0 for x in r]))
            for k, _, _ in CONT:
                per_seed_cont[k].append(np.mean([float(x[k]) for x in r]))
        dfrac_m.append(np.mean(per_seed_frac)); dfrac_s.append(np.std(per_seed_frac))
        for k, _, _ in CONT:
            cont_m[k].append(np.mean(per_seed_cont[k]))
    return np.array(dfrac_m), np.array(dfrac_s), {k: np.array(v) for k, v in cont_m.items()}


fig, ax = plt.subplots(2, 2, figsize=(9.6, 6.4))

# (a) acceleration discrete -- densify active over iterations
it3, d3, c3cont = accel("3dgs")
itf, df, cfcont = accel("fastergs")
ax[0, 0].step(it3, d3, where="post", color=C3, lw=2, label="3DGS-Agent")
ax[0, 0].step(itf, df, where="post", color=CF, lw=2, ls="--", label="FasterGS-Agent")
ax[0, 0].set_ylim(-0.15, 1.35); ax[0, 0].set_yticks([0, 1]); ax[0, 0].set_yticklabels(["off", "on"])
ax[0, 0].set_xlabel("training iteration (k)"); ax[0, 0].set_ylabel("densification")
ax[0, 0].set_title("(a) Acceleration: densify decision", fontsize=10)
ax[0, 0].legend(fontsize=8, loc="center right"); ax[0, 0].grid(alpha=.3)

# (b) acceleration continuous -- multipliers over iterations (3DGS agent)
for k, lbl, c in CONT:
    ax[0, 1].plot(it3, c3cont[k], color=c, lw=1.8, label=lbl)
ax[0, 1].axhline(1.0, color="gray", lw=.8, ls=":")
ax[0, 1].set_xlabel("training iteration (k)"); ax[0, 1].set_ylabel(r"action multiplier ($\times$ default)")
ax[0, 1].set_title("(b) Acceleration: continuous actions (3DGS)", fontsize=10)
ax[0, 1].legend(fontsize=8, ncol=2); ax[0, 1].grid(alpha=.3)

# (c) budget discrete -- densify fraction vs budget, both backends, 10-seed
fm3, fs3, cont3 = budget("3dgs_agent")
fmf, fsf, contf = budget("fastergs_agent")
ax[1, 0].errorbar(BX, fm3, yerr=fs3, fmt="-o", color=C3, capsize=3, lw=2, ms=5, label="3DGS-Agent")
ax[1, 0].errorbar(BX, fmf, yerr=fsf, fmt="--s", color=CF, capsize=3, lw=2, ms=5, label="FasterGS-Agent")
ax[1, 0].set_xticks(BX); ax[1, 0].set_xticklabels(BLABEL)
ax[1, 0].set_xlabel("wall-clock budget (s)"); ax[1, 0].set_ylabel("fraction of blocks densifying")
ax[1, 0].set_title("(c) Budget: densify decision", fontsize=10)
ax[1, 0].legend(fontsize=8); ax[1, 0].grid(alpha=.3)

# (d) budget continuous -- mean multipliers vs budget (3DGS agent, 10-seed)
for k, lbl, c in CONT:
    ax[1, 1].plot(BX, cont3[k], "-o", color=c, lw=1.8, ms=4, label=lbl)
ax[1, 1].axhline(1.0, color="gray", lw=.8, ls=":")
ax[1, 1].set_xticks(BX); ax[1, 1].set_xticklabels(BLABEL)
ax[1, 1].set_xlabel("wall-clock budget (s)"); ax[1, 1].set_ylabel(r"mean action multiplier ($\times$ default)")
ax[1, 1].set_title("(d) Budget: continuous actions (3DGS)", fontsize=10)
ax[1, 1].legend(fontsize=8, ncol=2); ax[1, 1].grid(alpha=.3)

fig.tight_layout()
fig.savefig(OUT / "action_sets.pdf", bbox_inches="tight")
print("wrote", OUT / "action_sets.pdf")
