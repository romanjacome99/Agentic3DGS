"""Per-budget PSNR/SSIM comparison of the budget-conditioned agent vs the fixed-schedule
baseline, for both backends. Emits:
  Paper/figures/budget_vs_baseline.pdf   (PSNR-vs-budget and SSIM-vs-budget, agent vs baseline)
  Paper/tables/budget_vs_baseline.tex    (per-budget table body rows)

Agent metrics come from the 10-seed sweep (multiseed10). Baseline metrics come from the
matched 10-seed baseline sweep (multiseed10_baseline) when available; otherwise it falls
back to the single-run baseline (ultimate/*_baseline/budget_sweep.csv) and marks std as n/a.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
BUD = ROOT / "outputs/agentic_rl_budget"
FIG = ROOT / "Paper/figures"; FIG.mkdir(parents=True, exist_ok=True)
TAB = ROOT / "Paper/tables"; TAB.mkdir(parents=True, exist_ok=True)

BUDGETS = ["30", "60", "120", "300", "360", "unlimited"]
BX = {"30": 30, "60": 60, "120": 120, "300": 300, "360": 360, "unlimited": 420}
BLABEL = {"30": "30", "60": "60", "120": "120", "300": "300", "360": "360", "unlimited": r"$\infty$"}
BROW = {"30": "30\\,s", "60": "60\\,s", "120": "120\\,s", "300": "300\\,s", "360": "360\\,s", "unlimited": "$\\infty$"}
BACKENDS = [("FasterGS", "fastergs", "#e0682f"), ("3DGS", "3dgs", "#3a5da8")]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def agent(short):
    """10-seed agent summary -> {budget: (psnr_m, psnr_s, ssim_m, ssim_s)}."""
    d = {}
    for r in rows(BUD / "multiseed10" / f"{short}_agent" / "summary.csv"):
        d[r["budget"]] = (float(r["test_psnr_mean"]), float(r["test_psnr_std"]),
                          float(r["test_ssim_mean"]), float(r["test_ssim_std"]))
    return d


def baseline(short):
    """Prefer matched 10-seed baseline; else single-run. Returns (dict, has_std)."""
    ms = BUD / "multiseed10_baseline" / f"{short}_baseline" / "summary.csv"
    if ms.exists():
        d = {}
        for r in rows(ms):
            d[r["budget"]] = (float(r["test_psnr_mean"]), float(r["test_psnr_std"]),
                              float(r["test_ssim_mean"]), float(r["test_ssim_std"]))
        return d, True
    d = {}
    for r in rows(BUD / "ultimate" / f"{short}_baseline" / "budget_sweep.csv"):
        b = "unlimited" if float(r["budget_s"]) >= 1e6 else str(int(float(r["budget_s"])))
        d[b] = (float(r["test_psnr"]), 0.0, float(r["test_ssim"]), 0.0)
    return d, False


# ---------------- figure ----------------
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.7))
has_std_any = False
for name, short, c in BACKENDS:
    ag = agent(short); bl, hs = baseline(short); has_std_any |= hs
    xs = [BX[b] for b in BUDGETS]
    ap = [ag[b][0] for b in BUDGETS]; aps = [ag[b][1] for b in BUDGETS]
    asm = [ag[b][2] for b in BUDGETS]; ass = [ag[b][3] for b in BUDGETS]
    bp = [bl[b][0] for b in BUDGETS]; bps = [bl[b][1] for b in BUDGETS]
    bsm = [bl[b][2] for b in BUDGETS]; bss = [bl[b][3] for b in BUDGETS]
    ax[0].errorbar(xs, ap, yerr=aps, fmt="-o", color=c, capsize=3, lw=2, ms=5, label=f"{name}-Agent")
    ax[0].errorbar(xs, bp, yerr=bps, fmt="--s", color=c, capsize=3, lw=1.6, ms=5, mfc="white", label=f"{name} baseline")
    ax[1].errorbar(xs, asm, yerr=ass, fmt="-o", color=c, capsize=3, lw=2, ms=5, label=f"{name}-Agent")
    ax[1].errorbar(xs, bsm, yerr=bss, fmt="--s", color=c, capsize=3, lw=1.6, ms=5, mfc="white", label=f"{name} baseline")
for a, ylab, title in [(ax[0], "test PSNR (dB)", "(a) PSNR vs budget"), (ax[1], "test SSIM", "(b) SSIM vs budget")]:
    a.set_xticks(list(BX.values())); a.set_xticklabels([BLABEL[b] for b in BUDGETS])
    a.set_xlabel("wall-clock budget (s)"); a.set_ylabel(ylab); a.set_title(title, fontsize=10)
    a.grid(alpha=.3)
ax[0].legend(fontsize=7.5, ncol=2, loc="lower right")
fig.tight_layout(); fig.savefig(FIG / "budget_vs_baseline.pdf", bbox_inches="tight")
print("wrote", FIG / "budget_vs_baseline.pdf", "(baseline 10-seed:", has_std_any, ")")


# ---------------- table body ----------------
def fmt(m, s, prec):
    return f"${m:.{prec}f}$" if s == 0.0 else f"${m:.{prec}f}{{\\scriptstyle\\pm{s:.{prec}f}}}$"

lines = [r"\begin{tabular}{llccccc}", r"\toprule",
         r"        &        & \multicolumn{2}{c}{PSNR (dB)} & $\Delta$PSNR & \multicolumn{2}{c}{SSIM} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
         r"Backend & Budget & Agent & Baseline & (dB) & Agent & Baseline \\", r"\midrule"]
for name, short, _ in BACKENDS:
    ag = agent(short); bl, hs = baseline(short)
    n = len(BUDGETS)
    for i, b in enumerate(BUDGETS):
        ap, aps, asm, ass = ag[b]; bp, bps, bsm, bss = bl[b]
        dp = ap - bp
        lead = f"\\multirow{{{n}}}{{*}}{{{name}}}\n" if i == 0 else ""
        row = (f"{lead}& {BROW[b]} & {fmt(ap,aps,2)} & {fmt(bp,bps,2)} & "
               f"${dp:+.2f}$ & {fmt(asm,ass,3)} & {fmt(bsm,bss,3)} \\\\")
        lines.append(row)
    if short != BACKENDS[-1][1]:
        lines.append("\\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(TAB / "budget_vs_baseline.tex").write_text("\n".join(lines) + "\n")
print("wrote", TAB / "budget_vs_baseline.tex")
