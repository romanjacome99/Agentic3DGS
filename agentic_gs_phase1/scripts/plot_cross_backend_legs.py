"""LeGS-only excerpt of the cross-backend transfer results (held-out `train`), for the side-by-side
table + plot float in the main text.

Emits
  Paper/tables/cross_backend_legs.tex     the three policies run unchanged on the LeGS backend
  Paper/figures/cross_backend_transfer_legs.pdf/.png   the LeGS panel of cross_backend_transfer.pdf
Run with the env_pytorch_3dgs python (same as plot_cross_backend.py).
"""
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_cross_backend import ROOT, POLICIES, TARGETS, T, load, fmt_t  # noqa: E402

TABLE = os.path.join(ROOT, "Paper", "tables", "cross_backend_legs.tex")
FIG = os.path.join(ROOT, "Paper", "figures", "cross_backend_transfer_legs.pdf")


def main():
    tk, tlabel, runs = next(t for t in TARGETS if t[0] == "legs")
    block = {pk: load(run) for pk, run in runs.items()}
    block = {pk: r for pk, r in block.items() if r is not None}
    assert block, "no LeGS transfer runs found"
    order = sorted(block, key=lambda p: list(POLICIES).index(p))

    # ---------------------------------------------------------------- table (transposed: rows = metrics)
    gs = [r["gmean"] for r in block.values() if not math.isnan(r["gmean"])]
    best = max(gs)
    heads = " & ".join(POLICIES[pk][0] for pk in order)
    rows = [
        r"\begin{tabular}{@{}l" + "c" * len(order) + r"@{}}",
        r"\toprule",
        r"Policy (trained on) & " + heads + r" \\",
        r"\midrule",
        r"\multicolumn{%d}{@{}l}{\emph{At $30$k iterations (PACE\,/\,LeGS)}} \\" % (len(order) + 1),
    ]
    def cells(fn):
        return " & ".join(fn(block[pk]) for pk in order)
    def n_cell(r):
        a, b = r["fin"]["agentic"], r["fin"]["baseline"]
        Na, Nb = round(a["N"] / 1000), round(b["N"] / 1000)
        return ("$\\mathbf{%d}/%d$" % (Na, Nb)) if Na < 0.8 * Nb else ("$%d/%d$" % (Na, Nb))
    rows.append(r"\quad Gaussians (k) & " + cells(n_cell) + r" \\")
    rows.append(r"\quad test PSNR (dB) & " + cells(lambda r: "$%.2f/%.2f$" % (r["fin"]["agentic"]["test_psnr"], r["fin"]["baseline"]["test_psnr"])) + r" \\")
    rows.append(r"\midrule")
    rows.append(r"\multicolumn{%d}{@{}l}{\emph{Speed-up vs.\ LeGS's native schedule}} \\" % (len(order) + 1))
    for tg, _ in zip(T, range(len(T))):
        vals = [block[pk]["tt"].get(tg, {}).get("speedup") for pk in order]
        if all(v is None for v in vals):
            continue
        rows.append((r"\quad %d\,dB & " % tg) + " & ".join("---" if v is None else "$%.2f\\times$" % v for v in vals) + r" \\")
    rows.append(r"\quad \textbf{geometric mean} & " + " & ".join(("$\\mathbf{%.2f\\times}$" % block[pk]["gmean"]) if abs(block[pk]["gmean"] - best) < 1e-9 else ("$%.2f\\times$" % block[pk]["gmean"]) for pk in order) + r" \\")
    rows += [r"\bottomrule", r"\end{tabular}"]
    with open(TABLE, "w", encoding="utf-8") as f:
        f.write("\n".join(rows) + "\n")
    print("wrote", TABLE)

    # ---------------------------------------------------------------- figure (single panel)
    INK, MUTED, GRID = "#222222", "#6b6b6b", "#e6e6e6"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
        "legend.fontsize": 8.2, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "axes.edgecolor": MUTED, "axes.linewidth": 0.8, "text.color": INK,
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    })
    fig, ax = plt.subplots(figsize=(2.75, 3.0))
    ax.axhspan(0.4, 1.0, color="#f2f2f2", zorder=0)
    ax.axhline(1.0, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    ymax = 1.0
    for pk in order:
        r = block[pk]
        lab, col = POLICIES[pk]
        xs = [t for t, y in zip(T, r["curve"]) if y is not None]
        ys = [y for y in r["curve"] if y is not None]
        ymax = max(ymax, max(ys))
        Nk = round(r["fin"]["agentic"]["N"] / 1000)
        ax.plot(xs, ys, "--", color=col, lw=2.0, marker="o", ms=4.8, markerfacecolor=col,
                markeredgecolor="white", markeredgewidth=0.6,
                label=u"%s policy · %.2f× · %dk" % (lab, r["gmean"], Nk), zorder=3)
    ax.set_xlabel("Target PSNR (dB)")
    ax.set_ylabel("Speed-up vs.\nLeGS's native schedule")
    ax.set_xlim(15.7, 21.3)
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.set_yscale("log")
    yt = [0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    ax.set_yticks(yt)
    ax.set_yticklabels(["%g" % v for v in yt])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(0.7, ymax * 1.7)
    ax.grid(True, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.annotate("LeGS native (1$\\times$)", xy=(15.8, 1.0), xytext=(15.8, 1.03), fontsize=6.8, color=MUTED)
    ax.legend(loc="upper right", frameon=False, handlelength=1.8, labelspacing=0.3, borderpad=0.2)
    fig.tight_layout(pad=0.3)
    fig.savefig(FIG, bbox_inches="tight")
    fig.savefig(FIG.replace(".pdf", ".png"), dpi=200, bbox_inches="tight")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
