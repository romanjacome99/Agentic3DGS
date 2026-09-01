"""Compare the learned acceleration-mode action sets across backends.

Reads the per-block action logs of the exact single-seed rollouts behind the
time-to-target table (Paper Table 4):
  - 3DGS   (non-augmented):  outputs/agentic_rl_real/protocol_nonaug_3dgs_train_fr
  - FasterGS (photometric-augmented): outputs/agentic_rl_real/protocol_aug_fastergs_train
  - Dash   (non-augmented):  outputs/agentic_rl_real/protocol_nonaug_dash_train

Writes Paper/figures/action_sets_backends.pdf:
  (a) when densification / pruning / resets fire (per-block timeline strips)
  (b) the resulting Gaussian count (agent vs. fixed schedule)
  (c) the seven continuous multipliers per iteration (relative to each
      backend's default), one mini-panel per multiplier
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUT_PDF = PROJECT_ROOT / "Paper" / "figures" / "action_sets_backends.pdf"

BACKENDS = [
    ("3DGS", "#0072B2", "protocol_nonaug_3dgs_train_fr"),
    ("FasterGS", "#E69F00", "protocol_aug_base_fastergs_train"),
    ("Dash", "#009E73", "protocol_nonaug_dash_train"),
]
OFF_COLOR = "#d8d8d8"

CONT_COLS = [
    ("densify_threshold_mult", r"densify $\nabla$-thr."),
    ("prune_opacity_threshold", "prune thr."),
    ("position_lr_mult", "position lr"),
    ("feature_lr_mult", "SH lr"),
    ("opacity_lr_mult", "opacity lr"),
    ("scaling_lr_mult", "scale lr"),
    ("rotation_lr_mult", "rotation lr"),
]

plt.rcParams.update({
    "font.size": 7,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})


def load(run: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = PROJECT_ROOT / "outputs" / "agentic_rl_real" / run
    agent = pd.read_csv(root / "agentic_blocks.csv")
    base = pd.read_csv(root / "baseline_blocks.csv")
    return agent, base


def main() -> None:
    data = {name: load(run) for name, _, run in BACKENDS}

    fig = plt.figure(figsize=(5.5, 3.5))
    gs = fig.add_gridspec(2, 7, height_ratios=[1.0, 1.0],
                          left=0.105, right=0.985, top=0.915, bottom=0.075,
                          hspace=0.75, wspace=0.22)
    ax_a = fig.add_subplot(gs[0, :4])
    ax_b = fig.add_subplot(gs[0, 4:])
    axs_c = [fig.add_subplot(gs[1, i]) for i in range(7)]

    # ---------------- (a) when: densify / prune / reset timeline ----------
    for row, (name, color, _) in enumerate(BACKENDS):
        agent, _ = data[name]
        y = len(BACKENDS) - 1 - row
        start = (agent["iter"] - agent["block_steps"]).to_numpy()
        width = agent["block_steps"].to_numpy()
        on = agent["densify_mode"].ne("off").to_numpy()
        ax_a.broken_barh(list(zip(start[~on], width[~on])), (y - 0.28, 0.56),
                         color=OFF_COLOR, linewidth=0)
        if on.any():
            ax_a.broken_barh(list(zip(start[on], width[on])), (y - 0.28, 0.56),
                             color=color, linewidth=0)
        resets = agent["opacity_reset"].eq("force_reset")
        ax_a.plot(agent.loc[resets, "iter"], np.full(resets.sum(), y + 0.40),
                  marker="v", ms=2.6, ls="none", color="#444444", clip_on=False)
        size_prune = agent["prune_mode"].eq("opacity_and_size")
        ax_a.plot(agent.loc[size_prune, "iter"], np.full(size_prune.sum(), y - 0.40),
                  marker="|", ms=3.2, ls="none", color="#444444", clip_on=False)
        on_frac = on.mean()
        if on_frac > 0:
            interval = int(agent.loc[on, "densification_interval"].median())
            ax_a.text(29500, y + 0.42, f"on {on_frac:.0%} of blocks, interval {interval}",
                      va="bottom", ha="right", fontsize=5.6, color="#333333")
        else:
            ax_a.text(15000, y, "densify off (all blocks)", va="center",
                      ha="center", fontsize=5.6, color="#333333")
    ax_a.set_yticks(range(len(BACKENDS)))
    ax_a.set_yticklabels([b[0] for b in reversed(BACKENDS)])
    for tick, (_, color, _) in zip(ax_a.get_yticklabels(), reversed(BACKENDS)):
        tick.set_color(color)
        tick.set_fontweight("bold")
    ax_a.set_xlim(0, 30000)
    ax_a.set_ylim(-0.75, len(BACKENDS) - 0.25)
    ax_a.set_xticks([0, 10000, 20000, 30000])
    ax_a.set_xticklabels(["0", "10k", "20k", "30k"])
    ax_a.set_title("(a) densification schedule", loc="left")
    ax_a.text(1.0, 1.005, r"$\blacktriangledown$ forced reset   $|$ size-prune",
              transform=ax_a.transAxes, fontsize=5.6, color="#555555",
              ha="right", va="bottom")
    ax_a.spines[["top", "right", "left"]].set_visible(False)
    ax_a.tick_params(axis="y", length=0)

    # ---------------- (b) result: Gaussian count --------------------------
    for name, color, _ in BACKENDS:
        agent, base = data[name]
        ax_b.plot(base["iter"], base["gaussians"] / 1e6, color=color, lw=0.9,
                  ls=(0, (3, 1.6)), alpha=0.55)
        ax_b.plot(agent["iter"], agent["gaussians"] / 1e6, color=color, lw=1.4)
    ax_b.set_yscale("log")
    yticks = [0.2, 0.5, 1, 2]
    ax_b.set_yticks(yticks)
    ax_b.set_yticklabels([f"{v:g}M" for v in yticks])
    ax_b.set_xlim(0, 30000)
    ax_b.set_xticks([0, 10000, 20000, 30000])
    ax_b.set_xticklabels(["0", "10k", "20k", "30k"])
    ax_b.set_title("(b) Gaussian count", loc="left")
    ax_b.spines[["top", "right"]].set_visible(False)
    ax_b.legend(handles=[
        Line2D([], [], color="#555555", lw=1.4, label="agent"),
        Line2D([], [], color="#555555", lw=0.9, ls=(0, (3, 1.6)), alpha=0.6,
               label="baseline"),
    ], loc="center right", bbox_to_anchor=(1.0, 0.38), frameon=False,
       fontsize=5.8, handlelength=1.6, borderaxespad=0.1, labelspacing=0.25)

    # ---------------- (c) continuous multipliers per iteration ------------
    for gx, (ax, (col, lbl)) in enumerate(zip(axs_c, CONT_COLS)):
        ax.axhline(1.0, color="#aaaaaa", lw=0.6, ls=(0, (3, 2)), zorder=0)
        for name, color, _ in BACKENDS:
            agent, base = data[name]
            vals = agent[col].to_numpy(dtype=float)
            if col == "prune_opacity_threshold":
                vals = vals / float(base[col].iloc[0])  # absolute -> multiplier
            x = (agent["iter"] - agent["block_steps"]).to_numpy(dtype=float)
            ax.plot(np.append(x, agent["iter"].iloc[-1]), np.append(vals, vals[-1]),
                    drawstyle="steps-post", color=color, lw=0.9)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.22, 4.5)
        ax.set_xlim(0, 30000)
        ax.set_xticks([0, 30000])
        labels = ax.set_xticklabels(["0", "30k"], fontsize=6)
        labels[0].set_horizontalalignment("left")
        labels[1].set_horizontalalignment("right")
        ax.set_title(lbl, fontsize=6.2, pad=2)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        if gx == 0:
            ax.set_yticklabels([r"$\frac{1}{4}\times$", r"$\frac{1}{2}\times$",
                                r"$1\times$", r"$2\times$", r"$4\times$"])
        else:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    fig.text(0.105, 0.475, r"(c) continuous multipliers $\theta_k$ per iteration"
             " (relative to backend default)", fontsize=7.5, ha="left",
             va="bottom")

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}")

    for name, _, _ in BACKENDS:
        agent, base = data[name]
        on = agent["densify_mode"].ne("off")
        print(f"{name}: {len(agent)} blocks, densify-on {on.mean():.0%}, "
              f"agent final {agent['gaussians'].iloc[-1]/1e3:.0f}k vs "
              f"baseline {base['gaussians'].iloc[-1]/1e3:.0f}k")


if __name__ == "__main__":
    main()
