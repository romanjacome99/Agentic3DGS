"""Visual reconstruction quality aligned with wall-clock time, three backends.

Uses the wall-clock snapshot renders of the exact Table-1 checkpoints
(outputs/gaussian_evolution/<run>/render, snapshots.csv) on held-out `train`,
plus each backend's full-resolution test-camera protocol curve for the
PSNR-vs-time insets. Writes Paper/figures/visual_time.pdf.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVO = PROJECT_ROOT / "outputs" / "gaussian_evolution"
PROT = PROJECT_ROOT / "outputs" / "agentic_rl_real"
OUT_PDF = PROJECT_ROOT / "Paper" / "figures" / "visual_time.pdf"

BACKENDS = [
    ("(a) 3DGS", "3dgs_accel_agent", "3dgs_accel_baseline", "protocol_nonaug_3dgs_train_fr"),
    ("(b) Faster-GS", "fastergs_base_accel_agent", "fastergs_base_accel_baseline", "protocol_aug_base_fastergs_train"),
    ("(c) DashGaussian", "dash_accel_agent", "dash_accel_baseline", "protocol_nonaug_dash_train"),
]
TIMES = ["t15s", "t30s", "t60s", "t120s"]
TIME_LABELS = ["t = 15 s", "t = 30 s", "t = 60 s", "t = 120 s"]
AGENT_C = "#0072B2"
BASE_C = "#5a5a5a"

plt.rcParams.update({
    "font.size": 7,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})


def load_snapshots(run: str) -> dict[str, dict]:
    rows = {}
    with (EVO / run / "snapshots.csv").open() as f:
        for row in csv.DictReader(f):
            rows[row["tag"]] = row
    return rows


def annotate(ax, text: str) -> None:
    ax.text(0.02, 0.045, text, transform=ax.transAxes, fontsize=6.0,
            color="white", ha="left", va="bottom",
            bbox=dict(facecolor="black", alpha=0.55, pad=1.2, edgecolor="none"))


def fmt_n(n: int) -> str:
    return f"{n/1e6:.1f}M" if n >= 1e6 else f"{n//1000}k"


def draw_inset(ax, curve_dir: str, snaps: dict) -> None:
    bb = ax.get_position()
    ax.set_position([bb.x0 + 0.027, bb.y0 + 0.016, bb.width - 0.029, bb.height - 0.028])
    curve = pd.read_csv(PROT / curve_dir / "curve.csv")
    for method, color, ls in [("agentic", AGENT_C, "-"), ("baseline", BASE_C, (0, (3, 1.6)))]:
        c = curve[(curve["method"] == method) & (curve["t"] <= 135)]
        ax.plot(c["t"], c["test_psnr"], color=color, ls=ls, lw=1.1)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 60, 120])
    ax.set_xticklabels(["0", "60", "120 s"], fontsize=5.5)
    ax.set_ylim(11.5, 21.5)
    ax.set_yticks([12, 16, 20])
    ax.tick_params(labelsize=5.5, length=2, pad=1)
    ax.text(0.05, 0.97, "PSNR (dB)", transform=ax.transAxes, fontsize=5.2,
            va="top", ha="left", color="#444444")
    ax.spines[["top", "right"]].set_visible(False)


def main() -> None:
    fig = plt.figure(figsize=(5.5, 4.55))
    # 8 rows: (img, img, spacer) x 3 blocks, no trailing spacer
    gs = fig.add_gridspec(8, 5, left=0.035, right=0.995, top=0.945, bottom=0.035,
                          wspace=0.03, hspace=0.07,
                          height_ratios=[1, 1, 0.34, 1, 1, 0.34, 1, 1])
    block_rows = [0, 3, 6]

    for bi, (name, agent_run, base_run, curve_dir) in enumerate(BACKENDS):
        r0 = block_rows[bi]
        snaps = {"agent": load_snapshots(agent_run), "base": load_snapshots(base_run)}
        render = {"agent": EVO / agent_run / "render", "base": EVO / base_run / "render"}
        col0_axes = []
        for row, method in enumerate(["agent", "base"]):
            for col, (tag, lbl) in enumerate(zip(TIMES, TIME_LABELS)):
                ax = fig.add_subplot(gs[r0 + row, col])
                if col == 0:
                    col0_axes.append(ax)
                # a run that hits its iteration cap just before a trigger only
                # writes `final` (e.g. at 119.5 s instead of 120 s)
                use = tag if tag in snaps[method] else "final"
                ax.imshow(Image.open(render[method] / f"snap_{use}.png"))
                ax.set_axis_off()
                s = snaps[method][use]
                annotate(ax, f"{float(s['psnr']):.1f} dB · {fmt_n(int(s['gaussians']))}")
                if bi == 0 and row == 0:
                    ax.set_title(lbl, fontsize=7, pad=2)
        # block title + row labels via axes positions
        top_bb = col0_axes[0].get_position()
        fig.text(top_bb.x0, top_bb.y1 + (0.028 if bi == 0 else 0.006), name,
                 fontsize=7, fontweight="bold", ha="left", va="bottom")
        for row, method in enumerate(["agent", "base"]):
            bbx = col0_axes[row].get_position()
            fig.text(0.028, (bbx.y0 + bbx.y1) / 2,
                     "Agent" if method == "agent" else "Fixed",
                     rotation=90, va="center", ha="center", fontsize=6.5,
                     color=AGENT_C if method == "agent" else BASE_C, fontweight="bold")
        # right column
        if bi == 0:
            ax_gt = fig.add_subplot(gs[r0, 4])
            ax_gt.imshow(Image.open(render["agent"] / "gt.png"))
            ax_gt.set_axis_off()
            ax_gt.set_title("ground truth", fontsize=7, pad=2)
            ax_c = fig.add_subplot(gs[r0 + 1, 4])
        else:
            ax_c = fig.add_subplot(gs[r0:r0 + 2, 4])
        draw_inset(ax_c, curve_dir, snaps)

    fig.savefig(OUT_PDF, dpi=300)
    print(f"wrote {OUT_PDF}")

    for name, agent_run, base_run, _ in BACKENDS:
        sa, sb = load_snapshots(agent_run), load_snapshots(base_run)
        a = sa.get("t120s", sa["final"])
        b = sb.get("t120s", sb["final"])
        print(f"{name}: @120s agent {float(a['psnr']):.2f} dB / {fmt_n(int(a['gaussians']))} "
              f"vs base {float(b['psnr']):.2f} dB / {fmt_n(int(b['gaussians']))}")


if __name__ == "__main__":
    main()
