"""Visualize the eval_protocol.py artifacts: PSNR/SSIM-vs-time, Gaussian trajectory,
and the agent's per-block action evolution. Run in env_pytorch_3dgs (matplotlib)."""
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--dir", required=True)
ap.add_argument("--scene", default="")
args = ap.parse_args()
D = Path(args.dir)

def load(p):
    return list(csv.DictReader(open(p))) if Path(p).exists() else []

curve = load(D / "curve.csv")
blk = {m: load(D / f"{m}_blocks.csv") for m in ["baseline", "agentic"]}

def col(rows, k, f=float):
    return [f(r[k]) for r in rows if r.get(k) not in (None, "", "None")]

fig, ax = plt.subplots(2, 3, figsize=(17, 9))
C = {"baseline": "#5b6b8c", "agentic": "#e0682f"}

# (0,0) PSNR vs time ; (0,1) SSIM vs time
for m in ["baseline", "agentic"]:
    cr = sorted([(float(r["t"]), float(r["test_psnr"]), float(r["test_ssim"])) for r in curve if r["method"] == m])
    if cr:
        ax[0, 0].plot([t for t, _, _ in cr], [p for _, p, _ in cr], "-o", color=C[m], label=m, lw=2)
        ax[0, 1].plot([t for t, _, _ in cr], [s for _, _, s in cr], "-o", color=C[m], label=m, lw=2)
ax[0, 0].set(xlabel="training wall-clock (s)", ylabel="test PSNR (dB)", title=f"Time-to-target PSNR {args.scene}")
ax[0, 1].set(xlabel="training wall-clock (s)", ylabel="test SSIM", title="SSIM vs time")

# (0,2) Gaussian count vs iteration
for m in ["baseline", "agentic"]:
    if blk[m]:
        ax[0, 2].plot(col(blk[m], "iter"), [g/1000 for g in col(blk[m], "gaussians")], color=C[m], label=m, lw=2)
ax[0, 2].set(xlabel="iteration", ylabel="Gaussians (k)", title="Gaussian count")

# (1,0) agent continuous LR / threshold multipliers
a = blk["agentic"]
if a:
    it = col(a, "iter")
    for k, lbl in [("position_lr_mult", "pos-lr"), ("opacity_lr_mult", "opac-lr"),
                   ("scaling_lr_mult", "scale-lr"), ("feature_lr_mult", "feat-lr"),
                   ("densify_threshold_mult", "densify-thr")]:
        ax[1, 0].plot(it, col(a, k), label=lbl, lw=1.8)
    ax[1, 0].axhline(1.0, color="gray", ls=":", lw=1)
    ax[1, 0].set(xlabel="iteration", ylabel="multiplier (x)", title="Agent continuous actions")
    ax[1, 0].legend(fontsize=8, ncol=2)

    # (1,1) agent discrete: densify_mode + prune_mode as numeric levels
    DENS = {"off": 0, "conservative": 1, "default": 2, "aggressive": 3}
    PRUN = {"off": 0, "opacity_only": 1, "opacity_and_size": 2}
    ax[1, 1].step(it, [DENS.get(r["densify_mode"], -1) for r in a], where="post", color="#2a8", label="densify_mode", lw=1.8)
    ax[1, 1].step(it, [PRUN.get(r["prune_mode"], -1) for r in a], where="post", color="#a52", label="prune_mode", lw=1.8)
    ax[1, 1].set(xlabel="iteration", ylabel="mode level", title="Agent discrete actions")
    ax[1, 1].set_yticks([0, 1, 2, 3]); ax[1, 1].legend(fontsize=8)

    # (1,2) added / pruned per block
    ax[1, 2].plot(it, col(a, "added"), color="#2a8", label="added", lw=1.5)
    ax[1, 2].plot(it, [-p for p in col(a, "pruned")], color="#a52", label="pruned", lw=1.5)
    ax[1, 2].axhline(0, color="gray", ls=":", lw=1)
    ax[1, 2].set(xlabel="iteration", ylabel="Gaussians +/- per block", title="Agent densify/prune events")
    ax[1, 2].legend(fontsize=8)

for a_ in ax.flat:
    a_.grid(alpha=.3)
    if a_.get_legend_handles_labels()[0] and not a_.get_legend():
        a_.legend(fontsize=8)
fig.tight_layout()
fig.savefig(D / "protocol_overview.png", dpi=120)
print("wrote", D / "protocol_overview.png")
