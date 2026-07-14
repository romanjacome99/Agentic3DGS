"""Build the paper figure 'what the agent does to the Gaussians' for the 3DGS
acceleration case, from the snapshots (save_gaussian_snapshots.py) and their rendered
test views (render_snapshots.py).

Layout (width = \\linewidth):
  4x5 image block -- rows = {Agent recon, Agent points, Baseline recon, Baseline points},
                     columns = wall-clock times, plus a final GT column on the recon rows.
                     'recon' rows imshow the rendered held-out view; 'points' rows show the
                     RAW Gaussian centres (x,y,z) in a shared 3D world frame, coloured by
                     opacity. Every panel is annotated with count N and test PSNR.
  metrics row     -- count / median scale / mean opacity / PSNR vs time (agent vs baseline).

Emits Paper/figures/gaussian_evolution.pdf.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.gridspec import GridSpecFromSubplotSpec
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
EVO = ROOT / "outputs/gaussian_evolution"
FIG = ROOT / "Paper/figures"; FIG.mkdir(parents=True, exist_ok=True)

CASE = "3dgs_accel"
TIMES = ["init", "t15s", "t60s", "t120s"]
COL_LABEL = {"init": "init (0 s)", "t15s": "15 s", "t60s": "60 s", "t120s": "120 s"}
KMAX = 30000
VIEW_ELEV, VIEW_AZIM = 18, -72
PCTL = (1.0, 99.0)
C_AGENT, C_BASE = "#3a5da8", "#e0682f"


def read_ply(path, want=("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")):
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        assert b"binary_little_endian" in f.readline()
        n, props = 0, []
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith(b"property"):
                props.append(ln.split()[-1].decode())
            elif ln == b"end_header":
                break
        data = np.fromfile(f, dtype=np.dtype([(p, "<f4") for p in props]), count=n)
    return {k: np.asarray(data[k], np.float32) for k in want if k in data.dtype.names}, n


def rows(who):
    return list(csv.DictReader(open(EVO / f"{CASE}_{who}" / "snapshots.csv")))


def load_xyz_op(path):
    d, n = read_ply(path, want=("x", "y", "z", "opacity"))
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    op = 1.0 / (1.0 + np.exp(-d["opacity"]))
    if n > KMAX:
        idx = np.linspace(0, n - 1, KMAX).astype(int)
        xyz, op = xyz[idx], op[idx]
    return xyz, op


def stats(path):
    d, n = read_ply(path)
    op = 1.0 / (1.0 + np.exp(-d["opacity"]))
    sc = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)).mean(1)
    return n, float(np.median(sc)), float(op.mean())


agent, base = rows("agent"), rows("baseline")
adir, bdir = EVO / f"{CASE}_agent", EVO / f"{CASE}_baseline"
SRC = {"agent": (agent, adir), "baseline": (base, bdir)}


def rec(who, tg):
    src, d = SRC[who]
    r = next((x for x in src if x["tag"] == tg), None)
    return r, d


# preload point clouds + shared world box
cloud, allxyz = {}, []
for who in ("agent", "baseline"):
    for tg in TIMES:
        r, d = rec(who, tg)
        if r is None:
            continue
        xyz, op = load_xyz_op(d / "ply" / r["ply"])
        cloud[(who, tg)] = (xyz, op, r)
        allxyz.append(xyz)
allxyz = np.concatenate(allxyz, 0)
lo = np.percentile(allxyz, PCTL[0], axis=0) - 0.03 * np.ptp(allxyz, axis=0)
hi = np.percentile(allxyz, PCTL[1], axis=0) + 0.03 * np.ptp(allxyz, axis=0)
box_aspect = tuple(hi - lo)


def in_box(xyz, op):
    m = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    return xyz[m], op[m]


fig = plt.figure(figsize=(8.4, 8.6))
outer = fig.add_gridspec(2, 1, height_ratios=[3.4, 0.8], hspace=0.14)
blk = GridSpecFromSubplotSpec(4, 5, subplot_spec=outer[0], hspace=0.06, wspace=0.03)
ROWDEF = [("agent", "recon"), ("agent", "points"), ("baseline", "recon"),
          ("baseline", "points")]
ROWLAB = {("agent", "recon"): "Agent\nreconstruction", ("agent", "points"): "Agent\npoints",
          ("baseline", "recon"): "Baseline\nreconstruction", ("baseline", "points"): "Baseline\npoints"}
sc = None
for i, (who, kind) in enumerate(ROWDEF):
    for j, tg in enumerate(TIMES):
        r, d = rec(who, tg)
        if kind == "recon":
            a = fig.add_subplot(blk[i, j])
            png = d / "render" / (Path(r["ply"]).stem + ".png")
            a.imshow(mpimg.imread(png)); a.set_xticks([]); a.set_yticks([])
            for s in a.spines.values():
                s.set_linewidth(0.4)
            a.text(0.5, 1.02, f"$N$={int(r['gaussians'])//1000}k, {r['psnr']} dB",
                   transform=a.transAxes, ha="center", va="bottom", fontsize=6.5)
        else:
            a = fig.add_subplot(blk[i, j], projection="3d")
            xyz, op, r2 = cloud[(who, tg)]
            xyz, op = in_box(xyz, op)
            order = np.argsort(op)
            sc = a.scatter(xyz[order, 0], xyz[order, 1], xyz[order, 2], c=op[order], s=0.7,
                           cmap="viridis", vmin=0, vmax=1, linewidths=0, depthshade=False, rasterized=True)
            a.set_xlim(lo[0], hi[0]); a.set_ylim(lo[1], hi[1]); a.set_zlim(lo[2], hi[2])
            a.set_box_aspect(box_aspect); a.view_init(elev=VIEW_ELEV, azim=VIEW_AZIM)
            a.set_axis_off()
        if i == 0:
            a.set_title(COL_LABEL[tg], fontsize=9, fontweight="bold", y=1.10)
        if j == 0:
            a.set_ylabel(ROWLAB[(who, kind)], fontsize=8.5)
            if kind == "points":  # 3d axes ignore ylabel when axis off -> use text
                a.text2D(-0.06, 0.5, ROWLAB[(who, kind)], transform=a.transAxes, rotation=90,
                         ha="center", va="center", fontsize=8.5)

    # GT column (col 4) on the recon rows only
    aG = fig.add_subplot(blk[i, 4])
    if kind == "recon":
        aG.imshow(mpimg.imread(d / "render" / "gt.png")); aG.set_xticks([]); aG.set_yticks([])
        for s in aG.spines.values():
            s.set_linewidth(0.4)
        if i == 0:
            aG.set_title("GT (target)", fontsize=9, fontweight="bold", y=1.10)
    else:
        aG.set_axis_off()
if sc is not None:
    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.008, pad=0.01)
    cb.set_label("opacity", fontsize=8); cb.ax.tick_params(labelsize=7)

# metrics row
met = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.42)
series = {}
for who in ("agent", "baseline"):
    src, d = SRC[who]
    t, N, S, O, P = [], [], [], [], []
    for r in src:
        if r["tag"] == "init":
            continue
        n, s, o = stats(d / "ply" / r["ply"])
        t.append(float(r["time_s"])); N.append(n / 1e3); S.append(s); O.append(o); P.append(float(r["psnr"]))
    series[who] = (t, N, S, O, P)
for k, (ylab, idx) in enumerate([("count (k)", 1), ("median scale", 2), ("mean opacity", 3), ("PSNR (dB)", 4)]):
    a = fig.add_subplot(met[0, k])
    for who, c in (("agent", C_AGENT), ("baseline", C_BASE)):
        s = series[who]
        a.plot(s[0], s[idx], "-o" if who == "agent" else "--s", color=c, lw=1.8, ms=3.5, label=who)
    a.set_xlabel("time (s)", fontsize=8); a.set_ylabel(ylab, fontsize=8)
    a.tick_params(labelsize=7); a.grid(alpha=.3)
    if k == 0:
        a.legend(fontsize=7, loc="upper left")
fig.savefig(FIG / "gaussian_evolution.pdf", bbox_inches="tight", dpi=150)
print("wrote", FIG / "gaussian_evolution.pdf")
