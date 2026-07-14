"""Build the paper figure 'what the agent does to the Gaussians' for the 3DGS
acceleration case, from the snapshots saved by save_gaussian_snapshots.py.

Layout (landscape, width = \\linewidth):
  rows 0-1 : 2x4 point-cloud grid -- agent (top) vs baseline (bottom), columns = wall-clock
             times, points sub-sampled, coloured by opacity, shared spatial scale.
  row 2    : count / median scale / mean opacity / PSNR vs time (agent vs baseline).

Emits Paper/figures/gaussian_evolution.pdf.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
EVO = ROOT / "outputs/gaussian_evolution"
FIG = ROOT / "Paper/figures"; FIG.mkdir(parents=True, exist_ok=True)

CASE = "3dgs_accel"
CLOUD_TIMES = ["init", "t15s", "t60s", "t120s"]
COL_LABEL = {"init": "init (0 s)", "t15s": "15 s", "t60s": "60 s", "t120s": "120 s"}
KMAX = 45000
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


def pca_basis(path):
    d, n = read_ply(path, want=("x", "y", "z"))
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    if n > KMAX:
        xyz = xyz[np.linspace(0, n - 1, KMAX).astype(int)]
    mu = xyz.mean(0)
    _, _, vt = np.linalg.svd(xyz - mu, full_matrices=False)
    return mu, vt[:2].T


def project(path, mu, basis):
    d, n = read_ply(path, want=("x", "y", "z", "opacity"))
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    op = 1.0 / (1.0 + np.exp(-d["opacity"]))
    if n > KMAX:
        idx = np.linspace(0, n - 1, KMAX).astype(int)
        xyz, op = xyz[idx], op[idx]
    return (xyz - mu) @ basis, op


def stats(path):
    d, n = read_ply(path)
    op = 1.0 / (1.0 + np.exp(-d["opacity"]))
    sc = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)).mean(1)
    return n, float(np.median(sc)), float(op.mean())


agent, base = rows("agent"), rows("baseline")
adir, bdir = EVO / f"{CASE}_agent", EVO / f"{CASE}_baseline"
mu, basis = pca_basis(adir / "ply" / "snap_init.ply")

# pre-project cloud panels + shared limits
proj, xs, ys = {}, [], []
for who, d in (("agent", adir), ("baseline", bdir)):
    src = agent if who == "agent" else base
    for tg in CLOUD_TIMES:
        r = next((x for x in src if x["tag"] == tg), None)
        if r is None:
            continue
        xy, op = project(d / "ply" / r["ply"], mu, basis)
        proj[(who, tg)] = (xy, op, r)
        xs.append(xy[:, 0]); ys.append(xy[:, 1])
xa, ya = np.concatenate(xs), np.concatenate(ys)
xlo, xhi = np.percentile(xa, [0.5, 99.5]); ylo, yhi = np.percentile(ya, [0.5, 99.5])
pad = 0.04 * max(xhi - xlo, yhi - ylo)
xlim, ylim = (xlo - pad, xhi + pad), (ylo - pad, yhi + pad)

fig = plt.figure(figsize=(7.6, 6.0))
outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.05], hspace=0.28)
cloud = GridSpecFromSubplotSpec(2, 4, subplot_spec=outer[0], hspace=0.16, wspace=0.06)
sc = None
for i, who in enumerate(("agent", "baseline")):
    for j, tg in enumerate(CLOUD_TIMES):
        a = fig.add_subplot(cloud[i, j])
        if (who, tg) in proj:
            xy, op, r = proj[(who, tg)]
            order = np.argsort(op)
            sc = a.scatter(xy[order, 0], xy[order, 1], c=op[order], s=1.1, cmap="viridis",
                           vmin=0, vmax=1, linewidths=0, rasterized=True)
            a.text(0.5, 1.02, f"$N$={int(r['gaussians'])//1000}k, {r['psnr']} dB",
                   transform=a.transAxes, ha="center", va="bottom", fontsize=7)
        a.set_xlim(*xlim); a.set_ylim(*ylim); a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
        if i == 0:
            a.set_title(COL_LABEL[tg], fontsize=9, pad=12)
        if j == 0:
            a.set_ylabel({"agent": "Agent", "baseline": "Baseline"}[who], fontsize=10)
if sc is not None:
    cb = fig.colorbar(sc, ax=fig.axes, fraction=0.012, pad=0.01)
    cb.set_label("opacity", fontsize=8); cb.ax.tick_params(labelsize=7)

# metrics row
met = GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[1], wspace=0.42)
series = {}
for who, d in (("agent", adir), ("baseline", bdir)):
    src = agent if who == "agent" else base
    t, N, S, O, P = [], [], [], [], []
    for r in src:
        if r["tag"] == "init":
            continue
        n, s, o = stats(d / "ply" / r["ply"])
        t.append(float(r["time_s"])); N.append(n / 1e3); S.append(s); O.append(o); P.append(float(r["psnr"]))
    series[who] = (t, N, S, O, P)
titles = [("count (k)", 1), ("median scale", 2), ("mean opacity", 3), ("PSNR (dB)", 4)]
for k, (ylab, idx) in enumerate(titles):
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
