"""Analyze how the Gaussian population evolves under the agent vs. the fixed-schedule
baseline, from the .ply snapshots saved by save_gaussian_snapshots.py.

For each case <backend>_<mode> it produces, in outputs/gaussian_evolution/analysis/:
  <case>_metrics.pdf : count / median scale / mean opacity / spatial extent vs wall-clock
                       time, agent vs baseline (plus PSNR from snapshots.csv).
  <case>_cloud.png   : a grid of point-cloud projections (rows = snapshot times,
                       cols = agent | baseline), points sub-sampled, coloured by opacity.

The .ply is the standard 3DGS/Faster-GS format (binary little-endian): x,y,z, normals,
f_dc_*, f_rest_*, opacity (logit), scale_* (log), rot_*. We read only what we need.
A single PCA basis is fit on the shared initial cloud so agent/baseline/time frames are
spatially aligned and directly comparable.
"""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
EVO = ROOT / "outputs/gaussian_evolution"
OUT = EVO / "analysis"; OUT.mkdir(parents=True, exist_ok=True)

BACKENDS = ["3dgs", "fastergs"]
MODES = ["accel", "budget"]
CLOUD_TAGS = ["init", "t15s", "t60s", "t120s", "final"]   # rows shown in the cloud grid
KMAX = 40000                                              # max points scattered per panel
C_AGENT, C_BASE = "#3a5da8", "#e0682f"


def read_ply(path, want=("x", "y", "z", "opacity", "scale_0", "scale_1", "scale_2")):
    """Minimal binary-little-endian PLY reader -> dict of float32 arrays for `want`."""
    with open(path, "rb") as f:
        assert f.readline().strip() == b"ply"
        fmt = f.readline().strip()
        assert b"binary_little_endian" in fmt, fmt
        n, props = 0, []
        while True:
            ln = f.readline().strip()
            if ln.startswith(b"element vertex"):
                n = int(ln.split()[-1])
            elif ln.startswith(b"property"):
                props.append(ln.split()[-1].decode())
            elif ln == b"end_header":
                break
        dtype = np.dtype([(p, "<f4") for p in props])   # 3DGS ply: all float32
        data = np.fromfile(f, dtype=dtype, count=n)
    return {k: np.asarray(data[k], dtype=np.float32) for k in want if k in data.dtype.names}, n


def load_case(backend, mode):
    """Return {'agent': rows, 'baseline': rows} where rows come from snapshots.csv, or None."""
    out = {}
    for who in ("agent", "baseline"):
        d = EVO / f"{backend}_{mode}_{who}"
        csvp = d / "snapshots.csv"
        if not csvp.exists():
            return None
        rows = list(csv.DictReader(open(csvp)))
        for r in rows:
            r["_dir"] = d
        out[who] = rows
    return out


def stats_from_ply(path):
    d, n = read_ply(path)
    if n == 0:
        return dict(n=0, scale=np.nan, opacity=np.nan, extent=np.nan)
    opac = 1.0 / (1.0 + np.exp(-d["opacity"]))
    scale = np.exp(np.stack([d["scale_0"], d["scale_1"], d["scale_2"]], 1)).mean(1)
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    extent = float(np.linalg.norm(xyz.std(0)))
    return dict(n=n, scale=float(np.median(scale)), opacity=float(opac.mean()), extent=extent)


def pca_basis(path):
    d, n = read_ply(path, want=("x", "y", "z"))
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    if n > KMAX:
        idx = np.linspace(0, n - 1, KMAX).astype(int)
        xyz = xyz[idx]
    mu = xyz.mean(0)
    u, s, vt = np.linalg.svd(xyz - mu, full_matrices=False)
    return mu, vt[:2].T   # (3,), (3,2)


def project(path, mu, basis):
    d, n = read_ply(path, want=("x", "y", "z", "opacity"))
    xyz = np.stack([d["x"], d["y"], d["z"]], 1)
    opac = 1.0 / (1.0 + np.exp(-d["opacity"]))
    if n > KMAX:
        idx = np.linspace(0, n - 1, KMAX).astype(int)
        xyz, opac = xyz[idx], opac[idx]
    return (xyz - mu) @ basis, opac


# ---------------- per-case metrics + cloud figures ----------------
def process(backend, mode):
    case = load_case(backend, mode)
    if case is None:
        print(f"[skip] {backend}_{mode}: snapshots missing")
        return
    tag = f"{backend}_{mode}"

    # ---- metrics vs time ----
    series = {}
    for who, rows in case.items():
        t, N, sc, op, ex, ps = [], [], [], [], [], []
        for r in rows:
            if r["tag"] == "init":
                continue
            st = stats_from_ply(r["_dir"] / "ply" / r["ply"])
            t.append(float(r["time_s"])); N.append(st["n"]); sc.append(st["scale"])
            op.append(st["opacity"]); ex.append(st["extent"]); ps.append(float(r["psnr"]))
        series[who] = dict(t=t, N=N, sc=sc, op=op, ex=ex, ps=ps)

    fig, ax = plt.subplots(1, 4, figsize=(15, 3.4))
    panels = [("N", "Gaussian count", 1e-3, "count (k)"),
              ("sc", "median scale", 1.0, "median scale"),
              ("op", "mean opacity", 1.0, "mean opacity"),
              ("ps", "test PSNR (dB)", 1.0, "PSNR (dB)")]
    for a, (key, ttl, mul, ylab) in zip(ax, panels):
        for who, c in [("agent", C_AGENT), ("baseline", C_BASE)]:
            s = series[who]
            a.plot(s["t"], [v * mul for v in s[key]], "-o", color=c, lw=2, ms=4,
                   label=who, ls="-" if who == "agent" else "--")
        a.set_xlabel("wall-clock time (s)"); a.set_ylabel(ylab); a.set_title(ttl, fontsize=10)
        a.grid(alpha=.3)
    ax[0].legend(fontsize=9)
    fig.suptitle(f"{backend.upper()} - {mode} mode: Gaussian evolution (agent vs baseline)", fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / f"{tag}_metrics.pdf", bbox_inches="tight"); plt.close(fig)

    # ---- point-cloud projection grid ----
    init_ply = case["agent"][0]["_dir"] / "ply" / "snap_init.ply"
    mu, basis = pca_basis(init_ply)
    tags = [t for t in CLOUD_TAGS if any(r["tag"] == t for r in case["agent"])]
    # pre-project all panels, then use SHARED axis limits so agent/baseline footprints
    # (spatial compactness) are directly comparable rather than autoscaled per panel.
    proj = {}
    allx, ally = [], []
    for tg in tags:
        for who in ("agent", "baseline"):
            row = next((r for r in case[who] if r["tag"] == tg), None)
            if row is None:
                continue
            xy, opac = project(row["_dir"] / "ply" / row["ply"], mu, basis)
            proj[(tg, who)] = (xy, opac, row)
            allx.append(xy[:, 0]); ally.append(xy[:, 1])
    ax_all = np.concatenate(allx); ay_all = np.concatenate(ally)
    xlo, xhi = np.percentile(ax_all, [0.5, 99.5]); ylo, yhi = np.percentile(ay_all, [0.5, 99.5])
    pad = 0.04 * max(xhi - xlo, yhi - ylo)
    xlim = (xlo - pad, xhi + pad); ylim = (ylo - pad, yhi + pad)

    fig, axs = plt.subplots(len(tags), 2, figsize=(8.2, 2.6 * len(tags)), squeeze=False)
    sc = None
    for i, tg in enumerate(tags):
        for j, who in enumerate(("agent", "baseline")):
            a = axs[i][j]
            if (tg, who) not in proj:
                a.axis("off"); continue
            xy, opac, row = proj[(tg, who)]
            order = np.argsort(opac)   # draw opaque points last
            sc = a.scatter(xy[order, 0], xy[order, 1], c=opac[order], s=1.4, cmap="viridis",
                           vmin=0, vmax=1, linewidths=0, rasterized=True)
            a.set_title(f"{who} | {tg} | N={int(row['gaussians'])//1000}k | {row['psnr']} dB", fontsize=8.5)
            a.set_xlim(*xlim); a.set_ylim(*ylim)
            a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    fig.suptitle(f"{backend.upper()} - {mode}: point-cloud evolution (shared scale; colour = opacity)", fontsize=11)
    fig.tight_layout(rect=[0, 0.03, 1, 0.99])
    if sc is not None:
        cb = fig.colorbar(sc, ax=axs, fraction=0.025, pad=0.02); cb.set_label("opacity", fontsize=9)
    fig.savefig(OUT / f"{tag}_cloud.png", dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[ok] {backend}_{mode}: wrote {backend}_{mode}_metrics.pdf + {backend}_{mode}_cloud.png")


def main():
    for b in BACKENDS:
        for m in MODES:
            process(b, m)
    print("analysis ->", OUT)


if __name__ == "__main__":
    main()
