"""Render a fixed test view from each Gaussian snapshot .ply, to pair a visual
reconstruction with the point-cloud evolution figure. No retraining: it loads the saved
Gaussian state and renders it.

For each --dir (a save_gaussian_snapshots.py output folder) it writes render/<ply>.png for
every snapshot in snapshots.csv, plus render/gt.png (the held-out ground-truth view), all
from the same test camera. Run in the conda env matching the backend.
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from agentic_gs_phase1.envs import AgenticGSEnv


def to_png(t, path, size=None):
    arr = (t.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    if size:
        w, h = im.size
        s = size / max(w, h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
    im.save(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--dirs", nargs="+", required=True, help="snapshot output dirs to render")
    ap.add_argument("--view-index", type=int, default=0)
    ap.add_argument("--size", type=int, default=512, help="longest-side px for saved PNGs")
    args = ap.parse_args()

    dev = torch.device("cuda")
    import json
    cfg = json.loads(Path(args.config).read_text())
    cfg["fastergs_acknowledge_grad_bug"] = True
    cfg["save_episode_models"] = False

    env = AgenticGSEnv(cfg, run_dir=ROOT / "outputs/gaussian_evolution/_render_run", seed=0)
    env.reset(args.scene, episode_id=0)
    cams = list(env.scene.getTestCameras())
    cam = cams[args.view_index % len(cams)]
    sh_deg, opt_type = env.dataset.sh_degree, env.opt.optimizer_type

    for d in args.dirs:
        d = Path(d)
        rd = d / "render"; rd.mkdir(exist_ok=True)
        to_png(cam.original_image.to(dev), rd / "gt.png", args.size)
        for r in csv.DictReader(open(d / "snapshots.csv")):
            ply = d / "ply" / r["ply"]
            g = env.backend.make_gaussians(sh_deg, opt_type)
            g.load_ply(str(ply))
            with torch.no_grad():
                img = env.backend.render_image(cam, g, env.pipe, env.background, use_trained_exp=False)
            to_png(img, rd / (Path(r["ply"]).stem + ".png"), args.size)
            del g
            torch.cuda.empty_cache()
            print(f"[{d.name}] {r['tag']:>6}  N={int(r['gaussians'])//1000}k  psnr={r['psnr']}", flush=True)
    env.close()
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
