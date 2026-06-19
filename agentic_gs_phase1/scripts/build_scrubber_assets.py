"""Curate frames + timing for the interactive per-block reconstruction scrubber.

For each scene: pick a shared iteration timeline, find the nearest baseline and
agentic frames, convert them to small JPEGs under docs/anim_frames/<scene>/, and
estimate the training wall-clock time to reach each iteration from the measured
force-full curves. Writes docs/_anim_manifest.json for the page to embed.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from PIL import Image

ROOT = Path(r"C:\Roman\3DGS_PROPOSAL")
ANIM = ROOT / "outputs" / "agentic_gs_phase1_reports" / "training_anim"
FFREP = ROOT / "outputs" / "agentic_gs_phase1_reports"
DOCS = ROOT / "agentic_gs_phase1" / "docs"
OUT_IMG = DOCS / "anim_frames"
TIMELINE = [0, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2500, 3000, 4000, 5000, 6000, 7000]
# Real scenes train to 30k force-full, so they get a longer timeline.
TIMELINES = {
    "truck": [0, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000, 15000, 20000, 30000],
}
SIZE = 260
SCENES = ["hotdog", "drums", "materials", "truck"]


def timeline_for(scene):
    return TIMELINES.get(scene, TIMELINE)


def nearest(frames, t):
    return min(frames, key=lambda f: abs(f["iter"] - t))


def time_curve(scene, method):
    """(iter -> seconds) anchor points from the force-full sampled curve."""
    f = FFREP / f"{scene}_time_to_target_v8_forcefull" / "sampled_test_curve.csv"
    pts = [(int(r["iteration"]), float(r["train_wall_seconds"]))
           for r in csv.DictReader(open(f)) if r["method"] == ("agentic" if method == "agentic" else "baseline")]
    pts = sorted(set(pts))
    return [(0, 0.0)] + pts


def interp(curve, it):
    if it <= curve[0][0]:
        return curve[0][1]
    for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
        if it <= x1:
            return y0 + (y1 - y0) * (it - x0) / max(1, x1 - x0)
    return curve[-1][1]


def save_jpg(src, dst):
    im = Image.open(src).convert("RGB").resize((SIZE, SIZE), Image.LANCZOS)
    im.save(dst, "JPEG", quality=82)


def main():
    manifest = {}
    for scene in SCENES:
        bman = json.loads((ANIM / f"{scene}_baseline" / "manifest.json").read_text())
        aman = json.loads((ANIM / f"{scene}_agentic" / "manifest.json").read_text())
        bcurve, acurve = time_curve(scene, "baseline"), time_curve(scene, "agentic")
        sdir = OUT_IMG / scene
        sdir.mkdir(parents=True, exist_ok=True)
        save_jpg(ANIM / f"{scene}_baseline" / "gt.png", sdir / "gt.jpg")
        frames = []
        for t in timeline_for(scene):
            if t > min(bman["frames"][-1]["iter"], aman["frames"][-1]["iter"]):
                break
            bf, af = nearest(bman["frames"], t), nearest(aman["frames"], t)
            bn, an = f"b_{t:05d}.jpg", f"a_{t:05d}.jpg"
            save_jpg(ANIM / f"{scene}_baseline" / "frames" / bf["file"], sdir / bn)
            save_jpg(ANIM / f"{scene}_agentic" / "frames" / af["file"], sdir / an)
            entry = {
                "it": t,
                "b": {"img": bn, "psnr": bf["psnr"], "N": bf["count"], "t": round(interp(bcurve, bf["iter"]), 1)},
                "a": {"img": an, "psnr": af["psnr"], "N": af["count"], "t": round(interp(acurve, af["iter"]), 1)},
            }
            if af.get("act"):
                entry["act"] = af["act"]          # discrete + continuous action the agent took
            frames.append(entry)
        manifest[scene] = {"gt": "gt.jpg", "frames": frames}
        print(f"{scene}: {len(frames)} timeline frames")
    (DOCS / "_anim_manifest.json").write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
    total = sum(len(list((OUT_IMG / s).glob('*.jpg'))) for s in SCENES if (OUT_IMG / s).exists())
    print(f"wrote {total} JPEGs under {OUT_IMG} + _anim_manifest.json")


if __name__ == "__main__":
    main()
