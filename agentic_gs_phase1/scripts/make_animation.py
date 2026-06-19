"""Assemble baseline-vs-agentic training-progress animations.

For each scene, load the baseline and agentic frame manifests, pick frames at a
shared iteration timeline, composite [GT | Baseline | Agentic] with per-panel
PSNR labels, and write an MP4 (and GIF) per scene.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TIMELINE = [0, 200, 400, 600, 800, 1000, 1400, 2000, 3000, 4000, 5000, 7000]
BG = (16, 19, 26)
INK = (232, 236, 244)
BASE_C = (78, 163, 255)
AGENT_C = (255, 138, 61)
MUTED = (154, 166, 191)


def load_font(size):
    for p in (r"C:\Windows\Fonts\segoeui.ttf", r"C:\Windows\Fonts\arial.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


def nearest(frames, target):
    return min(frames, key=lambda f: abs(f["iter"] - target))


def build_scene(scene, base_dir, agent_dir, out_dir, fps, size):
    bman = json.loads((base_dir / "manifest.json").read_text(encoding="utf-8"))
    aman = json.loads((agent_dir / "manifest.json").read_text(encoding="utf-8"))
    bframes, aframes = bman["frames"], aman["frames"]
    maxit = min(bframes[-1]["iter"], aframes[-1]["iter"])
    timeline = [t for t in TIMELINE if t <= maxit] or [maxit]

    gt = Image.open(base_dir / "gt.png").convert("RGB").resize((size, size))
    title_f, lab_f, sub_f = load_font(26), load_font(22), load_font(18)
    pad, header, labh = 12, 46, 34
    W = 3 * size + 4 * pad
    H = header + labh + size + pad

    rendered = []
    for t in timeline:
        bf, af = nearest(bframes, t), nearest(aframes, t)
        canvas = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(canvas)
        d.text((pad, 10), f"{scene}", font=title_f, fill=INK)
        it_txt = f"iteration {t}"
        d.text((W - pad - d.textlength(it_txt, font=title_f), 12), it_txt, font=title_f, fill=MUTED)
        cols = [
            (gt, "Ground truth", None, INK),
            (Image.open(base_dir / "frames" / bf["file"]).convert("RGB").resize((size, size)),
             f"Baseline  {bf['psnr']:.2f} dB", f"{bf['count']//1000}k G", BASE_C),
            (Image.open(agent_dir / "frames" / af["file"]).convert("RGB").resize((size, size)),
             f"Agentic  {af['psnr']:.2f} dB", f"{af['count']//1000}k G", AGENT_C),
        ]
        for i, (img, label, sub, color) in enumerate(cols):
            x = pad + i * (size + pad)
            d.text((x, header + 5), label, font=lab_f, fill=color)
            if sub:
                d.text((x + size - d.textlength(sub, font=sub_f) - 4, header + 8), sub, font=sub_f, fill=MUTED)
            canvas.paste(img, (x, header + labh))
        rendered.append(np.asarray(canvas))

    out_dir.mkdir(parents=True, exist_ok=True)
    seq = rendered + [rendered[-1]] * max(0, fps * 2)  # hold final frame ~2s
    mp4 = out_dir / f"{scene}_training.mp4"
    imageio.mimsave(mp4, seq, fps=fps, codec="libx264", quality=8, macro_block_size=1)
    gif = out_dir / f"{scene}_training.gif"
    imageio.mimsave(gif, seq, duration=1.0 / fps)
    print(f"[{scene}] {len(timeline)} timeline frames -> {mp4.name} ({mp4.stat().st_size//1024} KB), "
          f"{gif.name} ({gif.stat().st_size//1024} KB)", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True, help="dir with <scene>_<method> subfolders")
    ap.add_argument("--scenes", nargs="+", default=["hotdog", "drums", "materials"])
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=3)
    ap.add_argument("--size", type=int, default=400)
    args = ap.parse_args()
    for scene in args.scenes:
        build_scene(scene, args.root / f"{scene}_baseline", args.root / f"{scene}_agentic",
                    args.out_dir, args.fps, args.size)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
