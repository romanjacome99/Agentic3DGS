#!/usr/bin/env python
"""Photometric data augmentation for COLMAP real scenes.

Generates N multiview-consistent photometric variants of each source scene so the
RL policy sees more diverse training dynamics. Each variant samples its OWN set of
photometric hyperparameters (a "schedule": strength grows with the variant index,
plus per-variant jitter) and applies them IDENTICALLY to every view of the scene, so
the augmented scene stays multiview-consistent and remains a valid reconstruction
target (PSNR/loss curves stay meaningful).

Geometry is left untouched: each variant's ``sparse/`` is a Windows directory
junction to the original scene's ``sparse/`` (no point-cloud duplication), and images
keep their original pixel dimensions so the shared intrinsics remain valid.

Output layout (default out-root = archive/real_scenes/rl_aug):
    rl_aug/<scene>            -> junction to rl/<scene>            (original, in pool)
    rl_aug/<scene>_aug00/
        images/<same filenames, photometrically transformed>
        sparse -> junction to rl/<scene>/sparse
    ...
    rl_aug/augmentations_manifest.json   (per-variant sampled params, reproducible)

Usage (validate first, then full run):
    python augment_scenes.py --limit-scenes 1 --limit-variants 3 --contact-sheet
    python augment_scenes.py            # full 11 scenes x 50 variants
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ---------------------------------------------------------------------------
# Paths / defaults
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../3DGS_PROPOSAL
DEFAULT_ARCHIVE = PROJECT_ROOT / "archive" / "real_scenes" / "rl"
DEFAULT_OUT = PROJECT_ROOT / "archive" / "real_scenes" / "rl_aug"

# Mip-360 (9) + truck + train  -> the 11-scene real training pool
DEFAULT_SCENES = [
    "bicycle", "bonsai", "counter", "flowers", "garden", "kitchen",
    "room", "stump", "treehill", "truck", "train",
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"}


# ---------------------------------------------------------------------------
# Deterministic RNG
# ---------------------------------------------------------------------------
def _rng_for(*keys) -> np.random.Generator:
    """Reproducible RNG derived from a global seed + arbitrary key parts."""
    h = abs(hash(("agsaug",) + tuple(keys))) % (2**32)
    return np.random.default_rng(h)


# ---------------------------------------------------------------------------
# Parameter sampling ("hyperparameter schedule": strength scales with variant idx)
# ---------------------------------------------------------------------------
def sample_params(global_seed: int, scene: str, variant: int, n_aug: int) -> dict:
    rng = _rng_for(global_seed, scene, variant)
    # Strength schedule: variant 0 is near-identity, last variant is strongest.
    # A small floor keeps even the mildest variant non-trivial.
    s = 0.25 + 0.75 * (variant / max(1, n_aug - 1))
    # Ranges dialed ~35% narrower than the initial sweep (hue/WB/vignette extra-mild)
    # so even the strongest variant stays close to photorealistic.
    return {
        "strength": float(s),
        "wb_gain": [float(1.0 + rng.uniform(-0.09, 0.09) * s) for _ in range(3)],
        "brightness": float(1.0 + rng.uniform(-0.16, 0.19) * s),
        "contrast": float(1.0 + rng.uniform(-0.19, 0.26) * s),
        "gamma": float(math.exp(rng.uniform(-0.29, 0.29) * s)),   # nonlinear tone
        "saturation": float(1.0 + rng.uniform(-0.26, 0.33) * s),
        "hue_deg": float(rng.uniform(-8.0, 8.0) * s),
        "scurve": float(rng.uniform(0.0, 0.32) * s),              # nonlinear S contrast
        "vignette": float(rng.uniform(0.0, 0.26) * s),
        "sharpen": float(rng.uniform(-0.65, 0.65) * s),           # <0 blur, >0 sharpen
        "noise": float(rng.uniform(0.0, 2.5) * s),               # sigma in 0..255
    }


def _hue_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072],
    ], dtype=np.float32)


def _apply_photometric(arr: np.ndarray, p: dict, noise_rng: np.random.Generator) -> np.ndarray:
    """arr: HxWx3 float32 in [0,1]. Returns transformed float32 in [0,1]."""
    x = arr
    # 1. white balance (per-channel gain)
    x = x * np.asarray(p["wb_gain"], dtype=np.float32)
    # 2. brightness
    x = x * p["brightness"]
    x = np.clip(x, 0.0, 1.0)
    # 3. gamma (nonlinear)
    x = np.power(x, p["gamma"], dtype=np.float32)
    # 4. contrast (linear about mid-grey)
    x = (x - 0.5) * p["contrast"] + 0.5
    x = np.clip(x, 0.0, 1.0)
    # 5. S-curve (nonlinear smoothstep blend) -> gentle bounded contrast
    k = p["scurve"]
    if k > 1e-4:
        x = (1.0 - k) * x + k * (x * x * (3.0 - 2.0 * x))
    # 6. saturation (blend toward luma)
    gray = (x * np.array([0.299, 0.587, 0.114], dtype=np.float32)).sum(axis=2, keepdims=True)
    x = gray + p["saturation"] * (x - gray)
    x = np.clip(x, 0.0, 1.0)
    # 7. hue rotation
    if abs(p["hue_deg"]) > 1e-3:
        x = x @ _hue_matrix(p["hue_deg"]).T
        x = np.clip(x, 0.0, 1.0)
    # 8. vignette (radial darkening, image-space, same for every view)
    if p["vignette"] > 1e-4:
        h, w = x.shape[:2]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        r2 = ((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2
        mask = (1.0 - p["vignette"] * np.clip(r2 / 2.0, 0.0, 1.0))[..., None]
        x = x * mask
    # 9. additive noise (per-image; realistic sensor noise, keeps PSNR meaningful when mild)
    if p["noise"] > 1e-3:
        x = x + noise_rng.normal(0.0, p["noise"] / 255.0, size=x.shape).astype(np.float32)
    return np.clip(x, 0.0, 1.0)


def _post_filter(img: Image.Image, sharpen: float) -> Image.Image:
    if sharpen > 0.05:
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=int(sharpen * 150), threshold=2))
    if sharpen < -0.05:
        return img.filter(ImageFilter.GaussianBlur(radius=min(2.0, abs(sharpen) * 1.4)))
    return img


# ---------------------------------------------------------------------------
# Junction helper (Windows directory junction; no admin required)
# ---------------------------------------------------------------------------
def make_junction(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)],
                       check=True, capture_output=True, text=True)
    else:  # posix fallback for portability
        os.symlink(target, link, target_is_directory=True)


# ---------------------------------------------------------------------------
# Per-variant worker
# ---------------------------------------------------------------------------
def process_variant(task: dict) -> dict:
    scene = task["scene"]
    variant = task["variant"]
    src_dir = Path(task["src_dir"])
    out_dir = Path(task["out_dir"])
    images_subdir = task["images_subdir"]
    jpeg_quality = task["jpeg_quality"]
    p = task["params"]

    src_images = src_dir / images_subdir
    out_images = out_dir / images_subdir
    out_images.mkdir(parents=True, exist_ok=True)

    files = sorted([f for f in src_images.iterdir() if f.suffix in IMG_EXTS])
    # Resume: if already complete, skip the heavy work.
    existing = {f.name for f in out_images.iterdir()} if out_images.exists() else set()
    done = 0
    for f in files:
        out_path = out_images / f.name
        if f.name in existing and out_path.stat().st_size > 0:
            done += 1
            continue
        noise_rng = _rng_for(task["global_seed"], scene, variant, f.name)
        with Image.open(f) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.float32) / 255.0
        out_arr = _apply_photometric(arr, p, noise_rng)
        out_im = Image.fromarray((out_arr * 255.0 + 0.5).astype(np.uint8), mode="RGB")
        out_im = _post_filter(out_im, p["sharpen"])
        ext = f.suffix.lower()
        if ext in (".jpg", ".jpeg"):
            out_im.save(out_path, quality=jpeg_quality, subsampling=0)
        else:
            out_im.save(out_path)
        done += 1

    # Share geometry via junction (no point-cloud duplication).
    make_junction(out_dir / "sparse", src_dir / "sparse")
    return {"scene": scene, "variant": variant, "images": done}


# ---------------------------------------------------------------------------
# Contact sheet (visual QA of the variant sweep for one scene)
# ---------------------------------------------------------------------------
def build_contact_sheet(out_root: Path, scene: str, n_aug: int, images_subdir: str,
                        sheet_path: Path, cols: int = 10, thumb: int = 200) -> None:
    variants = list(range(n_aug))
    rows = math.ceil(len(variants) / cols)
    sheet = Image.new("RGB", (cols * thumb, rows * thumb), (20, 20, 20))
    ref_name = None
    for i, v in enumerate(variants):
        vdir = out_root / f"{scene}_aug{v:02d}" / images_subdir
        if not vdir.exists():
            continue
        if ref_name is None:
            cand = sorted([f for f in vdir.iterdir() if f.suffix in IMG_EXTS])
            if not cand:
                continue
            ref_name = cand[0].name
        fp = vdir / ref_name
        if not fp.exists():
            continue
        with Image.open(fp) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb, thumb))
        sheet.paste(im, ((i % cols) * thumb, (i // cols) * thumb))
    sheet_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(sheet_path)
    print(f"[contact-sheet] {scene}: wrote {sheet_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--scenes", nargs="*", default=DEFAULT_SCENES)
    ap.add_argument("--n-aug", type=int, default=50)
    ap.add_argument("--global-seed", type=int, default=7)
    ap.add_argument("--images-subdir", default="images")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument("--limit-scenes", type=int, default=0, help="only first N scenes (test)")
    ap.add_argument("--limit-variants", type=int, default=0, help="only first N variants (test)")
    ap.add_argument("--contact-sheet", action="store_true", help="write QA sheet per scene")
    ap.add_argument("--link-originals", action="store_true",
                    help="also junction original scenes into out-root so one root holds the full pool")
    ap.add_argument("--dry-run", action="store_true", help="estimate disk + task count, write nothing")
    args = ap.parse_args()

    scenes = list(args.scenes)
    if args.limit_scenes:
        scenes = scenes[:args.limit_scenes]
    n_variants = args.limit_variants or args.n_aug

    # Build task list + disk estimate.
    tasks = []
    est_bytes = 0
    for scene in scenes:
        src_dir = args.archive_root / scene
        src_images = src_dir / args.images_subdir
        if not (src_dir / "sparse").exists() or not src_images.exists():
            print(f"[skip] {scene}: missing sparse/ or {args.images_subdir}/")
            continue
        img_files = [f for f in src_images.iterdir() if f.suffix in IMG_EXTS]
        scene_img_bytes = sum(f.stat().st_size for f in img_files)
        est_bytes += scene_img_bytes * n_variants
        for v in range(n_variants):
            tasks.append({
                "scene": scene,
                "variant": v,
                "src_dir": str(src_dir),
                "out_dir": str(args.out_root / f"{scene}_aug{v:02d}"),
                "images_subdir": args.images_subdir,
                "jpeg_quality": args.jpeg_quality,
                "global_seed": args.global_seed,
                "params": sample_params(args.global_seed, scene, v, args.n_aug),
            })

    print(f"scenes={len(scenes)}  variants/scene={n_variants}  tasks={len(tasks)}")
    print(f"estimated new image disk ~ {est_bytes / 1e9:.1f} GB "
          f"(geometry shared via junction, not counted)")
    if args.dry_run:
        return 0

    args.out_root.mkdir(parents=True, exist_ok=True)

    # Manifest of sampled params (reproducibility).
    manifest = {"global_seed": args.global_seed, "n_aug": args.n_aug,
                "archive_root": str(args.archive_root), "scenes": {}}
    for t in tasks:
        manifest["scenes"].setdefault(t["scene"], {})[f"aug{t['variant']:02d}"] = t["params"]
    (args.out_root / "augmentations_manifest.json").write_text(json.dumps(manifest, indent=2))

    # Optionally expose originals under the same root as the augmented pool.
    if args.link_originals:
        for scene in scenes:
            make_junction(args.out_root / scene, args.archive_root / scene)

    # Run.
    t0 = time.time()
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process_variant, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                r = fut.result()
                completed += 1
                if completed % 10 == 0 or completed == len(tasks):
                    dt = time.time() - t0
                    rate = completed / dt if dt > 0 else 0
                    eta = (len(tasks) - completed) / rate if rate > 0 else 0
                    print(f"[{completed}/{len(tasks)}] {r['scene']}_aug{r['variant']:02d} "
                          f"({r['images']} imgs) | {rate:.2f} var/s | ETA {eta/60:.1f} min",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[ERROR] {t['scene']}_aug{t['variant']:02d}: {e}", flush=True)

    # Contact sheets last (needs generated variants on disk).
    if args.contact_sheet:
        sheet_dir = args.out_root / "_contact_sheets"
        for scene in scenes:
            build_contact_sheet(args.out_root, scene, n_variants, args.images_subdir,
                                sheet_dir / f"{scene}_variants.png")

    print(f"done: {completed}/{len(tasks)} variants in {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    sys.exit(main())
