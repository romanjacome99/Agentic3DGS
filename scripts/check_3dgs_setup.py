from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_ROOT = ROOT / "archive" / "nerf_synthetic"
DEFAULT_3DGS_DIR = ROOT / "gaussian-splatting"


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def print_status(label: str, ok: bool, detail: str = "") -> None:
    state = "OK" if ok else "MISSING"
    suffix = f" - {detail}" if detail else ""
    print(f"{state:7} {label}{suffix}")


def discover_scenes(archive_root: Path) -> list[str]:
    if not archive_root.exists():
        return []
    scenes = []
    for item in sorted(archive_root.iterdir()):
        if item.is_dir() and (item / "transforms_train.json").exists() and (item / "transforms_test.json").exists():
            scenes.append(item.name)
    return scenes


def main() -> int:
    print(f"Python: {sys.executable}")
    print(f"Project: {ROOT}")
    print("")

    print_status("official 3DGS source", DEFAULT_3DGS_DIR.exists(), str(DEFAULT_3DGS_DIR))
    for submodule in [
        "submodules/diff-gaussian-rasterization",
        "submodules/simple-knn",
        "submodules/fused-ssim",
    ]:
        path = DEFAULT_3DGS_DIR / submodule
        print_status(submodule, path.exists(), str(path))

    print("")
    required_modules = ["plyfile", "PIL", "tqdm", "torch", "torchvision", "diff_gaussian_rasterization", "simple_knn"]
    optional_modules = ["fused_ssim"]
    missing_required = []

    for module in required_modules:
        ok = has_module(module)
        print_status(module, ok)
        if not ok:
            missing_required.append(module)

    for module in optional_modules:
        print_status(module, has_module(module), "optional acceleration")

    if has_module("torch"):
        try:
            import torch

            print_status("torch import", True, f"version={torch.__version__}, cuda={torch.version.cuda}")
            print_status("torch.cuda available", torch.cuda.is_available())
            if torch.cuda.is_available():
                print_status("cuda device", True, torch.cuda.get_device_name(0))
        except Exception as exc:
            missing_required.append("torch(import)")
            print_status("torch import", False, repr(exc))

    print("")
    scenes = discover_scenes(DEFAULT_ARCHIVE_ROOT)
    print_status("archive NeRF synthetic root", bool(scenes), str(DEFAULT_ARCHIVE_ROOT))
    if scenes:
        print("Scenes:", ", ".join(scenes))

    print("")
    if missing_required:
        print("Setup is incomplete. Create/activate the official environment, then install the CUDA submodules:")
        print("  conda env create --file gaussian-splatting/environment.yml")
        print("  conda activate gaussian_splatting")
        print("  python scripts/check_3dgs_setup.py")
        return 1

    print("Setup looks ready for baseline training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
