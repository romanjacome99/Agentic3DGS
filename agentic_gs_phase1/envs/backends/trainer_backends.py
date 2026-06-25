"""Pluggable trainer backends for the agentic env.

A *backend* selects which underlying Gaussian-Splatting codebase the env trains
on, behind a uniform interface. Exactly one backend is active per process (it
puts its repo on sys.path and imports the official-style `scene`/`gaussian_renderer`
modules), so a training run picks its backend once at construction.

Registry:
  - "3dgs"     -> official graphdeco-inria gaussian-splatting        -> 3DGS-Agent
  - "fastergs" -> fhahlbohm Faster-GS fork (faster rasterizer + fused Adam)
                                                                     -> FasterGS-Agent

Selected via config["trainer_backend"] (default "3dgs"). The FasterGS backend
requires the compiled `FasterGSCudaBackend` extension; if it is not built, importing
this backend raises a clear error and the 3DGS backend remains fully functional.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from .gaussian_patches import densify_and_prune_controlled as _densify_and_prune_controlled

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class TrainerBackend:
    """Uniform interface the env uses; subclasses bind a concrete GS codebase."""

    name = "base"
    display = "Base"
    repo_dirname = ""
    uses_faster_rasterizer = False

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self._activate_repo()
        self._pre_import()
        self._import_symbols()
        self._patch_model()

    # ---- repo / import wiring ----------------------------------------------
    @property
    def repo_dir(self) -> Path:
        return PROJECT_ROOT / self.repo_dirname

    def _activate_repo(self) -> None:
        d = str(self.repo_dir)
        if not (self.repo_dir / "scene").is_dir():
            raise FileNotFoundError(
                f"{self.display} backend repository not found at {d}. "
                f"Expected a Gaussian-Splatting codebase there."
            )
        # Put this backend's repo first so its `scene`/`gaussian_renderer` win.
        if d in sys.path:
            sys.path.remove(d)
        sys.path.insert(0, d)

    def _pre_import(self) -> None:
        """Hook to set module toggles before importing (subclass override)."""

    def _import_symbols(self) -> None:
        from gaussian_renderer import render
        from scene import GaussianModel, Scene
        from utils.image_utils import psnr
        from utils.loss_utils import l1_loss, ssim

        self.render_fn = render
        self.GaussianModel = GaussianModel
        self.Scene = Scene
        self.psnr = psnr
        self.l1_loss = l1_loss
        self.ssim = ssim

        try:
            from fused_ssim import fused_ssim

            self.fused_ssim = fused_ssim
            self.FUSED_SSIM_AVAILABLE = True
        except Exception:
            self.fused_ssim = None
            self.FUSED_SSIM_AVAILABLE = False

        try:
            from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

            self.SPARSE_ADAM_AVAILABLE = True
        except Exception:
            self.SPARSE_ADAM_AVAILABLE = False

    def _patch_model(self) -> None:
        if not hasattr(self.GaussianModel, "densify_and_prune_controlled"):
            self.GaussianModel.densify_and_prune_controlled = _densify_and_prune_controlled

    # ---- knobs --------------------------------------------------------------
    @property
    def render_separate_sh(self) -> bool:
        """separate_sh flag passed to render(); off for the faster rasterizer."""
        return False if self.uses_faster_rasterizer else self.SPARSE_ADAM_AVAILABLE

    # ---- construction -------------------------------------------------------
    def make_gaussians(self, sh_degree, optimizer_type):
        return self.GaussianModel(sh_degree, optimizer_type)

    def make_scene(self, dataset, gaussians):
        return self.Scene(dataset, gaussians, shuffle=False)

    # ---- per-iteration ops (backend-specific) -------------------------------
    def render_training(self, cam, gaussians, pipe, bg, *, use_trained_exp):
        """Return {image, radii, visibility_filter, ...backend handle}."""
        raise NotImplementedError

    def accumulate_densification_stats(self, gaussians, render_out) -> None:
        raise NotImplementedError

    def render_image(self, cam, gaussians, pipe, bg, *, use_trained_exp):
        pkg = self.render_fn(
            cam, gaussians, pipe, bg,
            use_trained_exp=use_trained_exp, separate_sh=self.render_separate_sh,
        )
        return pkg["render"]


class ThreeDGSBackend(TrainerBackend):
    name = "3dgs"
    display = "3DGS"
    repo_dirname = "gaussian-splatting"
    uses_faster_rasterizer = False

    def render_training(self, cam, gaussians, pipe, bg, *, use_trained_exp):
        pkg = self.render_fn(
            cam, gaussians, pipe, bg,
            use_trained_exp=use_trained_exp, separate_sh=self.render_separate_sh,
        )
        return {
            "image": pkg["render"],
            "radii": pkg["radii"],
            "visibility_filter": pkg["visibility_filter"],
            "viewspace_points": pkg["viewspace_points"],
        }

    def accumulate_densification_stats(self, gaussians, out) -> None:
        vf = out["visibility_filter"]
        radii = out["radii"]
        gaussians.max_radii2D[vf] = torch.max(gaussians.max_radii2D[vf], radii[vf])
        gaussians.add_densification_stats(out["viewspace_points"], vf)


class FasterGSBackend(TrainerBackend):
    name = "fastergs"
    display = "FasterGS"
    repo_dirname = "gaussian-splatting-fastergs"
    uses_faster_rasterizer = True

    def _pre_import(self) -> None:
        # Surface a precise error if the compiled extension is missing.
        try:
            import FasterGSCudaBackend  # noqa: F401
        except Exception as exc:  # pragma: no cover - environment dependent
            raise ImportError(
                "FasterGS backend requires the compiled 'FasterGSCudaBackend' extension. "
                "Build it with:\n"
                "  scripts/build_fastergs_backend.bat   (or)\n"
                "  pip install ./faster-gaussian-splatting/FasterGSCudaBackend --no-build-isolation\n"
                f"(import failed: {exc})"
            ) from exc
        # KNOWN ISSUE (2026-06): the FasterGSCudaBackend rasterizer compiled under
        # this environment (CUDA 12.1, MSVC 14.44 via -allow-unsupported-compiler)
        # renders correctly in the FORWARD pass but produces a BACKWARD gradient that
        # is direction-scrambled vs the reference 3DGS rasterizer (cosine ~ 0), so the
        # model trains far below parity (~11 dB vs ~25 dB at 1k iters on hotdog).
        # The authors recommend CUDA 12.8; matching that toolkit is the likely fix.
        # Until then this backend is wired but NOT training-correct. Set
        # config["fastergs_acknowledge_grad_bug"]=True to use it anyway (e.g. for
        # forward/inference or further debugging).
        if not bool(self.config.get("fastergs_acknowledge_grad_bug", False)):
            import warnings
            warnings.warn(
                "FasterGS backend: known backward-gradient correctness bug under this "
                "CUDA/MSVC build (forward OK, training does NOT converge to parity). "
                "See backends/trainer_backends.py. Set fastergs_acknowledge_grad_bug=True "
                "to silence.",
                RuntimeWarning,
            )

    def _import_symbols(self) -> None:
        super()._import_symbols()
        # Optional ablation toggles (the fork's modules default both to True).
        import gaussian_renderer
        from scene import gaussian_model
        if "fastergs_use_rasterizer" in self.config:
            gaussian_renderer.USE_FASTERGS_RASTERIZER = bool(self.config["fastergs_use_rasterizer"])
        if "fastergs_use_adam" in self.config:
            gaussian_model.USE_FASTERGS_ADAM = bool(self.config["fastergs_use_adam"])
        # Ensure the fastergs densification helper exists.
        if not hasattr(self.GaussianModel, "add_densification_stats_fastergs"):
            raise AttributeError(
                "FasterGS fork's GaussianModel is missing add_densification_stats_fastergs; "
                "expected the fhahlbohm fork."
            )

    def render_training(self, cam, gaussians, pipe, bg, *, use_trained_exp):
        pkg = self.render_fn(
            cam, gaussians, pipe, bg,
            use_trained_exp=use_trained_exp, separate_sh=False,
        )
        di = pkg["densification_info"]  # [2, N]: row0 = visibility count, row1 = 2D-mean grad
        return {
            "image": pkg["render"],
            # The faster rasterizer exposes no screen-space radius; row0>0 is the
            # visibility mask, and we use row0 itself as the radii proxy (only used
            # for visibility/visible-count, not metric pruning).
            "radii": di[0],
            "visibility_filter": di[0] > 0,
            "densification_info": di,
        }

    def accumulate_densification_stats(self, gaussians, out) -> None:
        gaussians.add_densification_stats_fastergs(out["densification_info"])


_REGISTRY = {
    ThreeDGSBackend.name: ThreeDGSBackend,
    FasterGSBackend.name: FasterGSBackend,
}


def available_backends() -> list[str]:
    return list(_REGISTRY)


def load_backend(config: dict | None = None) -> TrainerBackend:
    cfg = config or {}
    name = str(cfg.get("trainer_backend", "3dgs")).lower()
    if name not in _REGISTRY:
        raise ValueError(
            f"unknown trainer_backend '{name}'; choices: {available_backends()}"
        )
    return _REGISTRY[name](cfg)
