"""DashGaussian training schedule (resolution + primitive-budget).

Adapted from *DashGaussian: Optimizing 3D Gaussian Splatting in 200 Seconds*
(Chen et al., CVPR 2025), https://github.com/YouyuChen0207/DashGaussian.

We reimplement the two schedules DashGaussian adds on top of ordinary 3DGS
training, rather than vendoring the upstream code (which is CC-BY-NC-SA-4.0):

  1. **Resolution schedule** -- render training views at a progressively increasing
     resolution (coarse->fine). The schedule spends iterations in proportion to the
     frequency energy of the training images (fit low frequencies first), realized as
     a set of (iteration, scale) breakpoints with 1/scale^2 interpolated linearly in
     iteration between them (the interpolation form used by the released code).

  2. **Momentum primitive budget** -- an upper bound on the Gaussian count that grows
     with a momentum that is *suppressed at low resolution* (the allowed per-step growth
     scales with scale^2), preventing over-densification during the low-res phase.

This is the "pragmatic" integration: the schedules run on the existing 3DGS rasterizer
(no modified CUDA backward). The exact momentum constants approximate the released code;
the resolution curriculum and the resolution-coupled budget suppression are reproduced.
Within the agentic env, these are the *fixed substrate*; the RL policy still controls
densify/prune/opacity/LR/block-length on top (the "controllable substrate" setup).
"""
from __future__ import annotations

import numpy as np
import torch


def _radial_energy(images: list, n_bins: int, max_side: int = 256) -> np.ndarray:
    """Mean normalized-radial power spectrum over a sample of training images.

    Returns cumulative-energy fraction as a function of normalized frequency, sampled
    at ``n_bins`` radii in (0, 1]. Images are luminance-reduced and downsized to keep
    the FFT cheap; the shape of the cumulative curve (not absolute scale) is what the
    schedule uses.
    """
    acc = np.zeros(n_bins, dtype=np.float64)
    used = 0
    for img in images:
        t = img
        if isinstance(t, torch.Tensor):
            t = t.detach().float()
            if t.dim() == 3:  # C,H,W -> luminance
                t = t.mean(dim=0)
            arr = t.cpu().numpy()
        else:
            arr = np.asarray(t, dtype=np.float64)
            if arr.ndim == 3:
                arr = arr.mean(axis=0 if arr.shape[0] <= 4 else 2)
        h, w = arr.shape[-2], arr.shape[-1]
        if max(h, w) > max_side:  # cheap downsize via strided sampling
            step = int(np.ceil(max(h, w) / max_side))
            arr = arr[::step, ::step]
            h, w = arr.shape
        spec = np.abs(np.fft.fftshift(np.fft.fft2(arr))) ** 2
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)  # normalized radius
        r = np.clip(r / np.sqrt(2.0), 0.0, 1.0)
        idx = np.minimum((r * n_bins).astype(int), n_bins - 1)
        band = np.bincount(idx.ravel(), weights=spec.ravel(), minlength=n_bins)
        band = band[:n_bins]
        s = band.sum()
        if s > 0:
            acc += band / s
            used += 1
    if used == 0:
        acc = np.ones(n_bins, dtype=np.float64)  # fallback: uniform energy
    cum = np.cumsum(acc)
    cum = cum / max(cum[-1], 1e-12)
    return cum  # cum[k] = fraction of energy up to normalized frequency (k+1)/n_bins


class DashScheduler:
    """Resolution + primitive-budget schedule for DashGaussian-style training."""

    def __init__(
        self,
        train_images: list,
        max_steps: int,
        densify_until_iter: int | None = None,
        *,
        mode: str = "freq",
        max_reso_scale: float = 8.0,
        reso_sample_num: int = 32,
        start_significance_factor: float = 4.0,
        res_full_fraction: float = 0.9,
        max_densify_rate_per_step: float = 0.2,
        momentum_decay: float = 0.98,
        initial_count: int = 1,
    ) -> None:
        self.max_steps = int(max_steps)
        # Reach full resolution at res_full_fraction of the training horizon (keyed off
        # the actual episode length, robust to densify_until_iter > max_steps).
        self.res_full_iter = max(1, int(res_full_fraction * max_steps))
        self.s_min = 1.0 / float(max_reso_scale)
        self.mode = str(mode)
        self.max_densify_rate = float(max_densify_rate_per_step)
        self.momentum_decay = float(momentum_decay)
        self._m = float(max(1, initial_count))

        n = int(reso_sample_num)
        # scale grid from s_min up to 1.0 (Nyquist: scale s resolves normalized freq s)
        scales = np.clip(np.linspace(self.s_min, 1.0, n), self.s_min, 1.0)
        # Coarse->fine pacing: the iteration to *reach* scale level k is res_full_iter *
        # (k/(n-1))**gamma. gamma>=1 spends more of training at low resolution; we set it
        # from the training-image frequency concentration (more low-frequency energy ->
        # more time at low res), realizing DashGaussian's "fit low frequencies first".
        if self.mode == "freq":
            cum = _radial_energy(train_images, n)  # cum[k] = energy up to freq (k+1)/n
            low_frac = float(np.interp(0.5, np.linspace(1.0 / n, 1.0, n), cum))  # energy below mid-freq
            gamma = float(np.clip(1.0 + 2.0 * low_frac, 1.0, 3.0))
        else:  # "const"/linear fallback: uniform-in-iteration coarse->fine
            gamma = 1.0
        self.gamma = gamma
        frac = np.linspace(0.0, 1.0, n)
        iters = (self.res_full_iter * (frac ** gamma)).astype(np.float64)
        # store breakpoints as (iteration, 1/scale^2), monotone in iteration
        self._bp_iter = iters
        self._bp_invss = 1.0 / np.clip(scales, 1e-6, None) ** 2

    # ---- resolution schedule ------------------------------------------------
    def res_scale(self, iteration: int) -> float:
        """Rendering scale in (s_min, 1] for this iteration (1 = full resolution)."""
        it = float(iteration)
        if it >= self.res_full_iter:
            return 1.0
        bi = self._bp_iter
        # linear interpolation of 1/scale^2 between surrounding breakpoints
        k = int(np.searchsorted(bi, it, side="right"))
        k = max(1, min(k, len(bi) - 1))
        i0, i1 = bi[k - 1], bi[k]
        v0, v1 = self._bp_invss[k - 1], self._bp_invss[k]
        frac = 0.0 if i1 <= i0 else (it - i0) / (i1 - i0)
        invss = frac * (v1 - v0) + v0
        return float(min(1.0, max(self.s_min, invss ** -0.5)))

    # ---- momentum primitive budget -----------------------------------------
    def primitive_budget(self, iteration: int, current_count: int) -> int:
        """Upper bound on the Gaussian count for this step (advances the momentum).

        Allowed per-step growth scales with scale^2, so the budget barely grows during
        the low-resolution phase and opens up as the render resolution reaches full res.
        """
        s = self.res_scale(iteration)
        allowed_step = self.max_densify_rate * float(current_count) * (s * s)
        self._m = max(self._m, self.momentum_decay * self._m + min(1e6, allowed_step))
        return int(self._m)
