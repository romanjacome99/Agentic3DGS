"""Project-specific GaussianModel methods, kept backend-agnostic.

`densify_and_prune_controlled` is the agent-controlled densify/prune routine the
env drives. The official `gaussian-splatting` repo carries an identical copy in
its patched `scene/gaussian_model.py`; here we keep a standalone version so it can
be monkey-patched onto *any* backend's GaussianModel (e.g. the Faster-GS fork,
which ships vanilla). Logic must stay byte-for-byte equivalent to the 3DGS copy.
"""
from __future__ import annotations

import torch


def densify_and_prune_controlled(
    self,
    max_grad,
    min_opacity,
    extent,
    max_screen_size,
    radii,
    densify_enabled=True,
    prune_mode="opacity_and_size",
    min_remaining=1000,
):
    before_count = int(self.get_xyz.shape[0])
    grads = self.xyz_gradient_accum / self.denom
    grads[grads.isnan()] = 0.0

    self.tmp_radii = radii
    if densify_enabled:
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    after_densify_count = int(self.get_xyz.shape[0])
    prune_mode = str(prune_mode)
    if prune_mode == "off":
        prune_count = 0
    else:
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if prune_mode == "opacity_and_size" and max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        if prune_mask.numel() > 0:
            keep_count = min(max(1, int(min_remaining)), int(prune_mask.numel()))
            max_prunable = int(prune_mask.numel()) - keep_count
            if int(prune_mask.sum().item()) > max_prunable:
                keep_indices = torch.topk(self.get_opacity.squeeze(), k=keep_count, largest=True).indices
                prune_mask = torch.ones_like(prune_mask, dtype=bool)
                prune_mask[keep_indices] = False
        prune_count = int(prune_mask.sum().item())
        if prune_count > 0:
            self.prune_points(prune_mask)

    final_count = int(self.get_xyz.shape[0])
    self.tmp_radii = None

    torch.cuda.empty_cache()
    return {
        "before": before_count,
        "after_densify": after_densify_count,
        "after": final_count,
        "added": max(0, after_densify_count - before_count),
        "pruned": prune_count,
    }
