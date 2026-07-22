from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from math import exp, log
from typing import Any

import numpy as np


DISCRETE_ACTIONS = OrderedDict(
    [
        ("block_steps", [50, 100, 200]),
        ("densify_mode", ["off", "conservative", "default", "aggressive"]),
        ("densification_interval", [50, 100, 200, 400]),
        ("prune_mode", ["off", "opacity_only", "opacity_and_size"]),
        ("opacity_reset", ["no_reset", "reset_if_plateau", "force_reset"]),
        ("stop", ["continue", "stop"]),
    ]
)

CONTINUOUS_ACTIONS = OrderedDict(
    [
        ("densify_threshold_mult", (0.25, 4.0, "log")),
        ("prune_opacity_threshold", (0.001, 0.02, "linear")),
        ("position_lr_mult", (0.25, 4.0, "log")),
        ("feature_lr_mult", (0.25, 2.0, "log")),
        ("opacity_lr_mult", (0.25, 2.0, "log")),
        ("scaling_lr_mult", (0.25, 2.0, "log")),
        ("rotation_lr_mult", (0.25, 2.0, "log")),
    ]
)

# ---------------------------------------------------------------------------
# FasterGS-specific policy extension (opt-in via config["fastergs_policy_ext"]).
# These add controls/observations that are meaningful ONLY on the Faster-GS
# backend and that are actually wireable in its faster-rasterizer path. The base
# 3DGS action/observation space above is left completely unchanged; the base
# policy never sees these dims. See discrete_actions_for()/observation_names_for().
#   Actions:  antialiasing toggle (pipe.antialiasing) and SH-band unlock gate
#             (Faster-GS's separate-SH path makes view-dependent bands cheap to defer).
#   Deferred (need CUDA-backend exposure, would be no-ops here): selective/sparse
#             Adam (SparseGaussianAdam is disabled for the faster rasterizer; the
#             fused Adam takes no visibility mask), z-order reorder interval, and the
#             alpha-cull threshold tau_alpha -- all internal to FasterGSCudaBackend.
FASTERGS_DISCRETE_EXT = OrderedDict(
    [
        ("fastergs_antialiasing", ["off", "on"]),
        ("fastergs_sh_unlock", ["allow", "hold"]),
    ]
)

# Appended to OBSERVATION_NAMES (after the budget channel) when the extension is on.
# Per-stage GPU-time fractions expose Faster-GS's shifted cost structure. The densify-grad
# stats summarise the faster rasterizer's densification gradient (densification_info[1] --
# the one gradient the fork writes correctly on COLMAP), threshold-relative, telling the
# policy how strongly the population "wants" to densify. antialiasing_on reads back the toggle.
FASTERGS_OBS_EXT = [
    "fastergs.frac_time_forward",
    "fastergs.frac_time_backward",
    "fastergs.frac_time_optimizer",
    "fastergs.frac_time_densify",
    "fastergs.densify_grad_q50",
    "fastergs.densify_grad_q90",
    "fastergs.densify_grad_active_fraction",
    "fastergs.antialiasing_on",
]


def fastergs_ext_enabled(config: dict | None) -> bool:
    return bool((config or {}).get("fastergs_policy_ext", False))


def fastergs_aa_enabled(config: dict | None) -> bool:
    """Whether the antialiasing action/observation are part of the extension.
    Defaults True (v1). The comparison found AA a net-negative lever, so v2 configs
    set fastergs_ext_aa=false to drop it (fewer heads/obs); old checkpoints still load."""
    return fastergs_ext_enabled(config) and bool((config or {}).get("fastergs_ext_aa", True))


def _fastergs_discrete(config: dict | None) -> "OrderedDict":
    d = OrderedDict()
    if fastergs_aa_enabled(config):
        d["fastergs_antialiasing"] = ["off", "on"]
    d["fastergs_sh_unlock"] = ["allow", "hold"]
    return d


def _fastergs_obs(config: dict | None) -> list:
    obs = [
        "fastergs.frac_time_forward",
        "fastergs.frac_time_backward",
        "fastergs.frac_time_optimizer",
        "fastergs.frac_time_densify",
        "fastergs.densify_grad_q50",
        "fastergs.densify_grad_q90",
        "fastergs.densify_grad_active_fraction",
    ]
    if fastergs_aa_enabled(config):
        obs.append("fastergs.antialiasing_on")
    return obs


def discrete_actions_for(config: dict | None) -> "OrderedDict":
    """Base discrete actions, plus the FasterGS extension when enabled."""
    if not fastergs_ext_enabled(config):
        return DISCRETE_ACTIONS
    merged = OrderedDict(DISCRETE_ACTIONS)
    merged.update(_fastergs_discrete(config))
    return merged


def continuous_actions_for(config: dict | None) -> "OrderedDict":
    """Continuous actions are backend-agnostic; no FasterGS extension (kept for symmetry)."""
    return CONTINUOUS_ACTIONS

OBSERVATION_NAMES = [
    "progress.normalized_iteration",
    "progress.elapsed_wall_fraction",
    "progress.remaining_budget_fraction",
    "progress.current_block_index",
    "loss.ema_l1",
    "loss.ema_dssim",
    "loss.ema_total",
    "loss.slope_recent_blocks",
    "validation.psnr_norm",
    "validation.ssim",
    "validation.lpips_or_zero",
    "validation.improvement",
    "compute.time_per_iteration",
    "compute.time_per_validation_view",
    "compute.peak_vram_fraction",
    "compute.allocated_vram_fraction",
    "compute.active_resolution_scale",
    "gaussians.count_fraction",
    "gaussians.growth_rate",
    "gaussians.budget_used_fraction",
    "prev_action.block_steps",
    "prev_action.densify_mode",
    "prev_action.densification_interval",
    "prev_action.prune_mode",
    "prev_action.opacity_reset",
    "prev_action.stop",
    "prev_action.densify_threshold_mult",
    "prev_action.prune_opacity_threshold",
    "prev_action.position_lr_mult",
    "prev_action.feature_lr_mult",
    "prev_action.opacity_lr_mult",
    "prev_action.scaling_lr_mult",
    "prev_action.rotation_lr_mult",
    "prev_action.improved_validation",
    "gaussians.opacity_q10",
    "gaussians.opacity_q50",
    "gaussians.opacity_q90",
    "gaussians.near_transparent_fraction",
    "gaussians.scale_q50",
    "gaussians.scale_q90",
    "gaussians.anisotropy_q50",
    "gaussians.visible_fraction",
    "gaussians.added_fraction_prev_block",
    "gaussians.pruned_fraction_prev_block",
    "gaussians.active_sh_degree_fraction",
]

# Appended to OBSERVATION_NAMES only when budget-conditioning is enabled
# (config["budget_conditioned"] == true). Lets the policy read its sampled
# wall-clock training budget and adapt its schedule / stopping to it.
BUDGET_OBS_NAME = "progress.time_budget_norm"


def observation_names_for(config: dict) -> list:
    names = list(OBSERVATION_NAMES)
    if bool((config or {}).get("budget_conditioned", False)):
        names.append(BUDGET_OBS_NAME)
    if fastergs_ext_enabled(config):
        names.extend(_fastergs_obs(config))
    return names


@dataclass(frozen=True)
class DecodedAction:
    block_steps: int
    densify_mode: str
    densification_interval: int
    prune_mode: str
    opacity_reset: str
    stop: str
    densify_threshold_mult: float
    prune_opacity_threshold: float
    position_lr_mult: float
    feature_lr_mult: float
    opacity_lr_mult: float
    scaling_lr_mult: float
    rotation_lr_mult: float
    # FasterGS extension (defaults keep the base 3DGS path identical: AA off, SH
    # unlocked on the usual schedule). Only ever set when fastergs_policy_ext is on.
    fastergs_antialiasing: str = "off"
    fastergs_sh_unlock: str = "allow"

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-float(x)))


def scale_continuous(raw_value: float, low: float, high: float, mode: str) -> float:
    t = sigmoid(raw_value)
    if mode == "log":
        return exp(log(low) + t * (log(high) - log(low)))
    return low + t * (high - low)


def neutral_raw_for_value(value: float, low: float, high: float, mode: str) -> float:
    if mode == "log":
        t = (log(value) - log(low)) / (log(high) - log(low))
    else:
        t = (value - low) / (high - low)
    t = min(max(float(t), 1e-6), 1.0 - 1e-6)
    return log(t / (1.0 - t))


def decode_action(action: dict[str, Any], config: dict | None = None) -> DecodedAction:
    discrete = action.get("discrete", {})
    continuous = action.get("continuous", [])
    if hasattr(continuous, "detach"):
        continuous = continuous.detach().cpu().numpy()
    continuous = np.asarray(continuous, dtype=np.float32).reshape(-1)

    decoded: dict[str, Any] = {}
    for name, choices in discrete_actions_for(config).items():
        raw_index = int(discrete.get(name, 0))
        decoded[name] = choices[max(0, min(raw_index, len(choices) - 1))]

    for idx, (name, (low, high, mode)) in enumerate(continuous_actions_for(config).items()):
        raw_value = float(continuous[idx]) if idx < len(continuous) else 0.0
        decoded[name] = scale_continuous(raw_value, low, high, mode)

    return DecodedAction(**decoded)


def default_action(config: dict | None = None) -> dict[str, Any]:
    discrete = {
        "block_steps": DISCRETE_ACTIONS["block_steps"].index(100),
        "densify_mode": DISCRETE_ACTIONS["densify_mode"].index("default"),
        "densification_interval": DISCRETE_ACTIONS["densification_interval"].index(100),
        "prune_mode": DISCRETE_ACTIONS["prune_mode"].index("opacity_and_size"),
        "opacity_reset": DISCRETE_ACTIONS["opacity_reset"].index("reset_if_plateau"),
        "stop": DISCRETE_ACTIONS["stop"].index("continue"),
    }
    if fastergs_ext_enabled(config):
        # neutral defaults: AA off, SH unlocked on the usual schedule (index 0 of each)
        discrete["fastergs_antialiasing"] = 0
        discrete["fastergs_sh_unlock"] = 0
    target_values = {
        "densify_threshold_mult": 1.0,
        "prune_opacity_threshold": 0.005,
        "position_lr_mult": 1.0,
        "feature_lr_mult": 1.0,
        "opacity_lr_mult": 1.0,
        "scaling_lr_mult": 1.0,
        "rotation_lr_mult": 1.0,
    }
    continuous = [
        neutral_raw_for_value(target_values[name], low, high, mode)
        for name, (low, high, mode) in CONTINUOUS_ACTIONS.items()
    ]
    return {"discrete": discrete, "continuous": np.asarray(continuous, dtype=np.float32)}


def normalize_discrete_choice(name: str, value: Any) -> float:
    choices = DISCRETE_ACTIONS[name]
    if value in choices:
        index = choices.index(value)
    else:
        index = int(value)
    return 0.0 if len(choices) == 1 else index / float(len(choices) - 1)


def action_to_observation_values(decoded: DecodedAction) -> list[float]:
    values = [
        normalize_discrete_choice("block_steps", decoded.block_steps),
        normalize_discrete_choice("densify_mode", decoded.densify_mode),
        normalize_discrete_choice("densification_interval", decoded.densification_interval),
        normalize_discrete_choice("prune_mode", decoded.prune_mode),
        normalize_discrete_choice("opacity_reset", decoded.opacity_reset),
        normalize_discrete_choice("stop", decoded.stop),
    ]
    for name, (low, high, mode) in CONTINUOUS_ACTIONS.items():
        value = float(getattr(decoded, name))
        if mode == "log":
            values.append((log(value) - log(low)) / (log(high) - log(low)))
        else:
            values.append((value - low) / (high - low))
    return [float(min(max(v, 0.0), 1.0)) for v in values]

