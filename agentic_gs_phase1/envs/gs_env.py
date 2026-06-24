from __future__ import annotations

import csv
import gc
import json
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from .spaces import (
    OBSERVATION_NAMES,
    action_to_observation_values,
    decode_action,
    default_action,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The underlying Gaussian-Splatting codebase (3DGS official vs Faster-GS fork) is
# selected at env construction via a pluggable backend. The render/loss/model
# symbols are provided by self.backend rather than imported at module load, so a
# run can choose its trainer (3DGS-Agent, FasterGS-Agent, ...) without eagerly
# importing — and conflicting with — the other codebase.
from .backends import load_backend  # noqa: E402

# Backward-compat module-level flag: some eval scripts import this from here.
# It is independent of the active backend (purely whether the sparse-Adam
# rasterizer extension is importable).
try:  # noqa: E402
    from diff_gaussian_rasterization import SparseGaussianAdam  # noqa: F401

    SPARSE_ADAM_AVAILABLE = True
except Exception:  # noqa: E402
    SPARSE_ADAM_AVAILABLE = False


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _clip01(value: float) -> float:
    if not np.isfinite(value):
        return 0.0
    return float(min(max(value, 0.0), 1.0))


def _memory_gb() -> dict[str, float]:
    return {
        "allocated": torch.cuda.memory_allocated() / (1024**3),
        "reserved": torch.cuda.memory_reserved() / (1024**3),
        "peak": torch.cuda.max_memory_allocated() / (1024**3),
    }


class AgenticGSEnv:
    """Block-wise RL environment around the stock 3DGS optimizer.

    Rewards and observations use only a held-out subset of training cameras.
    Dataset test cameras are loaded by the upstream scene object but are never
    used for policy state or reward construction in this environment.
    """

    observation_names = OBSERVATION_NAMES

    def __init__(self, config: dict[str, Any], run_dir: str | Path | None = None, seed: int = 0):
        self.config = config
        # Pluggable trainer backend (3DGS-Agent / FasterGS-Agent / ...).
        self.backend = load_backend(config)
        self.seed = int(seed)
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self.archive_root = _resolve_path(config.get("archive_root", "archive/nerf_synthetic"))
        self.output_root = _resolve_path(config.get("output_root", "outputs/agentic_gs_phase1"))
        self.run_dir = _resolve_path(run_dir) if run_dir is not None else self.output_root / time.strftime("%Y%m%d_%H%M%S")
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.scene_id = ""
        self.model_path: Path | None = None
        self.scene: Any = None
        self.gaussians: Any = None
        self.dataset = None
        self.opt = None
        self.pipe = None
        self.background = None
        self.train_cameras = []
        self.validation_cameras = []
        self.train_stack = []

        self.iteration = 0
        self.block_index = 0
        self.episode_start_time = 0.0
        self.elapsed_seconds = 0.0
        self.training_seconds = 0.0
        self.last_validation = {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "quality": 0.0}
        self.initial_validation_quality = 0.0
        self.prev_validation_quality = 0.0
        self.last_quality_gain = 0.0
        self.prev_gaussian_count = 0
        self.prev_action = decode_action(default_action())
        self.prev_action_improved = False
        self.last_quality_gain = 0.0
        self.opacity_resets = 0
        self.last_opacity_reset_block = -10_000
        self.last_opacity_reset_iteration = -10_000
        self.last_block_stats = self._empty_block_stats()
        self.loss_history: list[float] = []
        self.block_log_path: Path | None = None
        self.initial_gaussian_count = 0

    @property
    def obs_dim(self) -> int:
        return len(self.observation_names)

    def reset(self, scene_id: str, episode_id: int = 0) -> np.ndarray:
        self.close()
        self.scene_id = scene_id
        scene_path = self.archive_root / scene_id
        if not scene_path.exists():
            raise FileNotFoundError(f"Scene '{scene_id}' was not found under {self.archive_root}")

        self.model_path = self.run_dir / scene_id / f"episode_{episode_id:04d}"
        self.model_path.mkdir(parents=True, exist_ok=True)
        self.block_log_path = self.model_path / "agentic_blocks.csv"
        self._init_block_log()

        self.iteration = 0
        self.block_index = 0
        self.elapsed_seconds = 0.0
        self.training_seconds = 0.0
        self.loss_history = []
        self.last_block_stats = self._empty_block_stats()
        self.prev_action = decode_action(default_action())
        self.prev_action_improved = False
        self.opacity_resets = 0
        self.last_opacity_reset_block = -10_000
        self.last_opacity_reset_iteration = -10_000

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        self.dataset = self._dataset_args(scene_path, self.model_path)
        self.opt = self._optimization_args()
        self.pipe = self._pipeline_args()
        self.gaussians = self.backend.make_gaussians(self.dataset.sh_degree, self.opt.optimizer_type)
        self.scene = self.backend.make_scene(self.dataset, self.gaussians)
        self.gaussians.training_setup(self.opt)

        bg_color = [1, 1, 1] if self.dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        self._split_train_validation()

        self.prev_gaussian_count = self.gaussians.get_xyz.shape[0]
        self.initial_gaussian_count = int(self.prev_gaussian_count)
        self.last_validation = self._evaluate_validation_subset()
        self.initial_validation_quality = self.last_validation["quality"]
        self.prev_validation_quality = self.last_validation["quality"]
        self.episode_start_time = time.perf_counter()

        self._write_episode_metadata()
        return self._build_observation()

    def step(self, action: dict[str, Any]) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        if self.scene is None or self.gaussians is None:
            raise RuntimeError("Call reset(scene_id) before step(action).")

        controls = decode_action(action)
        self._maybe_reset_opacity(controls)
        block_stats = self._empty_block_stats()
        block_stats["start_iteration"] = self.iteration + 1
        block_stats["start_gaussians"] = int(self.gaussians.get_xyz.shape[0])
        block_start = time.perf_counter()

        for _ in range(controls.block_steps):
            if self.iteration >= int(self.opt.iterations):
                break
            self.iteration += 1
            iter_stats = self._run_training_iteration(controls)
            for key in ["l1", "dssim", "total_loss", "visible_gaussians"]:
                block_stats[key] += iter_stats[key]
            block_stats["iterations_ran"] += 1
            block_stats["last_visible_fraction"] = iter_stats["visible_fraction"]
            block_stats["densify_events"] += iter_stats["densify_event"]
            block_stats["gaussians_added"] += iter_stats["gaussians_added"]
            block_stats["gaussians_pruned"] += iter_stats["gaussians_pruned"]

            if not np.isfinite(iter_stats["total_loss"]):
                block_stats["numerical_failure"] = 1
                break

            if self._hard_budget_exceeded():
                block_stats["hard_budget_exceeded"] = 1
                break

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        block_stats["block_seconds"] = time.perf_counter() - block_start
        self.elapsed_seconds = time.perf_counter() - self.episode_start_time
        self.training_seconds += block_stats["block_seconds"]

        iterations_ran = max(1, int(block_stats["iterations_ran"]))
        block_stats["l1"] /= iterations_ran
        block_stats["dssim"] /= iterations_ran
        block_stats["total_loss"] /= iterations_ran
        block_stats["visible_gaussians"] /= iterations_ran
        block_stats["time_per_iteration"] = block_stats["block_seconds"] / iterations_ran
        block_stats["end_gaussians"] = int(self.gaussians.get_xyz.shape[0])
        block_stats["gaussian_growth"] = block_stats["end_gaussians"] - block_stats["start_gaussians"]

        validation = self._evaluate_validation_subset()
        reward, reward_terms = self._compute_reward(validation, block_stats)
        quality_gain = validation["quality"] - self.prev_validation_quality
        self.last_quality_gain = quality_gain
        self.prev_action_improved = quality_gain > 0.0
        self.prev_validation_quality = validation["quality"]
        self.last_validation = validation
        self.prev_action = controls
        self.last_block_stats = block_stats
        self.loss_history.append(block_stats["total_loss"])

        self.block_index += 1
        done = self._is_done(controls, block_stats)
        if done and block_stats["numerical_failure"]:
            reward -= float(self.config.get("reward", {}).get("failure_penalty", 2.0))
            reward_terms["failure_penalty"] = -float(self.config.get("reward", {}).get("failure_penalty", 2.0))

        info = {
            "scene": self.scene_id,
            "iteration": self.iteration,
            "block_index": self.block_index,
            "action": controls.as_dict(),
            "validation": validation,
            "reward_terms": reward_terms,
            "block_stats": block_stats,
            "test_camera_accessed": False,
        }
        self._append_block_log(info, reward)
        return self._build_observation(), float(reward), done, info

    def close(self) -> None:
        self.scene = None
        self.gaussians = None
        self.dataset = None
        self.opt = None
        self.pipe = None
        self.background = None
        self.train_cameras = []
        self.validation_cameras = []
        self.train_stack = []
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def save_model(self) -> None:
        if self.scene is not None:
            self.scene.save(self.iteration)

    def _dataset_args(self, scene_path: Path, model_path: Path) -> SimpleNamespace:
        return SimpleNamespace(
            sh_degree=int(self.config.get("sh_degree", 3)),
            source_path=str(scene_path),
            model_path=str(model_path),
            images=str(self.config.get("images", "images")),
            depths="",
            resolution=int(self.config.get("resolution", 1)),
            white_background=bool(self.config.get("white_background", True)),
            data_device=str(self.config.get("data_device", "cuda")),
            eval=True,
            train_test_exp=False,
        )

    def _optimization_args(self) -> SimpleNamespace:
        opt_cfg = self.config.get("optimization", {})
        return SimpleNamespace(
            iterations=int(self.config.get("max_episode_iterations", 7000)),
            position_lr_init=float(opt_cfg.get("position_lr_init", 0.00016)),
            position_lr_final=float(opt_cfg.get("position_lr_final", 0.0000016)),
            position_lr_delay_mult=float(opt_cfg.get("position_lr_delay_mult", 0.01)),
            position_lr_max_steps=int(opt_cfg.get("position_lr_max_steps", self.config.get("max_episode_iterations", 7000))),
            feature_lr=float(opt_cfg.get("feature_lr", 0.0025)),
            opacity_lr=float(opt_cfg.get("opacity_lr", 0.025)),
            scaling_lr=float(opt_cfg.get("scaling_lr", 0.005)),
            rotation_lr=float(opt_cfg.get("rotation_lr", 0.001)),
            exposure_lr_init=float(opt_cfg.get("exposure_lr_init", 0.01)),
            exposure_lr_final=float(opt_cfg.get("exposure_lr_final", 0.001)),
            exposure_lr_delay_steps=int(opt_cfg.get("exposure_lr_delay_steps", 0)),
            exposure_lr_delay_mult=float(opt_cfg.get("exposure_lr_delay_mult", 0.0)),
            percent_dense=float(opt_cfg.get("percent_dense", 0.01)),
            lambda_dssim=float(opt_cfg.get("lambda_dssim", 0.2)),
            densification_interval=int(opt_cfg.get("densification_interval", 100)),
            opacity_reset_interval=int(opt_cfg.get("opacity_reset_interval", 3000)),
            densify_from_iter=int(opt_cfg.get("densify_from_iter", 500)),
            densify_until_iter=int(opt_cfg.get("densify_until_iter", 15000)),
            densify_grad_threshold=float(opt_cfg.get("densify_grad_threshold", 0.0002)),
            depth_l1_weight_init=float(opt_cfg.get("depth_l1_weight_init", 0.0)),
            depth_l1_weight_final=float(opt_cfg.get("depth_l1_weight_final", 0.0)),
            random_background=bool(opt_cfg.get("random_background", False)),
            optimizer_type=str(opt_cfg.get("optimizer_type", "default")),
        )

    def _pipeline_args(self) -> SimpleNamespace:
        pipe_cfg = self.config.get("pipeline", {})
        return SimpleNamespace(
            convert_SHs_python=bool(pipe_cfg.get("convert_SHs_python", False)),
            compute_cov3D_python=bool(pipe_cfg.get("compute_cov3D_python", False)),
            debug=bool(pipe_cfg.get("debug", False)),
            antialiasing=bool(pipe_cfg.get("antialiasing", False)),
        )

    def _split_train_validation(self) -> None:
        cameras = list(self.scene.getTrainCameras())
        if len(cameras) < 2:
            raise RuntimeError("Need at least two training cameras for train/validation split.")
        requested = int(self.config.get("validation_cameras", 8))
        validation_count = min(max(1, requested), max(1, len(cameras) // 5), len(cameras) - 1)
        selection = str(self.config.get("validation_selection", "coverage"))
        if selection == "coverage":
            self.validation_cameras = self._coverage_subset(cameras, validation_count)
        else:
            self.rng.shuffle(cameras)
            self.validation_cameras = cameras[:validation_count]
        validation_ids = {id(camera) for camera in self.validation_cameras}
        self.train_cameras = [camera for camera in cameras if id(camera) not in validation_ids]
        self.train_stack = []

    def _coverage_subset(self, cameras: list, count: int) -> list:
        """Farthest-point sampling over camera centers (random seed start).

        A random subset can cluster on one side of the hemisphere, letting the
        policy optimize those views while unobserved directions degrade — which
        the test cameras then expose. FPS spreads the reward cameras over the
        capture sphere so validation quality tracks novel-view quality better.
        """
        centers = np.stack(
            [
                np.asarray(camera.camera_center.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
                for camera in cameras
            ]
        )
        start = self.rng.randrange(len(cameras))
        chosen = [start]
        distances = np.linalg.norm(centers - centers[start], axis=1)
        while len(chosen) < min(count, len(cameras)):
            next_index = int(distances.argmax())
            chosen.append(next_index)
            distances = np.minimum(distances, np.linalg.norm(centers - centers[next_index], axis=1))
        return [cameras[i] for i in chosen]

    def _sample_training_camera(self):
        if not self.train_stack:
            self.train_stack = list(self.train_cameras)
            self.rng.shuffle(self.train_stack)
        return self.train_stack.pop()

    def _run_training_iteration(self, controls) -> dict[str, float]:
        assert self.gaussians is not None
        self.gaussians.update_learning_rate(self.iteration)
        self._apply_lr_multipliers(controls)

        if self.iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        viewpoint_cam = self._sample_training_camera()
        bg = torch.rand((3), device="cuda") if self.opt.random_background else self.background
        render_out = self.backend.render_training(
            viewpoint_cam,
            self.gaussians,
            self.pipe,
            bg,
            use_trained_exp=self.dataset.train_test_exp,
        )
        image = render_out["image"]
        visibility_filter = render_out["visibility_filter"]
        radii = render_out["radii"]

        if viewpoint_cam.alpha_mask is not None:
            image *= viewpoint_cam.alpha_mask.cuda()

        gt_image = viewpoint_cam.original_image.cuda()
        l1_value = self.backend.l1_loss(image, gt_image)
        if self.backend.FUSED_SSIM_AVAILABLE:
            ssim_value = self.backend.fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = self.backend.ssim(image, gt_image)
        dssim_value = 1.0 - ssim_value
        loss = (1.0 - self.opt.lambda_dssim) * l1_value + self.opt.lambda_dssim * dssim_value
        reg_value = self._compactness_regularizer(controls)
        loss = loss + reg_value
        loss.backward()

        densify_stats = {"added": 0, "pruned": 0}
        densify_event = 0
        with torch.no_grad():
            if self.iteration < self.opt.densify_until_iter:
                self.backend.accumulate_densification_stats(self.gaussians, render_out)
                if self._should_apply_densification(controls):
                    densify_event = 1
                    size_threshold = 20 if self.iteration > self.opt.opacity_reset_interval else None
                    threshold = self._controlled_densify_threshold(controls)
                    prune_mode = "opacity_only" if controls.prune_mode == "opacity_only" else controls.prune_mode
                    densify_stats = self.gaussians.densify_and_prune_controlled(
                        threshold,
                        self._effective_prune_opacity_threshold(controls),
                        self.scene.cameras_extent,
                        size_threshold,
                        radii,
                        densify_enabled=controls.densify_mode != "off",
                        prune_mode=prune_mode,
                        min_remaining=self._min_remaining_gaussians(),
                    )

            if self.iteration < self.opt.iterations:
                self.gaussians.exposure_optimizer.step()
                self.gaussians.exposure_optimizer.zero_grad(set_to_none=True)
                if self.opt.optimizer_type == "sparse_adam" and self.backend.SPARSE_ADAM_AVAILABLE:
                    visible = radii > 0
                    self.gaussians.optimizer.step(visible, radii.shape[0])
                else:
                    self.gaussians.optimizer.step()
                self.gaussians.optimizer.zero_grad(set_to_none=True)

        visible_count = int((radii > 0).sum().item())
        total_count = max(1, int(self.gaussians.get_xyz.shape[0]))
        return {
            "l1": float(l1_value.detach().item()),
            "dssim": float(dssim_value.detach().item()),
            "total_loss": float(loss.detach().item()),
            "visible_gaussians": float(visible_count),
            "visible_fraction": float(visible_count / total_count),
            "densify_event": float(densify_event),
            "gaussians_added": float(densify_stats.get("added", 0)),
            "gaussians_pruned": float(densify_stats.get("pruned", 0)),
        }

    def _apply_lr_multipliers(self, controls) -> None:
        mults = {
            "xyz": controls.position_lr_mult,
            "f_dc": controls.feature_lr_mult,
            "f_rest": controls.feature_lr_mult,
            "opacity": controls.opacity_lr_mult,
            "scaling": controls.scaling_lr_mult,
            "rotation": controls.rotation_lr_mult,
        }
        base_lrs = {
            "f_dc": self.opt.feature_lr,
            "f_rest": self.opt.feature_lr / 20.0,
            "opacity": self.opt.opacity_lr,
            "scaling": self.opt.scaling_lr,
            "rotation": self.opt.rotation_lr,
        }
        for group in self.gaussians.optimizer.param_groups:
            name = group["name"]
            if name == "xyz":
                group["lr"] *= mults[name]
            elif name in base_lrs:
                group["lr"] = base_lrs[name] * mults[name]

    def _compactness_regularizer(self, controls) -> torch.Tensor:
        """Opacity / volume regularization added to the photometric loss.

        Pushes low-contribution Gaussians toward transparency and discourages
        oversized primitives so that pruning actually reduces the model size.
        Gated to start after ``reg_from_iter`` so early densification is healthy.
        Strength can be scaled by an optional policy action ``compaction_mult``.
        """
        loss_cfg = self.config.get("loss", {})
        reg = torch.zeros((), device="cuda")
        if self.gaussians is None:
            return reg
        if self.iteration < int(loss_cfg.get("reg_from_iter", 0)):
            return reg
        mult = float(getattr(controls, "compaction_mult", 1.0))
        lambda_opacity = float(loss_cfg.get("lambda_opacity", 0.0)) * mult
        lambda_volume = float(loss_cfg.get("lambda_volume", 0.0)) * mult
        if lambda_opacity > 0.0:
            reg = reg + lambda_opacity * self.gaussians.get_opacity.mean()
        if lambda_volume > 0.0:
            extent = max(1e-6, float(self.scene.cameras_extent))
            reg = reg + lambda_volume * (self.gaussians.get_scaling / extent).prod(dim=1).mean()
        return reg

    def _reference_gaussian_count(self) -> float:
        """Budget the policy is expected to stay under (defaults to a multiple
        of the initial count, overridable per scene to the measured baseline)."""
        reward_cfg = self.config.get("reward", {})
        budgets = reward_cfg.get("reference_gaussian_budget", {})
        if isinstance(budgets, dict) and self.scene_id in budgets:
            return max(1.0, float(budgets[self.scene_id]))
        target = float(reward_cfg.get("reference_gaussian_target", 0.0))
        if target > 0.0:
            return target
        mult = float(reward_cfg.get("reference_gaussian_mult", 3.0))
        return max(1.0, mult * max(1, self.initial_gaussian_count))

    def _controlled_densify_threshold(self, controls) -> float:
        mode_mult = {
            "off": 1.0,
            "conservative": 2.0,
            "default": 1.0,
            "aggressive": 0.5,
        }[controls.densify_mode]
        return float(self.opt.densify_grad_threshold * mode_mult * controls.densify_threshold_mult)

    def _effective_prune_opacity_threshold(self, controls) -> float:
        """Post-reset prune guard.

        Stock 3DGS resets opacity to 0.01 and prunes at 0.005, so survivors can
        recover. Our action range allows prune thresholds above the reset value,
        which lets a reset followed by one prune event wipe the model down to the
        safety floor in a single block (observed: 104k -> 50k at iter ~800). Cap
        the threshold to the stock value for a recovery window after each reset.
        """
        safety = self.config.get("safety", {})
        guard_iters = int(safety.get("post_reset_prune_guard_iters", 300))
        if self.iteration - self.last_opacity_reset_iteration < guard_iters:
            cap = float(safety.get("post_reset_prune_cap", 0.005))
            return min(float(controls.prune_opacity_threshold), cap)
        return float(controls.prune_opacity_threshold)

    def _min_remaining_gaussians(self) -> int:
        safety = self.config.get("safety", {})
        absolute_min = int(safety.get("min_gaussians", 1000))
        initial_fraction = float(safety.get("min_gaussian_fraction_of_initial", 0.5))
        initial_min = int(round(max(0.0, initial_fraction) * max(1, self.initial_gaussian_count)))
        floor = max(absolute_min, initial_min)
        # Cap the loss from any single prune event so a collapse to the floor
        # takes sustained pruning across blocks (visible to the reward) instead
        # of one cliff-edge action.
        max_prune_fraction = float(safety.get("max_prune_fraction_per_event", 0.25))
        if 0.0 < max_prune_fraction < 1.0 and self.gaussians is not None:
            current = int(self.gaussians.get_xyz.shape[0])
            floor = max(floor, int(round((1.0 - max_prune_fraction) * current)))
        return floor

    def _should_apply_densification(self, controls) -> bool:
        if self.iteration <= self.opt.densify_from_iter:
            return False
        return self.iteration % int(controls.densification_interval) == 0

    def _maybe_reset_opacity(self, controls) -> None:
        if self.gaussians is None:
            return
        safety = self.config.get("safety", {})
        min_iter = int(safety.get("min_opacity_reset_iter", 500))
        cooldown = int(safety.get("opacity_reset_cooldown_blocks", 3))
        max_resets = int(safety.get("max_opacity_resets", 5))
        if self.iteration < min_iter:
            return
        if self.opacity_resets >= max_resets:
            return
        if self.block_index - self.last_opacity_reset_block < cooldown:
            return

        should_reset = False
        if controls.opacity_reset == "force_reset":
            should_reset = True
        elif controls.opacity_reset == "reset_if_plateau":
            plateau_eps = float(safety.get("plateau_psnr_epsilon", 0.02))
            should_reset = self.last_quality_gain <= plateau_eps

        if should_reset:
            self.gaussians.reset_opacity()
            self.opacity_resets += 1
            self.last_opacity_reset_block = self.block_index
            self.last_opacity_reset_iteration = self.iteration

    @torch.no_grad()
    def _render_camera_metrics(self, cameras: list) -> dict[str, list[float]]:
        per_view: dict[str, list[float]] = {"psnr": [], "ssim": [], "l1": []}
        for viewpoint in cameras:
            image = torch.clamp(
                self.backend.render_image(
                    viewpoint,
                    self.gaussians,
                    self.pipe,
                    self.background,
                    use_trained_exp=self.dataset.train_test_exp,
                ),
                0.0,
                1.0,
            )
            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
            per_view["l1"].append(float(self.backend.l1_loss(image, gt_image).item()))
            per_view["psnr"].append(float(self.backend.psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()))
            per_view["ssim"].append(float(self.backend.ssim(image, gt_image).item()))
        return per_view

    @torch.no_grad()
    def _evaluate_validation_subset(self) -> dict[str, float]:
        if not self.validation_cameras:
            return {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "psnr_min": 0.0, "quality": 0.0, "seconds": 0.0, "views": 0}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        per_view = self._render_camera_metrics(self.validation_cameras)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start

        reward_cfg = self.config.get("reward", {})
        ssim_weight = float(reward_cfg.get("ssim_quality_weight", 10.0))
        per_view_quality = [p + ssim_weight * s for p, s in zip(per_view["psnr"], per_view["ssim"])]
        mean_quality = float(np.mean(per_view_quality))
        min_quality = float(np.min(per_view_quality))
        # Mean quality lets the policy trade away poorly covered views, which the
        # (full-hemisphere) test cameras then expose. Blending in the worst view
        # makes the reward proxy robust to that view-selective optimization.
        min_weight = float(reward_cfg.get("min_view_weight", 0.0))
        quality = mean_quality + min_weight * (min_quality - mean_quality)
        return {
            "psnr": float(np.mean(per_view["psnr"])),
            "ssim": float(np.mean(per_view["ssim"])),
            "l1": float(np.mean(per_view["l1"])),
            "psnr_min": float(np.min(per_view["psnr"])),
            "quality": quality,
            "seconds": seconds,
            "views": len(per_view["psnr"]),
        }

    @torch.no_grad()
    def evaluate_test_cameras(self, max_views: int = 0) -> dict[str, float]:
        """Mean metrics over the dataset test cameras.

        For checkpoint selection and reporting ONLY. This is never called from
        reset()/step(), so test cameras still never influence policy state or
        reward construction. With max_views > 0 an evenly spaced deterministic
        subset is used (fast mid-episode checkpoints for acceleration scoring).
        """
        cameras = list(self.scene.getTestCameras()) if self.scene is not None else []
        if max_views > 0 and len(cameras) > max_views:
            step = max(1, len(cameras) // max_views)
            cameras = cameras[::step][:max_views]
        if not cameras:
            return {"psnr": 0.0, "ssim": 0.0, "l1": 1.0, "views": 0}
        per_view = self._render_camera_metrics(cameras)
        return {
            "psnr": float(np.mean(per_view["psnr"])),
            "ssim": float(np.mean(per_view["ssim"])),
            "l1": float(np.mean(per_view["l1"])),
            "views": len(per_view["psnr"]),
        }

    def _compute_reward(self, validation: dict[str, float], block_stats: dict[str, float]) -> tuple[float, dict[str, float]]:
        reward_cfg = self.config.get("reward", {})
        quality_gain = validation["quality"] - self.prev_validation_quality
        psnr_gain = validation["psnr"] - self.last_validation["psnr"]
        ssim_gain = validation["ssim"] - self.last_validation["ssim"]

        # --- model-complexity reference and discount ---------------------------
        n_now = float(block_stats["end_gaussians"])
        n_ref = self._reference_gaussian_count()
        excess = max(0.0, n_now / max(1.0, n_ref) - 1.0)
        kappa = float(reward_cfg.get("complexity_discount_kappa", 1.0))
        complexity_discount = 1.0 / (1.0 + kappa * excess)

        # --- quality credit: positive gains are discounted when over budget,
        #     drops keep full magnitude (handled again by instability term) -----
        pos_gain = max(0.0, quality_gain)
        neg_gain = min(0.0, quality_gain)
        quality_term = float(reward_cfg.get("quality_gain_weight", 1.0)) * (
            pos_gain * complexity_discount + neg_gain
        )

        # --- time value of progress (acceleration objective) --------------------
        # Plain gain telescopes to (final - initial) quality, which is indifferent
        # to WHEN quality is earned and so cannot prefer a faster-converging
        # schedule. Weighting each gain by exp(-train_seconds/tau) makes early
        # progress worth more — the policy maximizes the early area under the
        # quality-vs-time curve, which is what time-to-target measures. Stopping
        # emerges naturally once discounted marginal gains drop below time cost.
        tau = float(reward_cfg.get("time_value_tau_seconds", 0.0))
        time_value = 1.0
        if tau > 0.0:
            time_value = float(np.exp(-self.training_seconds / tau))
            quality_term *= time_value

        # --- convergence / compute cost (quality-per-time) ---------------------
        t_ref = max(1e-6, float(reward_cfg.get("block_time_ref_seconds", 1.0)))
        time_penalty = float(reward_cfg.get("time_penalty_weight", 0.05)) * (
            block_stats["block_seconds"] / t_ref
        )

        # --- explicit over-baseline count penalty ------------------------------
        count_penalty = float(reward_cfg.get("count_excess_penalty_weight", 0.3)) * excess

        # --- growth penalty / compaction bonus, gated to the over-budget range --
        # Below N_ref both terms are neutral: the objective is complexity <=
        # budget, not minimal count. Paying for pruning at any level collapses
        # the policy to the safety floor (v1/v3 runs); penalizing growth at any
        # level blocks recovery from that floor. Only the over-budget portion
        # of a change is priced.
        start_n = float(block_stats["start_gaussians"])
        growth_above_ref = max(0.0, n_now - max(start_n, n_ref))
        growth_penalty = float(reward_cfg.get("gaussian_growth_penalty_weight", 1.0)) * (
            growth_above_ref / max(1.0, n_ref)
        )
        tol = float(reward_cfg.get("compaction_quality_tolerance", 0.0))
        compaction_bonus = 0.0
        if quality_gain >= -tol:
            reduction_above_ref = max(0.0, start_n - max(n_now, n_ref))
            compaction_bonus = float(reward_cfg.get("compaction_bonus_weight", 1.0)) * (
                reduction_above_ref / max(1.0, n_ref)
            )

        mem = _memory_gb()
        target_vram = float(self.config.get("safety", {}).get("target_vram_gb", 20.0))
        vram_penalty = float(reward_cfg.get("vram_penalty_weight", 0.01)) * max(0.0, mem["peak"] - target_vram)
        instability_penalty = float(reward_cfg.get("validation_drop_penalty_weight", 0.1)) * max(0.0, -quality_gain)

        terminal_bonus = 0.0
        if self.iteration >= int(self.opt.iterations):
            terminal_bonus = float(reward_cfg.get("terminal_quality_weight", 0.1)) * (
                validation["quality"] - self.initial_validation_quality
            ) * complexity_discount * time_value
            terminal_bonus -= float(reward_cfg.get("terminal_count_penalty_weight", 0.5)) * excess

        reward = (
            quality_term
            - time_penalty
            - count_penalty
            - growth_penalty
            + compaction_bonus
            - vram_penalty
            - instability_penalty
            + terminal_bonus
        )
        terms = {
            "quality_gain": quality_gain,
            "psnr_gain": psnr_gain,
            "ssim_gain": ssim_gain,
            "time_value": time_value,
            "complexity_discount": complexity_discount,
            "count_excess": excess,
            "reference_gaussian_count": n_ref,
            "time_penalty": -time_penalty,
            "count_excess_penalty": -count_penalty,
            "gaussian_growth_penalty": -growth_penalty,
            "compaction_bonus": compaction_bonus,
            "vram_penalty": -vram_penalty,
            "instability_penalty": -instability_penalty,
            "terminal_quality_bonus": terminal_bonus,
        }
        return reward, terms

    def _is_done(self, controls, block_stats: dict[str, float]) -> bool:
        if self.iteration >= int(self.opt.iterations):
            return True
        max_wall = float(self.config.get("max_wall_seconds", 0.0))
        if max_wall > 0 and self.elapsed_seconds >= max_wall:
            return True
        if block_stats["numerical_failure"] or block_stats["hard_budget_exceeded"]:
            return True
        min_stop_iter = int(self.config.get("safety", {}).get("min_iterations_before_stop", 500))
        return controls.stop == "stop" and self.iteration >= min_stop_iter

    def _hard_budget_exceeded(self) -> bool:
        safety = self.config.get("safety", {})
        max_gaussians = int(safety.get("hard_max_gaussians", 3_000_000))
        if int(self.gaussians.get_xyz.shape[0]) > max_gaussians:
            return True
        max_vram = float(safety.get("hard_max_vram_gb", 24.0))
        return _memory_gb()["peak"] > max_vram

    def _build_observation(self) -> np.ndarray:
        max_iter = max(1, int(self.config.get("max_episode_iterations", 7000)))
        max_wall = max(1e-6, float(self.config.get("max_wall_seconds", 1.0)))
        hard_max_gaussians = max(1, int(self.config.get("safety", {}).get("hard_max_gaussians", 3_000_000)))
        target_vram = max(1e-6, float(self.config.get("safety", {}).get("target_vram_gb", 20.0)))
        block_stats = self.last_block_stats
        mem = _memory_gb()
        gstats = self._gaussian_stats()
        loss_slope = 0.0
        if len(self.loss_history) >= 2:
            loss_slope = self.loss_history[-1] - self.loss_history[-2]

        obs = [
            self.iteration / max_iter,
            self.elapsed_seconds / max_wall,
            max(0.0, 1.0 - self.iteration / max_iter),
            self.block_index / max(1.0, max_iter / 100.0),
            block_stats["l1"],
            block_stats["dssim"],
            block_stats["total_loss"],
            0.5 + 0.5 * np.tanh(loss_slope),
            self.last_validation["psnr"] / 50.0,
            self.last_validation["ssim"],
            0.0,
            0.5 + 0.5 * np.tanh(self.last_validation["quality"] - self.prev_validation_quality),
            block_stats["time_per_iteration"] / 0.2,
            (self.last_validation.get("seconds", 0.0) / max(1, self.last_validation.get("views", 1))) / 0.5,
            mem["peak"] / target_vram,
            mem["allocated"] / target_vram,
            1.0 / max(1.0, float(self.config.get("resolution", 1))),
            gstats["count"] / hard_max_gaussians,
            max(0.0, block_stats["gaussian_growth"]) / hard_max_gaussians,
            gstats["count"] / hard_max_gaussians,
            *action_to_observation_values(self.prev_action),
            1.0 if self.prev_action_improved else 0.0,
            gstats["opacity_q10"],
            gstats["opacity_q50"],
            gstats["opacity_q90"],
            gstats["near_transparent_fraction"],
            gstats["scale_q50"],
            gstats["scale_q90"],
            gstats["anisotropy_q50"],
            block_stats["last_visible_fraction"],
            block_stats["gaussians_added"] / hard_max_gaussians,
            block_stats["gaussians_pruned"] / hard_max_gaussians,
            gstats["active_sh_degree_fraction"],
        ]
        obs = np.asarray([_clip01(float(v)) for v in obs], dtype=np.float32)
        if obs.shape[0] != len(OBSERVATION_NAMES):
            raise RuntimeError(f"Observation length mismatch: {obs.shape[0]} != {len(OBSERVATION_NAMES)}")
        return obs

    @torch.no_grad()
    def _gaussian_stats(self) -> dict[str, float]:
        if self.gaussians is None:
            return {
                "count": 0.0,
                "opacity_q10": 0.0,
                "opacity_q50": 0.0,
                "opacity_q90": 0.0,
                "near_transparent_fraction": 0.0,
                "scale_q50": 0.0,
                "scale_q90": 0.0,
                "anisotropy_q50": 0.0,
                "active_sh_degree_fraction": 0.0,
            }
        opacity = self.gaussians.get_opacity.detach().flatten().float()
        scaling = self.gaussians.get_scaling.detach().float()
        max_scale = scaling.max(dim=1).values
        min_scale = torch.clamp(scaling.min(dim=1).values, min=1e-8)
        anisotropy = max_scale / min_scale
        extent = max(1e-6, float(self.scene.cameras_extent))

        def q(tensor, value):
            if tensor.numel() == 0:
                return 0.0
            return float(torch.quantile(tensor, value).item())

        return {
            "count": float(self.gaussians.get_xyz.shape[0]),
            "opacity_q10": q(opacity, 0.10),
            "opacity_q50": q(opacity, 0.50),
            "opacity_q90": q(opacity, 0.90),
            "near_transparent_fraction": float((opacity < 0.01).float().mean().item()) if opacity.numel() else 0.0,
            "scale_q50": min(q(max_scale / extent, 0.50), 1.0),
            "scale_q90": min(q(max_scale / extent, 0.90), 1.0),
            "anisotropy_q50": min(np.log1p(q(anisotropy, 0.50)) / 5.0, 1.0),
            "active_sh_degree_fraction": self.gaussians.active_sh_degree / max(1, self.gaussians.max_sh_degree),
        }

    def _empty_block_stats(self) -> dict[str, float]:
        return {
            "start_iteration": 0,
            "iterations_ran": 0,
            "l1": 0.0,
            "dssim": 0.0,
            "total_loss": 0.0,
            "visible_gaussians": 0.0,
            "last_visible_fraction": 0.0,
            "densify_events": 0.0,
            "gaussians_added": 0.0,
            "gaussians_pruned": 0.0,
            "start_gaussians": 0.0,
            "end_gaussians": 0.0,
            "gaussian_growth": 0.0,
            "block_seconds": 0.0,
            "time_per_iteration": 0.0,
            "numerical_failure": 0.0,
            "hard_budget_exceeded": 0.0,
        }

    def _init_block_log(self) -> None:
        fields = [
            "scene",
            "block",
            "iteration",
            "reward",
            "validation_psnr",
            "validation_psnr_min",
            "validation_ssim",
            "validation_l1",
            "quality_gain",
            "complexity_discount",
            "count_excess",
            "reference_gaussian_count",
            "time_penalty",
            "count_excess_penalty",
            "gaussian_growth_penalty",
            "compaction_bonus",
            "vram_penalty",
            "instability_penalty",
            "terminal_quality_bonus",
            "block_seconds",
            "validation_seconds",
            "elapsed_seconds",
            "time_per_iteration",
            "gaussian_count",
            "gaussians_added",
            "gaussians_pruned",
            "peak_vram_gb",
            "action_json",
        ]
        with self.block_log_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fields).writeheader()

    def _append_block_log(self, info: dict[str, Any], reward: float) -> None:
        reward_terms = info["reward_terms"]
        block_stats = info["block_stats"]
        validation = info["validation"]
        row = {
            "scene": info["scene"],
            "block": info["block_index"],
            "iteration": info["iteration"],
            "reward": reward,
            "validation_psnr": validation["psnr"],
            "validation_psnr_min": validation.get("psnr_min", 0.0),
            "validation_ssim": validation["ssim"],
            "validation_l1": validation["l1"],
            "quality_gain": reward_terms.get("quality_gain", 0.0),
            "complexity_discount": reward_terms.get("complexity_discount", 1.0),
            "count_excess": reward_terms.get("count_excess", 0.0),
            "reference_gaussian_count": reward_terms.get("reference_gaussian_count", 0.0),
            "time_penalty": reward_terms.get("time_penalty", 0.0),
            "count_excess_penalty": reward_terms.get("count_excess_penalty", 0.0),
            "gaussian_growth_penalty": reward_terms.get("gaussian_growth_penalty", 0.0),
            "compaction_bonus": reward_terms.get("compaction_bonus", 0.0),
            "vram_penalty": reward_terms.get("vram_penalty", 0.0),
            "instability_penalty": reward_terms.get("instability_penalty", 0.0),
            "terminal_quality_bonus": reward_terms.get("terminal_quality_bonus", 0.0),
            "block_seconds": block_stats["block_seconds"],
            "validation_seconds": validation.get("seconds", 0.0),
            "elapsed_seconds": self.elapsed_seconds,
            "time_per_iteration": block_stats["time_per_iteration"],
            "gaussian_count": block_stats["end_gaussians"],
            "gaussians_added": block_stats["gaussians_added"],
            "gaussians_pruned": block_stats["gaussians_pruned"],
            "peak_vram_gb": _memory_gb()["peak"],
            "action_json": json.dumps(info["action"], sort_keys=True),
        }
        with self.block_log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writerow(row)

    def _write_episode_metadata(self) -> None:
        metadata = {
            "scene": self.scene_id,
            "model_path": str(self.model_path),
            "train_camera_count": len(self.train_cameras),
            "validation_camera_count": len(self.validation_cameras),
            "train_camera_names": [camera.image_name for camera in self.train_cameras],
            "validation_camera_names": [camera.image_name for camera in self.validation_cameras],
            "test_camera_accessed_for_reward_or_state": False,
            "observation_names": OBSERVATION_NAMES,
            "config": self.config,
        }
        with (self.model_path / "agentic_episode_metadata.json").open("w") as f:
            json.dump(metadata, f, indent=2)
        cfg_args = {
            "sh_degree": self.dataset.sh_degree,
            "source_path": self.dataset.source_path,
            "model_path": self.dataset.model_path,
            "images": self.dataset.images,
            "depths": self.dataset.depths,
            "resolution": self.dataset.resolution,
            "white_background": self.dataset.white_background,
            "train_test_exp": self.dataset.train_test_exp,
            "data_device": self.dataset.data_device,
            "eval": self.dataset.eval,
            "convert_SHs_python": self.pipe.convert_SHs_python,
            "compute_cov3D_python": self.pipe.compute_cov3D_python,
            "debug": self.pipe.debug,
            "antialiasing": self.pipe.antialiasing,
        }
        namespace_items = ", ".join(f"{key}={repr(value)}" for key, value in cfg_args.items())
        with (self.model_path / "cfg_args").open("w", encoding="utf-8") as f:
            f.write(f"Namespace({namespace_items})")
