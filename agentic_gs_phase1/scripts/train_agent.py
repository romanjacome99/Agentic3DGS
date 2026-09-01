from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agentic_gs_phase1.envs import AgenticGSEnv
from agentic_gs_phase1.envs.spaces import (
    CONTINUOUS_ACTIONS,
    DISCRETE_ACTIONS,
    discrete_actions_for,
    observation_names_for,
)
from agentic_gs_phase1.policies import ActorCritic, PPOConfig, ppo_update


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunningMeanStd:
    """Online Welford mean/variance estimator for scalar streams."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.m2 = 0.0
        self.count = 0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float((self.m2 / self.count) ** 0.5)


class RewardNormalizer:
    """Per-scene reward standardization.

    Stabilizes PPO when scenes have different natural PSNR scales and keeps
    return magnitudes O(1) so the value function can actually fit them. With
    mode='none' rewards pass through unchanged.
    """

    def __init__(self, mode: str = "per_scene_zscore", std_floor: float = 1e-2) -> None:
        self.mode = mode
        self.std_floor = float(std_floor)
        self.scenes: dict[str, RunningMeanStd] = {}

    def normalize(self, scene: str, reward: float) -> float:
        if self.mode == "none":
            return reward
        rms = self.scenes.setdefault(scene, RunningMeanStd())
        rms.update(reward)
        return (reward - rms.mean) / max(rms.std, self.std_floor)


def ppo_config_from_dict(config: dict[str, Any]) -> PPOConfig:
    ppo = config.get("ppo", {})
    norm = config.get("normalization", {})
    return PPOConfig(
        gamma=float(ppo.get("gamma", 0.99)),
        gae_lambda=float(ppo.get("gae_lambda", 0.95)),
        clip_ratio=float(ppo.get("clip_ratio", 0.2)),
        value_weight=float(ppo.get("value_weight", 0.5)),
        entropy_weight=float(ppo.get("entropy_weight", 0.003)),
        learning_rate=float(ppo.get("learning_rate", 3e-4)),
        epochs=int(ppo.get("epochs", 4)),
        minibatch_size=int(ppo.get("minibatch_size", 64)),
        max_grad_norm=float(ppo.get("max_grad_norm", 0.5)),
        target_kl=float(ppo.get("target_kl", 0.03)),
        value_clip=float(norm.get("value_clip", ppo.get("value_clip", 0.0))),
        normalize_advantage=bool(norm.get("normalize_advantage", True)),
    )


def make_policy(config: dict[str, Any], obs_dim: int, device: torch.device) -> ActorCritic:
    policy_cfg = config.get("policy", {})
    policy = ActorCritic(
        obs_dim=obs_dim,
        hidden_width=int(policy_cfg.get("hidden_width", 256)),
        hidden_layers=int(policy_cfg.get("hidden_layers", 3)),
        activation=str(policy_cfg.get("activation", "gelu")),
        config=config,
    )
    return policy.to(device)


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros((), device=rewards.device)
    for t in reversed(range(rewards.shape[0])):
        next_nonterminal = 1.0 - dones[t]
        next_value = last_value if t == rewards.shape[0] - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * gae_lambda * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


def append_update_log(path: Path, row: dict[str, Any]) -> None:
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_checkpoint(
    path: Path,
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    update: int,
    obs_dim: int,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "policy_state_dict": policy.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "update": update,
        "obs_dim": obs_dim,
        "discrete_actions": dict(discrete_actions_for(config)),
        "continuous_actions": dict(CONTINUOUS_ACTIONS),
        "fastergs_policy_ext": bool((config or {}).get("fastergs_policy_ext", False)),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def run_probe_episode(
    env: AgenticGSEnv,
    policy: ActorCritic,
    scene: str,
    device: torch.device,
    episode_id: int,
    accel_checkpoints: list[float] | None = None,
    accel_test_views: int = 30,
) -> dict[str, Any]:
    """Frozen deterministic rollout on a held-out scene, scored on TEST cameras.

    The test-camera metric is used for checkpoint selection only — it never
    enters the reward or the observation, so the train/reward protocol is
    unchanged. This closes the train-cam/test-cam selection gap documented in
    the v3(60u) > v4(150u) regression: PPO-health metrics and train-cam reward
    kept improving while test PSNR fell, so checkpoints must be picked by a
    test-camera probe, not by update count.

    With accel_checkpoints set (training-wall-clock seconds), test PSNR is also
    sampled on a fast test-camera subset whenever the cumulative training time
    crosses a checkpoint; their mean (`accel_score`) approximates the early area
    under the quality-vs-time curve — the time-to-target objective. Checkpoints
    not reached before the policy stops are scored with the final model, since
    a stopped model's quality is constant from then on.
    """
    checkpoints = sorted(accel_checkpoints or [])
    checkpoint_psnrs: list[float] = []
    next_cp = 0
    start = time.perf_counter()
    # Budget-conditioned policies: pin the probe to the max budget so the agent
    # is in its full-budget regime and all accel checkpoints are reachable.
    if getattr(env, "budget_conditioned", False):
        env.budget_override = env.budget_max_seconds
    obs = env.reset(scene, episode_id=episode_id)
    done = False
    info: dict[str, Any] = {}
    blocks = 0
    while not done:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
        action, _, _ = policy.act(obs_tensor, deterministic=True)
        obs, _, done, info = env.step(action)
        blocks += 1
        while next_cp < len(checkpoints) and env.training_seconds >= checkpoints[next_cp]:
            checkpoint_psnrs.append(float(env.evaluate_test_cameras(max_views=accel_test_views)["psnr"]))
            next_cp += 1
        # Speedup: once every accel checkpoint is captured, training further is
        # pure waste for accel_auc selection. The budget stays pinned to max, so
        # the policy's action trajectory up to the last checkpoint (and thus every
        # captured PSNR / accel_score) is identical to running the full budget --
        # we just stop stepping. Cuts probe wall-time from ~budget_max to ~last
        # checkpoint (e.g. 300s -> ~60s). No effect when scoring by test_psnr
        # (checkpoints empty -> condition never triggers).
        if checkpoints and next_cp >= len(checkpoints):
            break
    while next_cp < len(checkpoints):
        checkpoint_psnrs.append(float(env.evaluate_test_cameras(max_views=accel_test_views)["psnr"]))
        next_cp += 1
    test_metrics = env.evaluate_test_cameras()
    result = {
        "scene": scene,
        "blocks": blocks,
        "iterations": int(info.get("iteration", 0)),
        "wall_seconds": time.perf_counter() - start,
        "train_seconds": float(env.training_seconds),
        "gaussian_count": int(info.get("block_stats", {}).get("end_gaussians", 0)),
        "val_psnr": float(info.get("validation", {}).get("psnr", 0.0)),
        "val_ssim": float(info.get("validation", {}).get("ssim", 0.0)),
        "test_psnr": float(test_metrics["psnr"]),
        "test_ssim": float(test_metrics["ssim"]),
        "test_views": int(test_metrics["views"]),
    }
    for cp, psnr_value in zip(checkpoints, checkpoint_psnrs):
        result[f"test_psnr_at_{int(cp)}s"] = psnr_value
    result["accel_score"] = float(np.mean(checkpoint_psnrs)) if checkpoint_psnrs else 0.0
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Phase 1 PPO policy for agentic 3DGS control.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "agentic_gs_phase1" / "configs" / "phase1_mvp.json")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-updates", type=int, default=None)
    parser.add_argument("--max-episode-iterations", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--train-scenes", nargs="+", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--check-config", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.output_root is not None:
        config["output_root"] = str(args.output_root)
    if args.max_episode_iterations is not None:
        config["max_episode_iterations"] = int(args.max_episode_iterations)
    if args.rollout_steps is not None:
        config.setdefault("ppo", {})["rollout_steps"] = int(args.rollout_steps)
    if args.max_updates is not None:
        config.setdefault("ppo", {})["max_updates"] = int(args.max_updates)
    if args.train_scenes is not None:
        config["train_scenes"] = list(args.train_scenes)

    seed = int(config.get("seed", 0))
    set_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("The 3DGS RL environment requires CUDA.")
    device = torch.device("cuda")

    run_name = args.run_name or time.strftime("ppo_%Y%m%d_%H%M%S")
    output_root = Path(config.get("output_root", "outputs/agentic_gs_phase1"))
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "resolved_config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    obs_dim = len(observation_names_for(config))
    policy = make_policy(config, obs_dim, device)
    ppo_cfg = ppo_config_from_dict(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=ppo_cfg.learning_rate)

    start_update = 0
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        policy.load_state_dict(ckpt["policy_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_update = int(ckpt.get("update", 0)) + 1

    if args.check_config:
        print(f"Config OK. obs_dim={obs_dim}, train_scenes={config.get('train_scenes')}")
        return 0

    train_scenes = list(config.get("train_scenes", ["chair", "drums", "mic"]))
    # Optional seeded shuffle of the training pool. With sequential round-robin
    # selection, a scene-grouped pool (e.g. augmentations listed consecutively)
    # combined with a small update budget starves later scenes. Shuffling makes
    # coverage order-independent: the pool is shuffled once at load and reshuffled
    # after every full pass. Off by default so existing configs are unaffected.
    shuffle_scenes = bool(config.get("shuffle_scenes", False))
    scene_rng = random.Random(int(config.get("seed", 0)))
    if shuffle_scenes:
        scene_rng.shuffle(train_scenes)
    rollout_steps = int(config.get("ppo", {}).get("rollout_steps", 32))
    max_updates = int(config.get("ppo", {}).get("max_updates", 10))
    save_every = int(config.get("ppo", {}).get("save_every_updates", 1))
    save_episode_models = bool(config.get("save_episode_models", True))

    env = AgenticGSEnv(config, run_dir=run_dir, seed=seed)
    scene_cursor = 0
    episode_id = 0
    obs = env.reset(train_scenes[scene_cursor], episode_id=episode_id)
    needs_reset = False
    update_log = run_dir / "ppo_updates.csv"

    norm_cfg = config.get("normalization", {})
    reward_normalizer = RewardNormalizer(
        mode=str(norm_cfg.get("reward", "per_scene_zscore")),
        std_floor=float(norm_cfg.get("reward_std_floor", 1e-2)),
    )

    selection_cfg = config.get("selection", {})
    probe_scene = selection_cfg.get("probe_scene")
    probe_every = int(selection_cfg.get("eval_every_updates", 0))
    probe_enabled = bool(probe_scene) and probe_every > 0
    if probe_enabled and probe_scene in train_scenes:
        raise ValueError(
            f"selection.probe_scene '{probe_scene}' is a training scene; "
            "checkpoint selection needs a held-out scene."
        )
    if probe_enabled and probe_scene in config.get("eval_scenes", []):
        print(
            f"WARNING: probe scene '{probe_scene}' is also an eval scene; "
            "checkpoint selection on it leaks into the final evaluation."
        )
    probe_metric = str(selection_cfg.get("probe_metric", "final_psnr"))
    accel_checkpoints = [float(v) for v in selection_cfg.get("probe_test_checkpoints_seconds", [15.0, 30.0, 60.0])]
    accel_test_views = int(selection_cfg.get("probe_test_views", 30))
    best_metric_value = float("-inf")
    probe_log = run_dir / "probe_eval.csv"

    print(
        f"Run '{run_name}' | scenes={train_scenes} | rollout_steps={rollout_steps} | "
        f"updates={max_updates} | reward_norm={reward_normalizer.mode} | "
        f"value_clip={ppo_cfg.value_clip} | norm_adv={ppo_cfg.normalize_advantage} | "
        f"probe={probe_scene if probe_enabled else 'off'}"
        + (f" every {probe_every}u" if probe_enabled else "")
    )

    for update in range(start_update, max_updates):
        if needs_reset:
            obs = env.reset(train_scenes[scene_cursor], episode_id=episode_id)
            needs_reset = False

        obs_items = []
        cont_items = []
        disc_items = {name: [] for name in policy.discrete_action_names}
        rewards = []          # normalized rewards fed to GAE/PPO
        raw_rewards = []      # raw env rewards, for interpretable logging
        dones = []
        values = []
        logprobs = []

        term_means: dict[str, list[float]] = {}
        count_ratios: list[float] = []
        discounts: list[float] = []
        episode_block_counts: list[int] = []
        episode_end_iters: list[int] = []
        blocks_in_episode = 0

        for step_idx in range(rollout_steps):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            action, logprob, value = policy.act(obs_tensor, deterministic=False)
            next_obs, reward, done, info = env.step(action)
            norm_reward = reward_normalizer.normalize(info["scene"], float(reward))

            obs_items.append(torch.as_tensor(obs, dtype=torch.float32))
            cont_items.append(action["continuous"].float())
            for name in policy.discrete_action_names:
                disc_items[name].append(int(action["discrete"][name]))
            rewards.append(float(norm_reward))
            raw_rewards.append(float(reward))
            dones.append(float(done))
            values.append(float(value.detach().cpu().item()))
            logprobs.append(float(logprob.detach().cpu().item()))

            # accumulate reward-term and model-complexity diagnostics
            for key, val in info["reward_terms"].items():
                term_means.setdefault(key, []).append(float(val))
            n_ref = max(1.0, float(info["reward_terms"].get("reference_gaussian_count", 1.0)))
            count_ratios.append(float(info["block_stats"]["end_gaussians"]) / n_ref)
            discounts.append(float(info["reward_terms"].get("complexity_discount", 1.0)))
            blocks_in_episode += 1

            obs = next_obs
            if done:
                episode_block_counts.append(blocks_in_episode)
                episode_end_iters.append(int(info["iteration"]))
                blocks_in_episode = 0
                if save_episode_models:
                    env.save_model()
                scene_cursor += 1
                if scene_cursor >= len(train_scenes):
                    scene_cursor = 0
                    if shuffle_scenes:
                        scene_rng.shuffle(train_scenes)  # reshuffle each full pass
                episode_id += 1
                if step_idx == rollout_steps - 1:
                    needs_reset = True
                else:
                    obs = env.reset(train_scenes[scene_cursor], episode_id=episode_id)

        with torch.no_grad():
            if needs_reset:
                last_value = torch.zeros((), dtype=torch.float32, device=device)
            else:
                last_value = policy(torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0))[3].squeeze(0)

        batch = {
            "obs": torch.stack(obs_items).to(device),
            "continuous": torch.stack(cont_items).to(device),
            "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
            "dones": torch.as_tensor(dones, dtype=torch.float32, device=device),
            "values": torch.as_tensor(values, dtype=torch.float32, device=device),
            "logprob": torch.as_tensor(logprobs, dtype=torch.float32, device=device),
        }
        for name, values_for_name in disc_items.items():
            batch[f"disc_{name}"] = torch.as_tensor(values_for_name, dtype=torch.long, device=device)

        advantages, returns = compute_gae(
            batch["rewards"],
            batch["values"],
            batch["dones"],
            last_value,
            ppo_cfg.gamma,
            ppo_cfg.gae_lambda,
        )
        batch["advantages"] = advantages
        batch["returns"] = returns

        def _mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        def _median(values: list[float]) -> float:
            return float(np.median(values)) if values else 0.0

        # ---- rollout summary (printed before the inner PPO epochs) ----
        episodes_this_update = len(episode_block_counts)
        print(f"\n{'='*28} update {update} / {max_updates - 1} {'='*28}")
        print(
            f" rollout      : {rollout_steps} steps | {episodes_this_update} episodes done | "
            f"last scene={env.scene_id}"
        )
        print(
            f" reward       : raw mean={_mean(raw_rewards):+.4f} sum={float(np.sum(raw_rewards)):+.2f}  |  "
            f"norm mean={_mean(rewards):+.4f}"
        )
        print(
            " reward terms : "
            f"qual*d={_mean(term_means.get('quality_gain', [])):+.3f}  "
            f"count_pen={_mean(term_means.get('count_excess_penalty', [])):+.3f}  "
            f"growth_pen={_mean(term_means.get('gaussian_growth_penalty', [])):+.3f}  "
            f"compact={_mean(term_means.get('compaction_bonus', [])):+.3f}  "
            f"time={_mean(term_means.get('time_penalty', [])):+.3f}  "
            f"instab={_mean(term_means.get('instability_penalty', [])):+.3f}"
        )
        print(
            " model        : "
            f"count/N_ref med={_median(count_ratios):.2f}  "
            f"discount med={_median(discounts):.2f}  "
            f"blocks/ep={_mean([float(b) for b in episode_block_counts]):.1f}  "
            f"end_iter med={_median([float(i) for i in episode_end_iters]):.0f}"
        )
        print(" training     :")
        stats = ppo_update(policy, optimizer, batch, ppo_cfg, verbose=True)
        early = " (early-stop)" if stats.get("early_stopped", 0.0) > 0.5 else ""
        print(
            f" ppo          : epochs={int(stats['epochs_ran'])}/{ppo_cfg.epochs}{early}  "
            f"value_loss={stats['value_loss']:.4f}  expl_var={stats['explained_variance']:+.3f}  "
            f"entropy={stats['entropy']:.3f}  kl={stats['approx_kl']:.4f}  "
            f"clip={stats['clip_fraction']:.3f}  |g|={stats['grad_norm']:.3f}"
        )

        epoch_log = stats.pop("epoch_log", [])
        row = {
            "update": update,
            "episodes_completed": episode_id,
            "episodes_this_update": episodes_this_update,
            "raw_reward_mean": _mean(raw_rewards),
            "raw_reward_sum": float(np.sum(raw_rewards)),
            "norm_reward_mean": _mean(rewards),
            "count_ratio_med": _median(count_ratios),
            "discount_med": _median(discounts),
            "blocks_per_episode": _mean([float(b) for b in episode_block_counts]),
            "end_iter_med": _median([float(i) for i in episode_end_iters]),
            "term_quality_gain": _mean(term_means.get("quality_gain", [])),
            "term_count_excess_penalty": _mean(term_means.get("count_excess_penalty", [])),
            "term_growth_penalty": _mean(term_means.get("gaussian_growth_penalty", [])),
            "term_compaction_bonus": _mean(term_means.get("compaction_bonus", [])),
            "term_time_penalty": _mean(term_means.get("time_penalty", [])),
            "term_instability_penalty": _mean(term_means.get("instability_penalty", [])),
            "last_scene": env.scene_id,
            "last_iteration": env.iteration,
            **{k: v for k, v in stats.items() if not isinstance(v, list)},
        }
        append_update_log(update_log, row)

        if save_every > 0 and (update + 1) % save_every == 0:
            save_checkpoint(run_dir / "checkpoints" / f"policy_update_{update:04d}.pth", policy, optimizer, config, update, obs_dim)
            save_checkpoint(run_dir / "checkpoints" / "latest.pth", policy, optimizer, config, update, obs_dim)

        if probe_enabled and (update + 1) % probe_every == 0:
            print(f" probe        : held-out scene '{probe_scene}' (test cameras, checkpoint selection only)")
            probe = run_probe_episode(
                env,
                policy,
                probe_scene,
                device,
                episode_id=90_000 + update,
                accel_checkpoints=accel_checkpoints if probe_metric == "accel_auc" else None,
                accel_test_views=accel_test_views,
            )
            metric_value = probe["accel_score"] if probe_metric == "accel_auc" else probe["test_psnr"]
            is_best = metric_value > best_metric_value
            if is_best:
                best_metric_value = metric_value
                save_checkpoint(
                    run_dir / "checkpoints" / "best.pth",
                    policy,
                    optimizer,
                    config,
                    update,
                    obs_dim,
                    extra={"probe": probe, "selection_metric": probe_metric},
                )
            append_update_log(probe_log, {"update": update, "is_best": int(is_best), **probe})
            cp_summary = "  ".join(
                f"{key.replace('test_psnr_at_', '@')}={probe[key]:.2f}"
                for key in sorted(probe)
                if key.startswith("test_psnr_at_")
            )
            print(
                f" probe        : {probe_metric}={metric_value:.3f} (best={best_metric_value:.3f}"
                f"{', new best -> best.pth' if is_best else ''})  {cp_summary}  "
                f"final_test={probe['test_psnr']:.3f}  val={probe['val_psnr']:.3f}  "
                f"gaussians={probe['gaussian_count']}  iters={probe['iterations']}  "
                f"{probe['wall_seconds']:.0f}s"
            )
            # The probe reset destroyed the in-flight training episode; start a
            # fresh one (new episode id so the aborted episode's logs survive).
            episode_id += 1
            needs_reset = True

    save_checkpoint(run_dir / "checkpoints" / "final.pth", policy, optimizer, config, max_updates - 1, obs_dim)
    env.close()
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
