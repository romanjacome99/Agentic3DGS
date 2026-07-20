from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical, Normal

from agentic_gs_phase1.envs.spaces import (
    CONTINUOUS_ACTIONS,
    DISCRETE_ACTIONS,
    continuous_actions_for,
    discrete_actions_for,
)


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_weight: float = 0.5
    entropy_weight: float = 0.003
    learning_rate: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 64
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    value_clip: float = 0.0
    normalize_advantage: bool = True


def _activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    return nn.ReLU()


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden_width: int = 256, hidden_layers: int = 3, activation: str = "gelu",
                 config: dict | None = None):
        super().__init__()
        # Config-aware action space: base 3DGS when config is None / flag off (identical
        # to before), extended with the FasterGS-specific heads when enabled.
        disc = discrete_actions_for(config)
        cont = continuous_actions_for(config)
        self.discrete_action_names = list(disc.keys())
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_width))
            layers.append(_activation(activation))
            in_dim = hidden_width
        self.encoder = nn.Sequential(*layers)
        self.discrete_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_width, len(choices)) for name, choices in disc.items()}
        )
        self.continuous_mean = nn.Linear(hidden_width, len(cont))
        self.continuous_log_std = nn.Parameter(torch.full((len(cont),), -0.5))
        self.value_head = nn.Linear(hidden_width, 1)

    def forward(self, obs: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(obs)
        logits = {name: head(features) for name, head in self.discrete_heads.items()}
        cont_mean = self.continuous_mean(features)
        cont_log_std = self.continuous_log_std.expand_as(cont_mean).clamp(-5.0, 2.0)
        value = self.value_head(features).squeeze(-1)
        return logits, cont_mean, cont_log_std, value

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        logits, cont_mean, cont_log_std, value = self.forward(obs)
        discrete: dict[str, torch.Tensor] = {}
        logprob = torch.zeros(obs.shape[0], device=obs.device)
        for name, head_logits in logits.items():
            dist = Categorical(logits=head_logits)
            sample = torch.argmax(head_logits, dim=-1) if deterministic else dist.sample()
            discrete[name] = sample
            logprob = logprob + dist.log_prob(sample)

        std = cont_log_std.exp()
        cont_dist = Normal(cont_mean, std)
        continuous = cont_mean if deterministic else cont_dist.sample()
        logprob = logprob + cont_dist.log_prob(continuous).sum(dim=-1)

        action = {
            "discrete": {name: int(sample[0].item()) for name, sample in discrete.items()},
            "continuous": continuous[0].detach().cpu(),
        }
        return action, logprob.squeeze(0), value.squeeze(0)

    def evaluate_actions(
        self,
        obs: torch.Tensor,
        discrete_actions: dict[str, torch.Tensor],
        continuous_actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits, cont_mean, cont_log_std, values = self.forward(obs)
        logprob = torch.zeros(obs.shape[0], device=obs.device)
        entropy = torch.zeros(obs.shape[0], device=obs.device)
        for name, head_logits in logits.items():
            dist = Categorical(logits=head_logits)
            action = discrete_actions[name].long()
            logprob = logprob + dist.log_prob(action)
            entropy = entropy + dist.entropy()

        cont_dist = Normal(cont_mean, cont_log_std.exp())
        logprob = logprob + cont_dist.log_prob(continuous_actions).sum(dim=-1)
        entropy = entropy + cont_dist.entropy().sum(dim=-1)
        return logprob, entropy, values


def ppo_update(
    policy: ActorCritic,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    config: PPOConfig,
    verbose: bool = False,
) -> dict[str, float]:
    obs = batch["obs"]
    continuous = batch["continuous"]
    old_logprob = batch["logprob"]
    advantages = batch["advantages"]
    returns = batch["returns"]
    old_values = batch["values"]
    discrete = {name: batch[f"disc_{name}"] for name in policy.discrete_action_names}

    if config.normalize_advantage:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    batch_size = obs.shape[0]
    minibatch_size = min(config.minibatch_size, batch_size)

    agg = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
           "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0}
    n_minibatches = 0.0
    epoch_log: list[dict[str, float]] = []
    early_stopped = False
    epochs_ran = 0

    for epoch in range(config.epochs):
        epochs_ran = epoch + 1
        ep = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
              "approx_kl": 0.0, "clip_fraction": 0.0, "grad_norm": 0.0}
        ep_count = 0
        indices = torch.randperm(batch_size, device=obs.device)
        for start in range(0, batch_size, minibatch_size):
            mb_idx = indices[start : start + minibatch_size]
            mb_discrete = {name: value[mb_idx] for name, value in discrete.items()}
            new_logprob, entropy, values = policy.evaluate_actions(obs[mb_idx], mb_discrete, continuous[mb_idx])
            ratio = torch.exp(new_logprob - old_logprob[mb_idx])
            adv = advantages[mb_idx]
            policy_loss = -torch.min(
                ratio * adv,
                torch.clamp(ratio, 1.0 - config.clip_ratio, 1.0 + config.clip_ratio) * adv,
            ).mean()

            # Value loss, optionally clipped to the rollout value estimate.
            if config.value_clip > 0.0:
                v_clipped = old_values[mb_idx] + torch.clamp(
                    values - old_values[mb_idx], -config.value_clip, config.value_clip
                )
                value_loss = 0.5 * torch.max(
                    torch.square(returns[mb_idx] - values),
                    torch.square(returns[mb_idx] - v_clipped),
                ).mean()
            else:
                value_loss = 0.5 * torch.square(returns[mb_idx] - values).mean()

            entropy_loss = entropy.mean()
            loss = policy_loss + config.value_weight * value_loss - config.entropy_weight * entropy_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = nn.utils.clip_grad_norm_(policy.parameters(), config.max_grad_norm)
            optimizer.step()

            with torch.no_grad():
                log_ratio = new_logprob - old_logprob[mb_idx]
                # Schulman's low-variance, non-negative KL estimator.
                approx_kl = torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio)
                clip_fraction = (torch.abs(ratio - 1.0) > config.clip_ratio).float().mean()

            for key, val in (
                ("policy_loss", policy_loss), ("value_loss", value_loss),
                ("entropy", entropy_loss), ("approx_kl", approx_kl),
                ("clip_fraction", clip_fraction), ("grad_norm", grad_norm),
            ):
                ep[key] += float(val.item())
            ep_count += 1

        for key in ep:
            ep[key] /= max(1, ep_count)
            agg[key] += ep[key] * ep_count
        n_minibatches += ep_count
        epoch_log.append({"epoch": epochs_ran, **ep})
        if verbose:
            print(
                f"    epoch {epochs_ran:>2d}/{config.epochs}: "
                f"pi={ep['policy_loss']:+.4f}  v={ep['value_loss']:.4f}  "
                f"ent={ep['entropy']:.3f}  kl={ep['approx_kl']:.4f}  "
                f"clip={ep['clip_fraction']:.3f}  |g|={ep['grad_norm']:.3f}"
            )
        if ep["approx_kl"] > config.target_kl:
            early_stopped = True
            if verbose:
                print(f"    [KL {ep['approx_kl']:.4f} > target {config.target_kl:.4f} -> early stop after epoch {epochs_ran}]")
            break

    denom = max(1.0, n_minibatches)
    stats = {key: agg[key] / denom for key in agg}

    # Explained variance of the critic over the full batch (1.0 = perfect fit).
    with torch.no_grad():
        _, _, values_all = policy.evaluate_actions(obs, discrete, continuous)
        var_returns = torch.var(returns, unbiased=False)
        explained_variance = float(
            1.0 - torch.var(returns - values_all, unbiased=False) / (var_returns + 1e-8)
        )
    stats["explained_variance"] = explained_variance
    stats["epochs_ran"] = float(epochs_ran)
    stats["early_stopped"] = 1.0 if early_stopped else 0.0
    stats["updates"] = n_minibatches
    stats["epoch_log"] = epoch_log
    return stats

