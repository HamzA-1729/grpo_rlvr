"""
Group Relative Policy Optimization (GRPO & Dr. GRPO) Loss Engine.

Implements:
1. Group-relative advantage estimation with Dr. GRPO (no std division) and
   standard GRPO (zero-std-guarded std division) modes.
2. PPO-style clipped surrogate objective over valid completion tokens.
3. Clamped Schulman k3 KL value estimator: exp(log r) - log r - 1 >= 0.
4. Constant (Dr. GRPO) vs. per-sequence length normalization switches.
"""

from typing import Dict, Tuple
import torch


def compute_group_advantages(
    rewards: torch.Tensor,
    normalize_std: bool = False,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, float]:
    """
    Computes group-relative advantages over (B, G) reward groups.

    Args:
        rewards: Tensor of shape (B, G) where B is prompts_per_step and G is group_size.
        normalize_std: If False (Dr. GRPO default), omits std division to avoid
            difficulty bias on near-uniform groups. If True, normalizes by group std.
        eps: Small constant for numerical stability in std division.

    Returns:
        advantages: Tensor of shape (B, G).
        zero_spread_frac: Fraction of groups with zero reward variance.
    """
    if rewards.ndim != 2:
        raise ValueError(f"Expected rewards of shape (B, G), got {tuple(rewards.shape)}.")

    mean = rewards.mean(dim=-1, keepdim=True)
    centered = rewards - mean

    spread = rewards.max(dim=-1).values - rewards.min(dim=-1).values
    zero_spread_frac = float((spread < 1e-8).float().mean().item())

    if not normalize_std:
        return centered, zero_spread_frac

    std = rewards.std(dim=-1, unbiased=False, keepdim=True)
    advantages = torch.where(
        std > 1e-8,
        centered / (std + eps),
        torch.zeros_like(centered),
    )
    return advantages, zero_spread_frac


def compute_k3_kl(
    log_pi_theta: torch.Tensor,
    log_pi_ref: torch.Tensor,
    kl_clamp: float = 20.0,
) -> torch.Tensor:
    """
    Schulman's k3 KL value estimator:
        log_r = clamp(log_pi_ref - log_pi_theta, -kl_clamp, kl_clamp)
        k3 = exp(log_r) - log_r - 1 >= 0
    """
    log_r = torch.clamp(log_pi_ref - log_pi_theta, min=-kl_clamp, max=kl_clamp)
    return torch.exp(log_r) - log_r - 1.0


def compute_grpo_loss(
    log_pi_theta: torch.Tensor,
    log_pi_old: torch.Tensor,
    log_pi_ref: torch.Tensor,
    advantages: torch.Tensor,
    completion_mask: torch.Tensor,
    eps_clip: float = 0.2,
    beta_kl: float = 0.04,
    kl_clamp: float = 20.0,
    length_norm_mode: str = "constant",
    max_new_tokens: int = 96,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Computes the scalar GRPO / Dr. GRPO loss and training diagnostics.

    Args:
        log_pi_theta: (N, T) log-probs under current trainable policy.
        log_pi_old: (N, T) detached log-probs under rollout policy.
        log_pi_ref: (N, T) detached log-probs under frozen reference policy.
        advantages: (N,) trajectory-level advantages.
        completion_mask: (N, T) binary mask (1 on generated tokens up to & including
            first EOS; 0 on prompt / post-EOS tokens).
        eps_clip: PPO clipping epsilon.
        beta_kl: KL penalty coefficient.
        kl_clamp: Clamp range [-kl_clamp, kl_clamp] before exp in k3.
        length_norm_mode: "constant" (Dr. GRPO, divides by max_new_tokens) or
            "sequence" (standard GRPO, divides by each trajectory's token count).
        max_new_tokens: Constant denominator when length_norm_mode == "constant".
    """
    mask = completion_mask.float()
    adv = advantages.unsqueeze(-1)

    log_ratio = log_pi_theta - log_pi_old.detach()
    rho = torch.exp(log_ratio)

    unclipped = rho * adv
    clipped = torch.clamp(rho, 1.0 - eps_clip, 1.0 + eps_clip) * adv
    surrogate = torch.min(unclipped, clipped)

    kl = compute_k3_kl(log_pi_theta, log_pi_ref.detach(), kl_clamp=kl_clamp)
    per_token_obj = (surrogate - beta_kl * kl) * mask

    if length_norm_mode == "constant":
        denom = float(max_new_tokens)
    elif length_norm_mode == "sequence":
        denom = mask.sum(dim=-1).clamp(min=1.0)
    else:
        raise ValueError(f"Unknown length_norm_mode: {length_norm_mode!r}")

    traj_obj = per_token_obj.sum(dim=-1) / denom
    loss = -traj_obj.mean()

    with torch.no_grad():
        total_active = mask.sum().clamp(min=1.0)
        mean_k3_kl = float(((kl * mask).sum() / total_active).item())
        mean_ratio = float(((rho * mask).sum() / total_active).item())
        is_clipped = ((rho > 1.0 + eps_clip) & (adv > 0)) | (
            (rho < 1.0 - eps_clip) & (adv < 0)
        )
        clip_fraction = float(((is_clipped.float() * mask).sum() / total_active).item())

    metrics = {
        "mean_k3_kl": mean_k3_kl,
        "mean_ratio": mean_ratio,
        "clip_fraction": clip_fraction,
    }
    return loss, metrics
