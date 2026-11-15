"""
Unit tests for GRPO & Dr. GRPO loss mechanics:
1. Advantage zero-std guard (identical group rewards yield 0.0, zero NaNs).
2. Dr. GRPO vs standard GRPO advantage scaling & length normalization switches.
3. k3 KL >= 0 everywhere and KL(p || p) == 0.
4. Importance ratio rho == 1.0 when log_pi_theta == log_pi_old.
5. Clipping direction for both positive and negative advantages.
6. Completion mask isolation (masked tokens produce zero gradient).
"""

import torch
from grpo_rlvr.grpo_loss import (
    compute_group_advantages,
    compute_grpo_loss,
    compute_k3_kl,
)


def test_advantage_zero_std_guard_and_modes():
    rewards = torch.tensor([[0.2, 0.2, 0.2, 0.2], [0.0, 0.2, 1.2, 0.2]])

    # Standard GRPO (normalize_std=True)
    adv_std, zero_frac = compute_group_advantages(rewards, normalize_std=True)
    assert zero_frac == 0.5
    assert torch.all(adv_std[0] == 0.0)
    assert not torch.isnan(adv_std).any()
    assert torch.allclose(adv_std[1].mean(), torch.tensor(0.0), atol=1e-6)

    # Dr. GRPO (normalize_std=False)
    adv_dr, zero_frac_dr = compute_group_advantages(rewards, normalize_std=False)
    assert zero_frac_dr == 0.5
    assert torch.all(adv_dr[0] == 0.0)
    assert torch.allclose(adv_dr[1], rewards[1] - rewards[1].mean())


def test_k3_kl_non_negative_and_zero_at_identity():
    log_p = torch.tensor([[-1.2, -0.5, -3.0]])
    log_q = torch.tensor([[-0.8, -2.5, -0.1]])

    kl_same = compute_k3_kl(log_p, log_p)
    assert torch.allclose(kl_same, torch.zeros_like(log_p), atol=1e-7)

    kl_diff = compute_k3_kl(log_p, log_q)
    assert (kl_diff >= 0.0).all()

    # Extreme difference clamped safely without inf/nan
    kl_extreme = compute_k3_kl(torch.tensor([-100.0]), torch.tensor([0.0]), kl_clamp=20.0)
    assert torch.isfinite(kl_extreme).all()


def test_rho_equals_one_and_clipping_directions():
    log_old = torch.tensor([[-1.0, -1.0]])
    log_ref = torch.tensor([[-1.0, -1.0]])
    mask = torch.tensor([[1.0, 1.0]])

    # 1. When theta == theta_old, mean_ratio == 1.0 and clip_fraction == 0.0
    log_theta_eq = log_old.clone().requires_grad_(True)
    _, metrics_eq = compute_grpo_loss(
        log_theta_eq, log_old, log_ref, torch.tensor([1.0]), mask, beta_kl=0.0
    )
    assert metrics_eq["mean_ratio"] == 1.0
    assert metrics_eq["clip_fraction"] == 0.0

    # 2. Positive advantage with rho > 1 + eps_clip must be clipped (zero surrogate grad)
    log_theta_pos = torch.tensor([[-0.5, -0.5]], requires_grad=True)  # exp(0.5) ~ 1.648 > 1.2
    loss_pos, m_pos = compute_grpo_loss(
        log_theta_pos, log_old, log_ref, torch.tensor([1.0]), mask, eps_clip=0.2, beta_kl=0.0
    )
    loss_pos.backward()
    assert m_pos["clip_fraction"] == 1.0
    assert torch.allclose(log_theta_pos.grad, torch.zeros_like(log_theta_pos))

    # 3. Negative advantage with rho < 1 - eps_clip must be clipped (zero surrogate grad)
    log_theta_neg = torch.tensor([[-1.5, -1.5]], requires_grad=True)  # exp(-0.5) ~ 0.606 < 0.8
    loss_neg, m_neg = compute_grpo_loss(
        log_theta_neg, log_old, log_ref, torch.tensor([-1.0]), mask, eps_clip=0.2, beta_kl=0.0
    )
    loss_neg.backward()
    assert m_neg["clip_fraction"] == 1.0
    assert torch.allclose(log_theta_neg.grad, torch.zeros_like(log_theta_neg))


def test_mask_isolation_and_length_norm_modes():
    log_theta = torch.tensor([[-1.0, -1.0, -2.0]], requires_grad=True)
    # exp(-1.0 - (-1.1)) = exp(0.1) ~ 1.105 < 1.2 (inside unclipped trust region)
    log_old = torch.tensor([[-1.1, -1.2, -0.5]])
    log_ref = torch.tensor([[-1.0, -1.0, -0.5]])
    mask = torch.tensor([[1.0, 0.0, 0.0]])

    loss_const, _ = compute_grpo_loss(
        log_theta, log_old, log_ref, torch.tensor([1.0]), mask,
        length_norm_mode="constant", max_new_tokens=10,
    )
    loss_const.backward()
    # Masked tokens (index 1, 2) must receive exact zero gradient
    assert log_theta.grad[0, 1].item() == 0.0
    assert log_theta.grad[0, 2].item() == 0.0
    assert log_theta.grad[0, 0].item() != 0.0

    # Sequence mode divides by 1 (active tokens), constant mode divides by 10 -> 10x ratio
    log_theta_seq = log_theta.detach().clone().requires_grad_(True)
    loss_seq, _ = compute_grpo_loss(
        log_theta_seq, log_old, log_ref, torch.tensor([1.0]), mask,
        length_norm_mode="sequence", max_new_tokens=10,
    )
    assert torch.allclose(loss_seq, 10.0 * loss_const, atol=1e-6)
