"""Integration tests for a single GRPO step and reference model immutability."""

from __future__ import annotations

import math

import torch
from modern_decoder_llm import DecoderLLM

import grpo_rlvr.train_grpo as train_grpo_mod
from grpo_rlvr.tokenizer import ArithmeticTokenizer
from grpo_rlvr.train_grpo import freeze_reference_model, grpo_step
from grpo_rlvr.verifier import RewardResult


def _make_tiny_model(vocab_size: int = 32, max_seq_len: int = 64) -> DecoderLLM:
    torch.manual_seed(7)
    return DecoderLLM(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        hidden_dim=32,
        depth=2,
        n_heads=4,
        n_kv_heads=2,
        head_dim=8,
        ffn_dim=64,
        rope_theta=10000.0,
        norm_eps=1e-5,
        tie_weights=True,
    )


def test_grpo_step_updates_policy_and_preserves_reference(monkeypatch):
    tok = ArithmeticTokenizer()
    policy = _make_tiny_model(vocab_size=tok.vocab_size, max_seq_len=64)
    ref = freeze_reference_model(policy)

    assert all(not p.requires_grad for p in ref.parameters())

    policy_before = [p.detach().clone() for p in policy.parameters()]
    ref_before = [p.detach().clone() for p in ref.parameters()]

    # Alternate rewards within each group so advantages are guaranteed non-zero.
    counter = {"idx": 0}

    def fake_reward(completion_text, numbers, target, truncated=False, format_only=False):
        val = 1.0 if (counter["idx"] % 2 == 0) else 0.0
        counter["idx"] += 1
        return RewardResult(total=val, format_reward=0.1 * val, accuracy_reward=val)

    monkeypatch.setattr(train_grpo_mod, "compute_reward", fake_reward)

    problems = [
        {"prompt": "T: 24|N: 2, 3, 4, 6\n", "numbers": [2, 3, 4, 6], "target": 24},
        {"prompt": "T: 10|N: 1, 2, 3, 4\n", "numbers": [1, 2, 3, 4], "target": 10},
    ]
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)

    metrics = grpo_step(
        policy_model=policy,
        ref_model=ref,
        optimizer=optimizer,
        batch_problems=problems,
        tokenizer=tok,
        group_size=4,
        max_new_tokens=12,
        temperature=1.0,
        mu_iters=1,
    )

    # First inner update has pi_theta == pi_old, so rho == 1.0 and clip_frac == 0.0.
    assert metrics["clip_frac"] == 0.0
    assert metrics["zero_std_frac"] == 0.0
    assert math.isfinite(metrics["loss"])

    # Policy parameters must have updated; reference parameters must remain bit-identical.
    assert any(not torch.equal(p, p0) for p, p0 in zip(policy.parameters(), policy_before))
    assert all(torch.equal(r, r0) for r, r0 in zip(ref.parameters(), ref_before))


def test_grpo_step_end_to_end_real_verifier():
    tok = ArithmeticTokenizer()
    policy = _make_tiny_model(vocab_size=tok.vocab_size, max_seq_len=64)
    ref = freeze_reference_model(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    problems = [
        {"prompt": "T: 24|N: 2, 3, 4, 6\n", "numbers": [2, 3, 4, 6], "target": 24},
    ]
    metrics = grpo_step(
        policy_model=policy,
        ref_model=ref,
        optimizer=optimizer,
        batch_problems=problems,
        tokenizer=tok,
        group_size=4,
        max_new_tokens=12,
        temperature=1.0,
        mu_iters=2,
        format_only=False,
    )
    assert 0.0 <= metrics["zero_std_frac"] <= 1.0
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["kl"])
