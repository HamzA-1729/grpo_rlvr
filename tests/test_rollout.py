"""Unit tests for batched StaticKVCache rollout generation and log-probability parity."""

from __future__ import annotations

import torch
from modern_decoder_llm import DecoderLLM

from grpo_rlvr.rollout import compute_sequence_logprobs, generate_rollouts
from grpo_rlvr.tokenizer import ArithmeticTokenizer


def _make_tiny_model(vocab_size: int = 32, max_seq_len: int = 64) -> DecoderLLM:
    torch.manual_seed(0)
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


def test_cached_vs_uncached_logprobs_parity_and_grad():
    tok = ArithmeticTokenizer()
    model = _make_tiny_model(vocab_size=tok.vocab_size, max_seq_len=64)
    p1 = tok.encode("T: 24|N: 2, 3, 4, 6\n")
    p2 = tok.encode("T: 10|N: 1, 2, 3, 4\n")
    prompt_ids = torch.tensor([p1, p2], dtype=torch.long)
    prompt_len = prompt_ids.size(1)

    batch = generate_rollouts(
        model=model,
        prompt_ids=prompt_ids,
        group_size=3,
        max_new_tokens=10,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        temperature=1.0,
    )

    assert batch.sequences.shape[0] == 6
    assert batch.sequences.shape[1] > prompt_len
    assert batch.completion_mask.shape == (6, batch.sequences.shape[1] - prompt_len)
    assert batch.truncated.shape == (6,)

    model.eval()
    with torch.no_grad():
        uncached_logprobs = compute_sequence_logprobs(model, batch.sequences, prompt_len)

    active = batch.completion_mask.bool()
    assert torch.allclose(
        batch.cached_logprobs[active],
        uncached_logprobs[active],
        atol=1e-4,
    )

    # Outside no_grad, compute_sequence_logprobs must preserve gradients.
    model.train()
    grad_logprobs = compute_sequence_logprobs(model, batch.sequences, prompt_len)
    loss = -(grad_logprobs * batch.completion_mask).sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad


def test_eos_inclusion_post_eos_masking_and_truncation():
    tok = ArithmeticTokenizer()
    model = _make_tiny_model(vocab_size=tok.vocab_size, max_seq_len=64)
    p1 = tok.encode("T: 24|N: 2, 3, 4, 6\n")
    prompt_ids = torch.tensor([p1], dtype=torch.long)

    # 1. Force no EOS within max_new_tokens -> truncated=True and mask all 1.0
    orig_forward = model.forward
    digit_1_id = tok.encode("1")[0]

    def no_eos_forward(token_ids, start_pos=0, kv_caches=None):
        logits, caches = orig_forward(token_ids, start_pos=start_pos, kv_caches=kv_caches)
        logits = logits.clone()
        logits.zero_()
        logits[:, -1, digit_1_id] = 20.0
        return logits, caches

    model.forward = no_eos_forward
    trunc_batch = generate_rollouts(
        model=model,
        prompt_ids=prompt_ids,
        group_size=2,
        max_new_tokens=5,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        greedy=True,
    )
    assert trunc_batch.truncated.tolist() == [True, True]
    assert torch.equal(trunc_batch.completion_mask, torch.ones(2, 5))

    # 2. Test partial EOS: wrap forward so row 0 emits EOS at step 1 and row 1 never emits EOS.
    call_idx = {"step": 0}

    def scripted_forward(token_ids, start_pos=0, kv_caches=None):
        logits, caches = orig_forward(token_ids, start_pos=start_pos, kv_caches=kv_caches)
        logits = logits.clone()
        logits.zero_()
        step = call_idx["step"]
        call_idx["step"] += 1
        # prefill (step 0) -> gen[0], forward step 1 -> gen[1], forward step 2 -> gen[2]
        logits[:, -1, digit_1_id] = 20.0
        if step == 1:
            logits[0, -1, tok.eos_id] = 50.0
        return logits, caches

    model.forward = scripted_forward
    eos_batch = generate_rollouts(
        model=model,
        prompt_ids=prompt_ids,
        group_size=2,
        max_new_tokens=4,
        eos_id=tok.eos_id,
        pad_id=tok.pad_id,
        greedy=True,
    )
    gen = eos_batch.sequences[:, eos_batch.prompt_len :]
    assert eos_batch.truncated.tolist() == [False, True]
    # Row 0 emitted EOS at index 1: mask must be [1, 1, 0, 0] and post-EOS tokens must be pad_id
    assert eos_batch.completion_mask[0].tolist() == [1.0, 1.0, 0.0, 0.0]
    assert gen[0, 1].item() == tok.eos_id
    assert gen[0, 2].item() == tok.pad_id
    assert gen[0, 3].item() == tok.pad_id
    # Row 1 never emitted EOS: mask must be all 1.0
    assert eos_batch.completion_mask[1].tolist() == [1.0, 1.0, 1.0, 1.0]
