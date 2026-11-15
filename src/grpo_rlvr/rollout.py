"""Batched KV-cached rollout generation and exact sequence log-probability evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from modern_decoder_llm import DecoderLLM, StaticKVCache


@dataclass(frozen=True)
class RolloutBatch:
    """Container for batched GRPO rollouts and their completion masks."""

    sequences: torch.Tensor
    completion_mask: torch.Tensor
    truncated: torch.Tensor
    prompt_len: int
    cached_logprobs: torch.Tensor


@torch.no_grad()
def generate_rollouts(
    model: DecoderLLM,
    prompt_ids: torch.Tensor,
    group_size: int,
    max_new_tokens: int,
    eos_id: int,
    pad_id: int,
    temperature: float = 1.0,
    greedy: bool = False,
) -> RolloutBatch:
    """Generate ``group_size`` completions per prompt using ``StaticKVCache``.

    Assumes fixed-width prompts (no ``<pad>`` tokens) so all ``B * G`` trajectories
    share ``StaticKVCache.cur_len`` without left-padding offsets.

    ``completion_mask`` is ``1.0`` up to and including the first ``eos_id``, and
    ``0.0`` afterwards. Sequences that never emit ``eos_id`` within ``max_new_tokens``
    are marked ``truncated=True``.
    """
    if prompt_ids.dim() != 2:
        raise ValueError(f"Expected 2D prompt_ids (B, prompt_len), got {prompt_ids.shape}.")
    if group_size < 1 or max_new_tokens < 1:
        raise ValueError("group_size and max_new_tokens must be >= 1.")
    if temperature <= 0.0:
        raise ValueError(f"temperature must be > 0, got {temperature}.")

    batch_size, prompt_len = prompt_ids.shape
    total_batch = batch_size * group_size
    device = prompt_ids.device

    expanded_prompts = prompt_ids.repeat_interleave(group_size, dim=0)
    kv_cache_mgr = StaticKVCache(
        batch_size=total_batch,
        max_seq_len=prompt_len + max_new_tokens,
        n_layers=model.depth,
        n_kv_heads=model.layers[0].attn.n_kv_heads,
        head_dim=model.layers[0].attn.head_dim,
        device=device,
        dtype=torch.float32,
    )
    layer_caches = [kv_cache_mgr.get_layer_cache(i) for i in range(model.depth)]

    was_training = model.training
    model.eval()
    try:
        logits, _ = model.forward(expanded_prompts, start_pos=0, kv_caches=layer_caches)
        kv_cache_mgr.advance(prompt_len)
        next_logits = logits[:, -1, :]

        finished = torch.zeros(total_batch, dtype=torch.bool, device=device)
        gen_tokens: list[torch.Tensor] = []
        masks: list[torch.Tensor] = []
        step_logprobs: list[torch.Tensor] = []

        for step in range(max_new_tokens):
            scaled_logits = next_logits if temperature == 1.0 else next_logits / temperature
            log_probs = F.log_softmax(scaled_logits, dim=-1)

            if greedy:
                next_tokens = scaled_logits.argmax(dim=-1)
            else:
                next_tokens = torch.multinomial(log_probs.exp(), num_samples=1).squeeze(-1)

            chosen_logprobs = log_probs.gather(dim=-1, index=next_tokens.unsqueeze(-1)).squeeze(-1)

            # Include the first EOS token in the active mask, then mask all subsequent steps.
            active = ~finished
            step_mask = active.to(torch.float32)
            next_tokens = torch.where(active, next_tokens, torch.full_like(next_tokens, pad_id))

            gen_tokens.append(next_tokens)
            masks.append(step_mask)
            step_logprobs.append(chosen_logprobs)

            finished = finished | (next_tokens == eos_id)
            if finished.all() or step + 1 == max_new_tokens:
                break

            logits, _ = model.forward(
                next_tokens.unsqueeze(-1),
                start_pos=kv_cache_mgr.cur_len,
                kv_caches=layer_caches,
            )
            kv_cache_mgr.advance(1)
            next_logits = logits[:, -1, :]
    finally:
        if was_training:
            model.train()

    generated = torch.stack(gen_tokens, dim=1)
    completion_mask = torch.stack(masks, dim=1)
    cached_logprobs = torch.stack(step_logprobs, dim=1)
    sequences = torch.cat([expanded_prompts, generated], dim=1)

    return RolloutBatch(
        sequences=sequences,
        completion_mask=completion_mask,
        truncated=~finished,
        prompt_len=prompt_len,
        cached_logprobs=cached_logprobs,
    )


def compute_sequence_logprobs(
    model: DecoderLLM,
    sequences: torch.Tensor,
    prompt_len: int,
) -> torch.Tensor:
    """Compute per-token log-probabilities for generated tokens via a full forward pass.

    Preserves autograd gradients when called outside ``torch.no_grad()`` so it can be
    used both for ``log_pi_theta`` (with grad) and ``log_pi_old`` / ``log_pi_ref`` (no grad).
    """
    if sequences.dim() != 2:
        raise ValueError(f"Expected 2D sequences (N, T), got {sequences.shape}.")
    if prompt_len < 1 or prompt_len >= sequences.size(1):
        raise ValueError(
            f"prompt_len={prompt_len} must be in [1, {sequences.size(1) - 1}] for sequences {sequences.shape}."
        )

    logits, _ = model.forward(sequences, start_pos=0, kv_caches=None)
    targets = sequences[:, prompt_len:]
    pred_logits = logits[:, prompt_len - 1 : -1, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)
    return log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
