"""Held-out evaluation for Countdown RLVR models (Pass@1 greedy, Pass@K, format, length)."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from modern_decoder_llm import DecoderLLM

from grpo_rlvr.rollout import generate_rollouts
from grpo_rlvr.tokenizer import ArithmeticTokenizer
from grpo_rlvr.train_sft import build_model
from grpo_rlvr.verifier import compute_reward


@dataclass(frozen=True)
class EvalMetrics:
    """Summary metrics over a held-out evaluation split."""

    num_problems: int
    pass_at_1: float
    format_rate_greedy: float
    pass_at_k: float
    k: int
    mean_completion_len: float
    truncation_rate: float
    mean_reward_greedy: float


@torch.no_grad()
def evaluate_model(
    model: DecoderLLM,
    problems: list[dict[str, Any]],
    tokenizer: ArithmeticTokenizer,
    max_new_tokens: int = 96,
    pass_k: int = 8,
    temperature: float = 1.0,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> EvalMetrics:
    """Evaluate greedy Pass@1 and sampled Pass@K on held-out Countdown problems."""
    if device is None:
        device = next(model.parameters()).device

    n = len(problems)
    if n == 0:
        raise ValueError("Evaluation problem list is empty.")

    was_training = model.training
    model.eval()
    try:
        greedy_correct = 0
        greedy_format = 0
        greedy_reward_sum = 0.0
        greedy_len_sum = 0.0
        greedy_trunc = 0
        pass_k_correct = 0

        for start in range(0, n, batch_size):
            chunk = problems[start : start + batch_size]
            prompt_ids = torch.tensor(
                [tokenizer.encode(item["prompt"]) for item in chunk],
                dtype=torch.long,
                device=device,
            )

            # 1. Greedy Pass@1
            greedy_batch = generate_rollouts(
                model=model,
                prompt_ids=prompt_ids,
                group_size=1,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                greedy=True,
            )
            gen_greedy = greedy_batch.sequences[:, greedy_batch.prompt_len :]
            for idx, item in enumerate(chunk):
                active_len = int(greedy_batch.completion_mask[idx].sum().item())
                comp_ids = gen_greedy[idx, :active_len].tolist()
                comp_text = tokenizer.decode(comp_ids, skip_pad_eos=True)
                truncated = bool(greedy_batch.truncated[idx].item())
                res = compute_reward(
                    completion_text=comp_text,
                    numbers=item["numbers"],
                    target=item["target"],
                    truncated=truncated,
                )
                if res.accuracy_reward > 0.0:
                    greedy_correct += 1
                if res.format_reward > 0.0:
                    greedy_format += 1
                greedy_reward_sum += res.total
                greedy_len_sum += active_len
                if truncated:
                    greedy_trunc += 1

            # 2. Sampled Pass@K
            if pass_k <= 1:
                continue

            sampled_batch = generate_rollouts(
                model=model,
                prompt_ids=prompt_ids,
                group_size=pass_k,
                max_new_tokens=max_new_tokens,
                eos_id=tokenizer.eos_id,
                pad_id=tokenizer.pad_id,
                temperature=temperature,
                greedy=False,
            )
            gen_sampled = sampled_batch.sequences[:, sampled_batch.prompt_len :]
            for p_idx, item in enumerate(chunk):
                solved = False
                for g_idx in range(pass_k):
                    row = p_idx * pass_k + g_idx
                    active_len = int(sampled_batch.completion_mask[row].sum().item())
                    comp_ids = gen_sampled[row, :active_len].tolist()
                    comp_text = tokenizer.decode(comp_ids, skip_pad_eos=True)
                    truncated = bool(sampled_batch.truncated[row].item())
                    res = compute_reward(
                        completion_text=comp_text,
                        numbers=item["numbers"],
                        target=item["target"],
                        truncated=truncated,
                    )
                    if res.accuracy_reward > 0.0:
                        solved = True
                        break
                if solved:
                    pass_k_correct += 1
    finally:
        if was_training:
            model.train()

    pass_at_1 = greedy_correct / n
    pass_at_k = pass_at_1 if pass_k <= 1 else pass_k_correct / n
    return EvalMetrics(
        num_problems=n,
        pass_at_1=pass_at_1,
        format_rate_greedy=greedy_format / n,
        pass_at_k=pass_at_k,
        k=pass_k,
        mean_completion_len=greedy_len_sum / n,
        truncation_rate=greedy_trunc / n,
        mean_reward_greedy=greedy_reward_sum / n,
    )


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[DecoderLLM, dict[str, Any]]:
    """Load ``DecoderLLM`` and its saved ``model_cfg`` from a training checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = ckpt["model_cfg"]
    model = build_model(model_cfg, vocab_size=int(model_cfg["vocab_size"]), device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Countdown RLVR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, default=Path("data/eval.json"))
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--pass-k", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArithmeticTokenizer()

    with args.data_path.open("r", encoding="utf-8") as f:
        problems = json.load(f)
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    model, _ = load_model_from_checkpoint(args.checkpoint, device=device)
    metrics = evaluate_model(
        model=model,
        problems=problems,
        tokenizer=tokenizer,
        max_new_tokens=args.max_new_tokens,
        pass_k=args.pass_k,
        temperature=args.temperature,
        batch_size=args.batch_size,
        device=device,
    )
    print(
        f"[EVAL] n={metrics.num_problems} | "
        f"Pass@1={metrics.pass_at_1:.3f} | "
        f"Pass@{metrics.k}={metrics.pass_at_k:.3f} | "
        f"Format={metrics.format_rate_greedy:.3f} | "
        f"MeanLen={metrics.mean_completion_len:.1f} | "
        f"Trunc={metrics.truncation_rate:.3f} | "
        f"Reward={metrics.mean_reward_greedy:.3f}"
    )


if __name__ == "__main__":
    main()
