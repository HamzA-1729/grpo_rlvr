"""
Procedural Countdown Problem Generator, Brute-Force Solver & Disjoint Splits.

Features:
1. Fixed-width prompt formatting ("T: 24|N: 2, 3, 4, 6\\n" -> 20 tokens, zero <pad>).
2. Exact integer-division brute-force solver producing deterministic step-by-step
   search/evaluation traces for R1-style SFT warmstart.
3. Exhaustive solvable-space enumeration (Step 2 Space-Size Gate) verifying
   total solvable (multiset, target) pairs >= 10x total used prompts.
4. Strictly disjoint SFT train, GRPO train, and held-out eval splits.
"""

from dataclasses import dataclass
import itertools
import json
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Tuple
import torch
from torch.utils.data import Dataset

from grpo_rlvr.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class CountdownProblem:
    """Single Countdown arithmetic instance with a verified reference solution."""

    numbers: Tuple[int, ...]
    target: int
    prompt: str
    trace: str
    answer: str


def format_prompt(numbers: Sequence[int], target: int) -> str:
    """
    Formats a 4-number Countdown prompt to an exact fixed 20-character (20-token) string.
    Example: 'T: 24|N: 2, 3, 4, 6\\n'
    """
    if len(numbers) != 4:
        raise ValueError(f"Expected 4 numbers for fixed-width prompt, got {len(numbers)}.")
    nums_str = ",".join(f"{n:2d}" for n in numbers)
    return f"T:{target:3d}|N:{nums_str}\n"


def _solve_all_targets_for_multiset(
    numbers: Tuple[int, ...],
    min_target: int,
    max_target: int,
) -> Dict[int, Tuple[List[str], str]]:
    """
    Exhaustively finds one exact integer-division solution for every reachable
    target in [min_target, max_target] using all numbers in `numbers`.

    Returns:
        Dict mapping target -> (list_of_step_strings, final_expression)
    """
    solutions: Dict[int, Tuple[List[str], str]] = {}

    # Each item is (value, expr_str, is_compound)
    initial_items = [(n, str(n), False) for n in numbers]

    def _search(items: List[Tuple[int, str, bool]], steps: List[str]) -> None:
        if len(items) == 1:
            val, expr, _ = items[0]
            if min_target <= val <= max_target and val not in solutions:
                solutions[val] = (list(steps), expr)
            return

        n = len(items)
        seen_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                a_val, a_expr, a_comp = items[i]
                b_val, b_expr, b_comp = items[j]

                # Canonicalize pair order for deduplication
                pair_key = (min(a_val, b_val), max(a_val, b_val))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                rest = [items[k] for k in range(n) if k != i and k != j]
                a_wrap = f"({a_expr})" if a_comp else a_expr
                b_wrap = f"({b_expr})" if b_comp else b_expr

                candidates: List[Tuple[int, str, str]] = [
                    (a_val + b_val, f"{a_wrap}+{b_wrap}", f"{a_val}+{b_val}={a_val + b_val}"),
                    (a_val * b_val, f"{a_wrap}*{b_wrap}", f"{a_val}*{b_val}={a_val * b_val}"),
                ]
                if a_val >= b_val:
                    candidates.append(
                        (a_val - b_val, f"{a_wrap}-{b_wrap}", f"{a_val}-{b_val}={a_val - b_val}")
                    )
                else:
                    candidates.append(
                        (b_val - a_val, f"{b_wrap}-{a_wrap}", f"{b_val}-{a_val}={b_val - a_val}")
                    )
                if b_val != 0 and a_val % b_val == 0:
                    candidates.append(
                        (a_val // b_val, f"{a_wrap}/{b_wrap}", f"{a_val}/{b_val}={a_val // b_val}")
                    )
                if a_val != 0 and b_val % a_val == 0 and a_val != b_val:
                    candidates.append(
                        (b_val // a_val, f"{b_wrap}/{a_wrap}", f"{b_val}/{a_val}={b_val // a_val}")
                    )

                for res_val, res_expr, step_str in candidates:
                    steps.append(step_str)
                    _search(rest + [(res_val, res_expr, True)], steps)
                    steps.pop()

    _search(initial_items, [])
    return solutions


def build_sft_trace(numbers: Tuple[int, ...], steps: List[str], answer_expr: str) -> str:
    """
    Constructs a deterministic search-and-verify <think>...</think><answer>...</answer> trace.
    Includes an initial exploratory pair check before the winning reduction steps.
    """
    a, b = numbers[0], numbers[1]
    probe = f"{a}+{b}={a + b}"
    if probe != steps[0]:
        think_body = "; ".join([probe] + steps)
    else:
        think_body = "; ".join(steps)
    return f"<think>{think_body}</think><answer>{answer_expr}</answer>"


def enumerate_solvable_problems(
    num_operands: int = 4,
    min_operand: int = 1,
    max_operand: int = 12,
    min_target: int = 1,
    max_target: int = 100,
) -> List[CountdownProblem]:
    """
    Exhaustively enumerates all solvable (sorted_numbers, target) problems in the space.
    """
    if num_operands != 4:
        raise ValueError("Fixed-width prompt format currently expects num_operands == 4.")

    problems: List[CountdownProblem] = []
    multisets = list(
        itertools.combinations_with_replacement(range(min_operand, max_operand + 1), num_operands)
    )

    for nums in multisets:
        sols = _solve_all_targets_for_multiset(nums, min_target, max_target)
        for target, (steps, answer_expr) in sols.items():
            prompt = format_prompt(nums, target)
            trace = build_sft_trace(nums, steps, answer_expr)
            problems.append(
                CountdownProblem(
                    numbers=nums,
                    target=target,
                    prompt=prompt,
                    trace=trace,
                    answer=answer_expr,
                )
            )
    return problems


def generate_disjoint_splits(
    sft_train_size: int = 2000,
    grpo_train_size: int = 2000,
    eval_size: int = 500,
    seed: int = 42,
    min_operand: int = 1,
    max_operand: int = 12,
    min_target: int = 1,
    max_target: int = 100,
    enforce_space_ratio: bool = True,
) -> Tuple[List[CountdownProblem], List[CountdownProblem], List[CountdownProblem], int]:
    """
    Generates strictly disjoint SFT train, GRPO train, and Eval splits at the
    (tuple(sorted(numbers)), target) level.
    """
    all_problems = enumerate_solvable_problems(
        num_operands=4,
        min_operand=min_operand,
        max_operand=max_operand,
        min_target=min_target,
        max_target=max_target,
    )
    total_solvable = len(all_problems)
    total_needed = sft_train_size + grpo_train_size + eval_size

    if total_solvable < total_needed:
        raise ValueError(
            f"Solvable space ({total_solvable}) is smaller than total requested ({total_needed})."
        )
    if enforce_space_ratio and total_solvable < 10 * total_needed:
        raise ValueError(
            f"Space-Size Gate failed: {total_solvable} solvable pairs < 10x requested ({10 * total_needed})."
        )

    rng = random.Random(seed)
    shuffled = list(all_problems)
    rng.shuffle(shuffled)

    sft_split = shuffled[:sft_train_size]
    grpo_split = shuffled[sft_train_size : sft_train_size + grpo_train_size]
    eval_split = shuffled[sft_train_size + grpo_train_size : total_needed]
    return sft_split, grpo_split, eval_split, total_solvable


def save_splits(
    output_dir: str | Path = "data",
    sft_train_size: int = 2000,
    grpo_train_size: int = 2000,
    eval_size: int = 500,
    seed: int = 42,
) -> Dict[str, int]:
    """Enumerates, validates, and saves disjoint Countdown JSON splits."""
    sft, grpo, eval_set, total_solvable = generate_disjoint_splits(
        sft_train_size=sft_train_size,
        grpo_train_size=grpo_train_size,
        eval_size=eval_size,
        seed=seed,
    )
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def _to_dicts(items: List[CountdownProblem]) -> List[dict]:
        return [
            {
                "numbers": list(p.numbers),
                "target": p.target,
                "prompt": p.prompt,
                "trace": p.trace,
                "answer": p.answer,
            }
            for p in items
        ]

    for filename, split in [
        ("sft_train.json", sft),
        ("grpo_train.json", grpo),
        ("eval.json", eval_set),
    ]:
        with open(out_path / filename, "w", encoding="utf-8") as f:
            json.dump(_to_dicts(split), f, indent=2)

    return {
        "total_solvable_pairs": total_solvable,
        "sft_train_size": len(sft),
        "grpo_train_size": len(grpo),
        "eval_size": len(eval_set),
    }


class SFTCountdownDataset(Dataset):
    """Supervised dataset masking the 20-token prompt with -100."""

    def __init__(
        self,
        data_path: str | Path,
        tokenizer: ArithmeticTokenizer,
        max_seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples: List[dict] = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.samples[idx]
        prompt_ids = self.tokenizer.encode(item["prompt"])
        target_ids = self.tokenizer.encode(item["trace"], add_eos=True)

        input_ids = (prompt_ids + target_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + target_ids)[: self.max_seq_len]

        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.tokenizer.pad_id] * pad_len
            labels = labels + [-100] * pad_len

        return (
            torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
        )


if __name__ == "__main__":
    summary = save_splits()
    print("=" * 60)
    print("STEP 2 SPACE-SIZE GATE SUMMARY:")
    for k, v in summary.items():
        print(f"  {k:<24}: {v:,}")
    print("=" * 60)
