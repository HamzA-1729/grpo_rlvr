"""
Unit tests for Countdown Problem Generator & Disjoint Splits:
1. Every generated problem's solver trace passes compute_reward() with 1.2.
2. Prompt length is identically 20 tokens and total trace fits within max_seq_len=128.
3. SFT train, GRPO train, and Eval splits are strictly disjoint at (sorted_numbers, target).
4. SFTCountdownDataset masks prompt and padding tokens with -100.
"""

import json
from pathlib import Path

from grpo_rlvr.data import (
    SFTCountdownDataset,
    enumerate_solvable_problems,
    generate_disjoint_splits,
)
from grpo_rlvr.tokenizer import ArithmeticTokenizer
from grpo_rlvr.verifier import compute_reward


def test_generated_problems_pass_verifier_and_length_bounds():
    tok = ArithmeticTokenizer()
    problems = enumerate_solvable_problems(
        num_operands=4,
        min_operand=1,
        max_operand=5,
        min_target=1,
        max_target=50,
    )
    assert len(problems) > 100

    for p in problems[:300]:
        res = compute_reward(p.trace, p.numbers, p.target)
        assert res.total == 1.2, f"Solver trace failed verifier for {p}: {res}"
        assert len(tok.encode(p.prompt)) == 20
        assert len(tok.encode(p.prompt + p.trace, add_eos=True)) <= 128


def test_disjoint_splits():
    sft, grpo, eval_set, total_solvable = generate_disjoint_splits(
        sft_train_size=200,
        grpo_train_size=200,
        eval_size=100,
        seed=42,
        min_operand=1,
        max_operand=6,
        min_target=1,
        max_target=50,
        enforce_space_ratio=False,
    )
    assert total_solvable >= 500

    sft_keys = {(tuple(sorted(p.numbers)), p.target) for p in sft}
    grpo_keys = {(tuple(sorted(p.numbers)), p.target) for p in grpo}
    eval_keys = {(tuple(sorted(p.numbers)), p.target) for p in eval_set}

    assert len(sft_keys) == 200
    assert len(grpo_keys) == 200
    assert len(eval_keys) == 100
    assert sft_keys.isdisjoint(grpo_keys)
    assert sft_keys.isdisjoint(eval_keys)
    assert grpo_keys.isdisjoint(eval_keys)


def test_sft_dataset_prompt_masking(tmp_path: Path):
    tok = ArithmeticTokenizer()
    problems = enumerate_solvable_problems(
        num_operands=4, min_operand=1, max_operand=4, min_target=1, max_target=24
    )
    p = problems[0]
    data_file = tmp_path / "sft.json"
    with open(data_file, "w", encoding="utf-8") as f:
        json.dump([{"prompt": p.prompt, "trace": p.trace}], f)

    ds = SFTCountdownDataset(data_file, tokenizer=tok, max_seq_len=128)
    input_ids, labels = ds[0]
    assert input_ids.shape == (128,)
    assert labels.shape == (128,)

    # First 20 prompt tokens must be masked with -100
    assert (labels[:20] == -100).all()

    trace_ids = tok.encode(p.trace, add_eos=True)
    # Trace tokens (including <eos>) must be supervised
    for i, tid in enumerate(trace_ids):
        assert labels[20 + i].item() == tid
    # Remaining padding must be -100
    assert (labels[20 + len(trace_ids) :] == -100).all()
