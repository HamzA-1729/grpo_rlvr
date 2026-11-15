"""
Unit tests for deterministic RLVR Countdown Verifier:
1. Exact Fraction intermediate arithmetic (e.g. (5 - 1/5) * 5 = 24).
2. Strict multiset enforcement (rejects subsets, excess duplicates, digit concatenation).
3. Rejection of unary minus, leading zeros, >64 char expressions, and disallowed AST nodes.
4. Safe handling of ZeroDivisionError and SyntaxError without crashing.
5. Format validation (empty spans, nested tags, trailing text, truncation penalty, format_only).
"""

from fractions import Fraction
from grpo_rlvr.verifier import compute_reward


def test_valid_exact_solution():
    completion = "<think>4*2=8; 6-3=3; 3*8=24</think><answer>(6-3)*(4*2)</answer><eos>"
    res = compute_reward(completion, numbers=[2, 3, 4, 6], target=24)
    assert res.format_reward == 0.2
    assert res.accuracy_reward == 1.0
    assert res.total == 1.2
    assert res.evaluated_value == Fraction(24)


def test_exact_fraction_intermediate_arithmetic():
    # (5 - 1/5) * 5 = 24/5 * 5 = 24 (impossible with naive float truncation or integer-only ops)
    completion = "<think>1/5; 5-1/5; *5=24</think><answer>(5-1/5)*5</answer>"
    res = compute_reward(completion, numbers=[1, 5, 5, 5], target=24)
    assert res.accuracy_reward == 1.0
    assert res.total == 1.2
    assert res.evaluated_value == Fraction(24)


def test_multiset_cheating_rejected():
    # Subset only (uses 4 and 6, ignores 2 and 3)
    res_subset = compute_reward("<think>4*6=24</think><answer>4*6</answer>", numbers=[2, 3, 4, 6], target=24)
    assert res_subset.format_reward == 0.2
    assert res_subset.accuracy_reward == 0.0

    # Digit concatenation ("12" built from 1 and 2)
    res_concat = compute_reward("<think>12*2=24</think><answer>12*2*1*1</answer>", numbers=[1, 2, 2, 1], target=24)
    assert res_concat.format_reward == 0.2
    assert res_concat.accuracy_reward == 0.0

    # Excess multiplicity (uses two 6s when only one 6 allowed)
    res_dup = compute_reward("<think>6*6</think><answer>6+6+6+6</answer>", numbers=[6, 6, 1, 1], target=24)
    assert res_dup.accuracy_reward == 0.0


def test_disallowed_syntax_and_safety():
    # Unary minus rejected
    res_unary = compute_reward("<think>t</think><answer>-2+3+4+6</answer>", numbers=[2, 3, 4, 6], target=11)
    assert res_unary.format_reward == 0.2
    assert res_unary.accuracy_reward == 0.0

    # Leading zero (03) rejected cleanly
    res_zero = compute_reward("<think>t</think><answer>03*4+6+6</answer>", numbers=[3, 4, 6, 6], target=24)
    assert res_zero.format_reward == 0.2
    assert res_zero.accuracy_reward == 0.0

    # ZeroDivisionError handled cleanly
    res_div0 = compute_reward("<think>t</think><answer>4/(3-3)+6+2</answer>", numbers=[2, 3, 3, 4], target=24)
    assert res_div0.format_reward == 0.2
    assert res_div0.accuracy_reward == 0.0

    # Power / function call rejected
    res_pow = compute_reward("<think>t</think><answer>2**3+4+6+6</answer>", numbers=[2, 3, 4, 6], target=24)
    assert res_pow.accuracy_reward == 0.0

    # Length > 64 chars rejected
    long_expr = "1+" * 33 + "1"
    res_long = compute_reward(f"<think>t</think><answer>{long_expr}</answer>", numbers=[1] * 34, target=34)
    assert res_long.format_reward == 0.2
    assert res_long.accuracy_reward == 0.0


def test_format_strictness_and_truncation():
    # Trailing text after </answer>
    res_trail = compute_reward("<think>t</think><answer>(6-3)*(4*2)</answer> extra", numbers=[2, 3, 4, 6], target=24)
    assert res_trail.total == 0.0

    # Empty think span
    res_empty = compute_reward("<think>   </think><answer>(6-3)*(4*2)</answer>", numbers=[2, 3, 4, 6], target=24)
    assert res_empty.total == 0.0

    # Nested tag
    res_nested = compute_reward("<think><think>x</think></think><answer>(6-3)*(4*2)</answer>", numbers=[2, 3, 4, 6], target=24)
    assert res_nested.total == 0.0

    # Truncated completion gets 0.0 even if answer looks valid
    res_trunc = compute_reward("<think>t</think><answer>(6-3)*(4*2)</answer>", numbers=[2, 3, 4, 6], target=24, truncated=True)
    assert res_trunc.total == 0.0

    # Format-only mode ignores math correctness
    res_fmt = compute_reward("<think>t</think><answer>1+1</answer>", numbers=[2, 3, 4, 6], target=24, format_only=True)
    assert res_fmt.total == 0.2
    assert res_fmt.format_reward == 0.2
    assert res_fmt.accuracy_reward == 0.0
