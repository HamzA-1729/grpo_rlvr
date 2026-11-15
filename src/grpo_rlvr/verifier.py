"""
Deterministic RLVR Verifier for Countdown Arithmetic Synthesis.

Enforces:
1. Truncation gate: completions hitting max_new_tokens without <eos> receive 0.0.
2. Strict format reward (0.2): <think>...</think><answer>...</answer> with non-empty
   spans, no nested tags, and no trailing text after </answer>.
3. Safe AST Fraction evaluator (1.0): whitelists only binary (+, -, *, /) and
   non-negative integer constants (no eval(), no unary minus, max 64 chars).
4. Strict multiset equality: Counter(used_numbers) == Counter(allowed_numbers).
"""

import ast
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
import re
from typing import List, Optional, Sequence, Tuple


MAX_EXPR_LENGTH = 64
FORMAT_REWARD = 0.2
ACCURACY_REWARD = 1.0

_FORMAT_REGEX = re.compile(
    r"^\s*<think>(?P<think>.+?)</think>\s*<answer>(?P<answer>.+?)</answer>(?:\s|<eos>)*$",
    flags=re.DOTALL,
)

_FORBIDDEN_INNER_TAGS = ("<think>", "</think>", "<answer>", "</answer>")


@dataclass(frozen=True)
class RewardResult:
    """Breakdown of deterministic verifier rewards for a single completion."""

    total: float
    format_reward: float
    accuracy_reward: float
    parsed_answer: Optional[str] = None
    evaluated_value: Optional[Fraction] = None


def extract_think_and_answer(text: str) -> Optional[Tuple[str, str]]:
    """
    Validates strict R1-style format and extracts (think_text, answer_text).
    Returns None if format is invalid, empty, nested, or has trailing text.
    """
    match = _FORMAT_REGEX.fullmatch(text)
    if match is None:
        return None

    think_text = match.group("think").strip()
    answer_text = match.group("answer").strip()

    if not think_text or not answer_text:
        return None

    for tag in _FORBIDDEN_INNER_TAGS:
        if tag in think_text or tag in answer_text:
            return None

    return think_text, answer_text


def safe_eval_countdown_expr(expr: str) -> Tuple[Fraction, List[int]]:
    """
    Safely parses and evaluates an arithmetic expression using fractions.Fraction.
    Whitelists only binary (+, -, *, /) and non-negative integer literals.

    Returns:
        (exact_fraction_value, list_of_used_integer_literals)

    Raises:
        ValueError, SyntaxError, ZeroDivisionError, RecursionError, TypeError
    """
    expr = expr.strip()
    if not expr or len(expr) > MAX_EXPR_LENGTH:
        raise ValueError(f"Expression empty or exceeds {MAX_EXPR_LENGTH} chars.")

    tree = ast.parse(expr, mode="eval")
    used_numbers: List[int] = []

    def _eval_node(node: ast.AST) -> Fraction:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)

        if isinstance(node, ast.Constant):
            # Reject bool (subclass of int), float, str, etc.
            if type(node.value) is not int or node.value < 0:
                raise ValueError(f"Disallowed literal: {node.value!r}")
            used_numbers.append(node.value)
            return Fraction(node.value)

        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError("Division by zero in candidate expression.")
                return left / right
            raise ValueError(f"Disallowed binary operator: {type(node.op).__name__}")

        # Rejects UnaryOp (unary minus), Call, Name, etc.
        raise ValueError(f"Disallowed AST node: {type(node).__name__}")

    value = _eval_node(tree)
    return value, used_numbers


def compute_reward(
    completion_text: str,
    numbers: Sequence[int],
    target: int,
    truncated: bool = False,
    format_only: bool = False,
) -> RewardResult:
    """
    Scores a completion deterministically.

    Args:
        completion_text: Decoded completion string (excluding prompt).
        numbers: Allowed operand multiset from the prompt.
        target: Target integer from the prompt.
        truncated: True if generation hit max_new_tokens without emitting <eos>.
        format_only: If True, returns only format_reward (for Step 0 sanity check).
    """
    if truncated:
        return RewardResult(total=0.0, format_reward=0.0, accuracy_reward=0.0)

    extracted = extract_think_and_answer(completion_text)
    if extracted is None:
        return RewardResult(total=0.0, format_reward=0.0, accuracy_reward=0.0)

    _, answer_expr = extracted
    r_fmt = FORMAT_REWARD

    if format_only:
        return RewardResult(
            total=r_fmt,
            format_reward=r_fmt,
            accuracy_reward=0.0,
            parsed_answer=answer_expr,
        )

    try:
        value, used_numbers = safe_eval_countdown_expr(answer_expr)
    except (SyntaxError, RecursionError, ZeroDivisionError, ValueError, TypeError):
        return RewardResult(
            total=r_fmt,
            format_reward=r_fmt,
            accuracy_reward=0.0,
            parsed_answer=answer_expr,
        )

    is_valid_multiset = Counter(used_numbers) == Counter(numbers)
    is_target_match = value == Fraction(target)
    r_acc = ACCURACY_REWARD if (is_valid_multiset and is_target_match) else 0.0

    return RewardResult(
        total=r_fmt + r_acc,
        format_reward=r_fmt,
        accuracy_reward=r_acc,
        parsed_answer=answer_expr,
        evaluated_value=value,
    )
