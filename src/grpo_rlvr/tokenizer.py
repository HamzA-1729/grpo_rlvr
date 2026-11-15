"""
Deterministic Character & Atomic-Tag Tokenizer for Countdown Arithmetic Reasoning.

Design:
- Atomic structural tags (<pad>, <eos>, <unk>, <think>, </think>, <answer>, </answer>)
  are matched greedily before single-character tokenization so tags are never split.
- Fixed 32-token vocabulary covering digits, arithmetic operators, parentheses,
  and fixed-width prompt characters ("T: 24|N: 2, 3, 4, 6\\n").
"""

import re
from typing import Dict, List, Sequence


ATOMIC_TAGS: List[str] = [
    "<pad>",
    "<eos>",
    "<unk>",
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
]

CHAR_TOKENS: List[str] = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "+", "-", "*", "/", "(", ")", "=",
    "T", "N", ",", "|", ":", ";", "\n", " ",
]

_TAG_SPLIT_REGEX = re.compile(
    "(" + "|".join(re.escape(tag) for tag in ATOMIC_TAGS) + ")"
)


class ArithmeticTokenizer:
    """Deterministic character-level tokenizer with atomic reasoning tags."""

    def __init__(self) -> None:
        all_tokens = ATOMIC_TAGS + CHAR_TOKENS
        self.token_to_id: Dict[str, int] = {tok: idx for idx, tok in enumerate(all_tokens)}
        self.id_to_token: Dict[int, str] = {idx: tok for idx, tok in enumerate(all_tokens)}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["<unk>"]

    @property
    def think_start_id(self) -> int:
        return self.token_to_id["<think>"]

    @property
    def think_end_id(self) -> int:
        return self.token_to_id["</think>"]

    @property
    def answer_start_id(self) -> int:
        return self.token_to_id["<answer>"]

    @property
    def answer_end_id(self) -> int:
        return self.token_to_id["</answer>"]

    def encode(self, text: str, add_eos: bool = False) -> List[int]:
        """Encodes text into token IDs, preserving atomic tags."""
        ids: List[int] = []
        for segment in _TAG_SPLIT_REGEX.split(text):
            if not segment:
                continue
            if segment in self.token_to_id:
                ids.append(self.token_to_id[segment])
            else:
                for ch in segment:
                    ids.append(self.token_to_id.get(ch, self.unk_id))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: Sequence[int], skip_pad_eos: bool = False) -> str:
        """Decodes token IDs back into a string."""
        skip_ids = {self.pad_id, self.eos_id} if skip_pad_eos else set()
        parts: List[str] = []
        for tid in token_ids:
            idx = int(tid)
            if idx in skip_ids:
                continue
            parts.append(self.id_to_token.get(idx, "<unk>"))
        return "".join(parts)
