"""
Unit tests for ArithmeticTokenizer:
1. Round-trip encode/decode on prompts and reasoning traces.
2. Atomic tag encoding (single token ID per structural tag).
3. Fixed-width prompt length invariance (always 20 tokens).
4. Unknown character fallback and skip_pad_eos decoding.
"""

from grpo_rlvr.tokenizer import ArithmeticTokenizer


def test_atomic_tags_are_single_tokens():
    tok = ArithmeticTokenizer()
    assert tok.vocab_size == 32

    assert tok.encode("<pad>") == [tok.pad_id]
    assert tok.encode("<eos>") == [tok.eos_id]
    assert tok.encode("<unk>") == [tok.unk_id]
    assert tok.encode("<think>") == [tok.think_start_id]
    assert tok.encode("</think>") == [tok.think_end_id]
    assert tok.encode("<answer>") == [tok.answer_start_id]
    assert tok.encode("</answer>") == [tok.answer_end_id]


def test_round_trip_encoding():
    tok = ArithmeticTokenizer()
    sample = "T: 24|N: 2, 3, 4, 6\n<think>4*2=8; 6-3=3; 3*8=24</think><answer>(6-3)*(4*2)</answer>"
    ids = tok.encode(sample, add_eos=True)

    assert tok.unk_id not in ids
    assert ids[-1] == tok.eos_id
    assert tok.decode(ids) == sample + "<eos>"
    assert tok.decode(ids, skip_pad_eos=True) == sample


def test_fixed_width_prompt_length_invariance():
    tok = ArithmeticTokenizer()
    prompts = [
        "T:  1|N: 1, 1, 1, 1\n",
        "T: 24|N: 2, 3, 4, 6\n",
        "T:100|N:12,12,12,12\n",
    ]
    for prompt in prompts:
        ids = tok.encode(prompt)
        assert len(ids) == 20, f"Prompt '{prompt}' encoded to {len(ids)} tokens, expected 20."


def test_unknown_char_fallback():
    tok = ArithmeticTokenizer()
    ids = tok.encode("1+z")
    assert ids[-1] == tok.unk_id
