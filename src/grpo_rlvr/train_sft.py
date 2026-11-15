"""Supervised fine-tuning (SFT) warmstart script for R1-style Countdown arithmetic."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from modern_decoder_llm import DecoderLLM
from torch.utils.data import DataLoader

from grpo_rlvr.data import SFTCountdownDataset
from grpo_rlvr.tokenizer import ArithmeticTokenizer


def build_model(model_cfg: dict[str, Any], vocab_size: int, device: torch.device) -> DecoderLLM:
    """Instantiate ``DecoderLLM`` from config, syncing ``vocab_size`` with the tokenizer."""
    model = DecoderLLM(
        vocab_size=vocab_size,
        max_seq_len=int(model_cfg["max_seq_len"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        depth=int(model_cfg["depth"]),
        n_heads=int(model_cfg["n_heads"]),
        n_kv_heads=int(model_cfg["n_kv_heads"]),
        head_dim=int(model_cfg["head_dim"]),
        ffn_dim=int(model_cfg["ffn_dim"]),
        rope_theta=float(model_cfg["rope_theta"]),
        norm_eps=float(model_cfg["norm_eps"]),
        tie_weights=True,
    )
    return model.to(device)


def compute_sft_loss(
    model: DecoderLLM,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Next-token cross-entropy loss with prompt/padding tokens masked to ``-100``."""
    logits, _ = model.forward(input_ids, start_pos=0, kv_caches=None)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )


def train_sft(
    cfg: dict[str, Any],
    data_path: Path,
    checkpoint_path: Path,
    max_steps: int,
    seed: int,
    init_from: Path | None = None,
) -> Path:
    """Run SFT warmstart training and save the final checkpoint."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = ArithmeticTokenizer()
    if not data_path.exists():
        raise FileNotFoundError(
            f"SFT dataset not found at {data_path}. Run `uv run python -m grpo_rlvr.data` first."
        )

    dataset = SFTCountdownDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        max_seq_len=int(cfg["model"]["max_seq_len"]),
    )
    train_cfg = cfg["training"]
    batch_size = int(train_cfg["batch_size"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model_cfg = dict(cfg["model"])
    model_cfg["vocab_size"] = tokenizer.vocab_size
    model = build_model(model_cfg, vocab_size=tokenizer.vocab_size, device=device)

    if init_from is not None:
        ckpt = torch.load(init_from, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_steps, eta_min=float(train_cfg["lr"]) * 0.1
    )
    grad_clip = float(train_cfg["grad_clip"])
    eval_interval = int(train_cfg["eval_interval"])

    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(loader)

    while step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad(set_to_none=True)
        loss = compute_sft_loss(model, input_ids, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        scheduler.step()

        step += 1
        running_loss += loss.item()

        if step % eval_interval == 0 or step == max_steps:
            avg_loss = running_loss / min(eval_interval, step)
            if step % eval_interval == 0:
                running_loss = 0.0
            lr_now = scheduler.get_last_lr()[0]
            print(f"[SFT] step {step:4d}/{max_steps} | loss={avg_loss:.4f} | lr={lr_now:.2e}")

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": model_cfg,
            "step": step,
            "seed": seed,
        },
        checkpoint_path,
    )
    print(f"Saved SFT checkpoint to {checkpoint_path}")
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SFT warmstart model for Countdown RLVR.")
    parser.add_argument("--config", type=Path, default=Path("configs/sft_warmstart.yaml"))
    parser.add_argument("--data-path", type=Path, default=Path("data/sft_train.json"))
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--init-from", type=Path, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    checkpoint_path = args.checkpoint_path or Path(cfg["training"]["checkpoint_path"])
    max_steps = args.max_steps if args.max_steps is not None else int(cfg["training"]["max_steps"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])

    train_sft(
        cfg=cfg,
        data_path=args.data_path,
        checkpoint_path=checkpoint_path,
        max_steps=max_steps,
        seed=seed,
        init_from=args.init_from,
    )


if __name__ == "__main__":
    main()
