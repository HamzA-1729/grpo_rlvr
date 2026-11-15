"""GRPO / Dr. GRPO RLVR training loop for Countdown arithmetic synthesis."""

from __future__ import annotations

import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from modern_decoder_llm import DecoderLLM

from grpo_rlvr.evaluate import evaluate_model, load_model_from_checkpoint
from grpo_rlvr.grpo_loss import compute_group_advantages, compute_grpo_loss
from grpo_rlvr.rollout import compute_sequence_logprobs, generate_rollouts
from grpo_rlvr.tokenizer import ArithmeticTokenizer
from grpo_rlvr.verifier import compute_reward


def freeze_reference_model(policy_model: DecoderLLM) -> DecoderLLM:
    """Create a frozen deep copy of ``policy_model`` to serve as ``pi_ref``."""
    ref_model = copy.deepcopy(policy_model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad_(False)
    return ref_model


def grpo_step(
    policy_model: DecoderLLM,
    ref_model: DecoderLLM,
    optimizer: torch.optim.Optimizer,
    batch_problems: list[dict[str, Any]],
    tokenizer: ArithmeticTokenizer,
    group_size: int = 8,
    max_new_tokens: int = 96,
    temperature: float = 1.0,
    mu_iters: int = 2,
    grad_clip: float = 1.0,
    eps_clip: float = 0.2,
    beta_kl: float = 0.04,
    kl_clamp: float = 20.0,
    normalize_std: bool = False,
    length_norm_mode: str = "constant",
    format_only: bool = False,
) -> dict[str, float]:
    """Execute one outer GRPO rollout + ``mu_iters`` policy-update step."""
    t0 = time.perf_counter()
    device = next(policy_model.parameters()).device
    bsz = len(batch_problems)

    prompt_ids = torch.tensor(
        [tokenizer.encode(item["prompt"]) for item in batch_problems],
        dtype=torch.long,
        device=device,
    )

    rollout = generate_rollouts(
        model=policy_model,
        prompt_ids=prompt_ids,
        group_size=group_size,
        max_new_tokens=max_new_tokens,
        eos_id=tokenizer.eos_id,
        pad_id=tokenizer.pad_id,
        temperature=temperature,
        greedy=False,
    )

    gen_tokens = rollout.sequences[:, rollout.prompt_len :]
    rewards_list: list[float] = []
    format_hits = 0
    acc_hits = 0
    trunc_hits = 0

    for p_idx, item in enumerate(batch_problems):
        for g_idx in range(group_size):
            row = p_idx * group_size + g_idx
            active_len = int(rollout.completion_mask[row].sum().item())
            comp_ids = gen_tokens[row, :active_len].tolist()
            comp_text = tokenizer.decode(comp_ids, skip_pad_eos=True)
            is_trunc = bool(rollout.truncated[row].item())
            res = compute_reward(
                completion_text=comp_text,
                numbers=item["numbers"],
                target=item["target"],
                truncated=is_trunc,
                format_only=format_only,
            )
            rewards_list.append(res.total)
            if res.format_reward > 0.0:
                format_hits += 1
            if res.accuracy_reward > 0.0:
                acc_hits += 1
            if is_trunc:
                trunc_hits += 1

    rewards = torch.tensor(rewards_list, dtype=torch.float32, device=device).view(bsz, group_size)
    advantages, zero_std_frac = compute_group_advantages(rewards, normalize_std=normalize_std)
    advantages_flat = advantages.reshape(-1)

    policy_model.eval()
    with torch.no_grad():
        log_pi_old = compute_sequence_logprobs(policy_model, rollout.sequences, rollout.prompt_len)
        log_pi_ref = compute_sequence_logprobs(ref_model, rollout.sequences, rollout.prompt_len)

    last_stats: dict[str, float] = {}
    last_loss = 0.0
    policy_model.train()
    for _ in range(mu_iters):
        log_pi_theta = compute_sequence_logprobs(policy_model, rollout.sequences, rollout.prompt_len)
        loss, last_stats = compute_grpo_loss(
            log_pi_theta=log_pi_theta,
            log_pi_old=log_pi_old,
            log_pi_ref=log_pi_ref,
            advantages=advantages_flat,
            completion_mask=rollout.completion_mask,
            eps_clip=eps_clip,
            beta_kl=beta_kl,
            kl_clamp=kl_clamp,
            length_norm_mode=length_norm_mode,
            max_new_tokens=max_new_tokens,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=grad_clip)
        optimizer.step()
        last_loss = float(loss.item())

    total_rollouts = bsz * group_size
    mean_len = float(rollout.completion_mask.sum(dim=1).mean().item())
    return {
        "reward_mean": float(rewards.mean().item()),
        "format_rate": format_hits / total_rollouts,
        "acc_rate": acc_hits / total_rollouts,
        "trunc_rate": trunc_hits / total_rollouts,
        "zero_std_frac": zero_std_frac,
        "mean_completion_len": mean_len,
        "loss": last_loss,
        "kl": last_stats["mean_k3_kl"],
        "clip_frac": last_stats["clip_fraction"],
        "step_time_sec": time.perf_counter() - t0,
    }


def train_grpo(
    cfg: dict[str, Any],
    sft_checkpoint: Path,
    train_data_path: Path,
    eval_data_path: Path,
    checkpoint_dir: Path,
    max_steps: int,
    seed: int,
    format_only: bool,
    eval_interval: int,
    eval_max_problems: int | None = None,
) -> Path:
    """Run full GRPO / Dr. GRPO RLVR training from an SFT checkpoint."""
    torch.manual_seed(seed)
    rng = random.Random(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = ArithmeticTokenizer()

    with train_data_path.open("r", encoding="utf-8") as f:
        train_problems = json.load(f)
    with eval_data_path.open("r", encoding="utf-8") as f:
        eval_problems = json.load(f)
    if eval_max_problems is not None:
        eval_problems = eval_problems[:eval_max_problems]

    policy_model, ckpt = load_model_from_checkpoint(sft_checkpoint, device=device)
    model_cfg = ckpt["model_cfg"]
    ref_model = freeze_reference_model(policy_model)

    rollout_cfg = cfg["rollout"]
    grpo_cfg = cfg["grpo"]

    optimizer = torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(grpo_cfg["lr"]),
        weight_decay=float(grpo_cfg["weight_decay"]),
    )

    prompts_per_step = int(rollout_cfg["prompts_per_step"])
    group_size = int(rollout_cfg["group_size"])
    max_new_tokens = int(rollout_cfg["max_new_tokens"])
    temperature = float(rollout_cfg["temperature"])

    history: list[dict[str, Any]] = []
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_pass1 = -1.0

    for step in range(1, max_steps + 1):
        batch_problems = rng.sample(train_problems, k=min(prompts_per_step, len(train_problems)))
        metrics = grpo_step(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            batch_problems=batch_problems,
            tokenizer=tokenizer,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            mu_iters=int(grpo_cfg["mu_iters"]),
            grad_clip=float(grpo_cfg["grad_clip"]),
            eps_clip=float(grpo_cfg["eps_clip"]),
            beta_kl=float(grpo_cfg["beta_kl"]),
            kl_clamp=float(grpo_cfg["kl_clamp"]),
            normalize_std=bool(grpo_cfg["normalize_std"]),
            length_norm_mode=str(grpo_cfg["length_norm_mode"]),
            format_only=format_only,
        )
        record: dict[str, Any] = {"step": step, **metrics}

        print(
            f"[GRPO s={seed}] step {step:3d}/{max_steps} | "
            f"R={metrics['reward_mean']:.3f} | "
            f"fmt={metrics['format_rate']:.3f} | "
            f"acc={metrics['acc_rate']:.3f} | "
            f"z_std={metrics['zero_std_frac']:.2f} | "
            f"KL={metrics['kl']:.4f} | "
            f"clip={metrics['clip_frac']:.3f} | "
            f"len={metrics['mean_completion_len']:.1f} | "
            f"{metrics['step_time_sec']:.2f}s"
        )

        if eval_interval > 0 and (step % eval_interval == 0 or step == max_steps):
            eval_res = evaluate_model(
                model=policy_model,
                problems=eval_problems,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                pass_k=group_size,
                temperature=temperature,
                device=device,
            )
            record["eval_pass_at_1"] = eval_res.pass_at_1
            record["eval_pass_at_k"] = eval_res.pass_at_k
            record["eval_format_rate"] = eval_res.format_rate_greedy
            print(
                f"  -> [EVAL step {step}] Pass@1={eval_res.pass_at_1:.3f} | "
                f"Pass@{eval_res.k}={eval_res.pass_at_k:.3f} | "
                f"Format={eval_res.format_rate_greedy:.3f} | "
                f"MeanLen={eval_res.mean_completion_len:.1f}"
            )
            if eval_res.pass_at_1 > best_pass1:
                best_pass1 = eval_res.pass_at_1
                torch.save(
                    {
                        "model_state_dict": policy_model.state_dict(),
                        "model_cfg": model_cfg,
                        "step": step,
                        "seed": seed,
                    },
                    checkpoint_dir / f"grpo_best_seed{seed}.pt",
                )

        history.append(record)

    final_ckpt = checkpoint_dir / f"grpo_final_seed{seed}.pt"
    torch.save(
        {
            "model_state_dict": policy_model.state_dict(),
            "model_cfg": model_cfg,
            "step": max_steps,
            "seed": seed,
        },
        final_ckpt,
    )
    with (checkpoint_dir / f"metrics_seed{seed}.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Saved final GRPO checkpoint to {final_ckpt}")
    return final_ckpt


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Countdown RLVR policy with GRPO / Dr. GRPO.")
    parser.add_argument("--config", type=Path, default=Path("configs/grpo.yaml"))
    parser.add_argument("--sft-checkpoint", type=Path, default=None)
    parser.add_argument("--train-data", type=Path, default=Path("data/grpo_train.json"))
    parser.add_argument("--eval-data", type=Path, default=Path("data/eval.json"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--format-only", action="store_true", default=None)
    parser.add_argument("--eval-interval", type=int, default=None)
    parser.add_argument("--eval-max-problems", type=int, default=None)
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    grpo_cfg = cfg["grpo"]
    sft_checkpoint = args.sft_checkpoint or Path(cfg["model"]["sft_checkpoint_path"])
    checkpoint_dir = args.checkpoint_dir or Path(grpo_cfg["checkpoint_dir"])
    max_steps = args.max_steps if args.max_steps is not None else int(grpo_cfg["max_steps"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    format_only = (
        args.format_only if args.format_only is not None else bool(grpo_cfg["format_only"])
    )
    eval_interval = (
        args.eval_interval if args.eval_interval is not None else int(grpo_cfg["eval_interval"])
    )

    train_grpo(
        cfg=cfg,
        sft_checkpoint=sft_checkpoint,
        train_data_path=args.train_data,
        eval_data_path=args.eval_data,
        checkpoint_dir=checkpoint_dir,
        max_steps=max_steps,
        seed=seed,
        format_only=format_only,
        eval_interval=eval_interval,
        eval_max_problems=args.eval_max_problems,
    )


if __name__ == "__main__":
    main()
