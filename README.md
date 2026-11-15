# Group Relative Policy Optimization (GRPO) & RLVR from Scratch

[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.5+](https://img.shields.io/badge/PyTorch-2.5%2B-EE4C2C.svg)](https://pytorch.org/)
[![Package Manager: uv](https://img.shields.io/badge/managed%20by-uv-DE5FE9.svg)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A clean, first-principles PyTorch implementation of **Group Relative Policy Optimization (GRPO / Dr. GRPO)** with **Reinforcement Learning with Verifiable Rewards (RLVR)** built from scratch without high-level RL wrappers (`trl`, `verl`). Features a critic-free group advantage estimator, Dr. GRPO normalization switches, a safe AST-based `fractions.Fraction` math verifier, fixed-width batched `StaticKVCache` rollouts, and R1-style SFT-warmstarted Countdown arithmetic training.

---

## 🌟 Core Capabilities & Key Highlights

* **Critic-Free Policy Optimization (GRPO & Dr. GRPO):** Eliminates the PPO Critic network completely by sampling $G$ completions per prompt and normalizing rewards within each group, with configurable **Dr. GRPO** switches (`normalize_std`, `length_norm_mode`) to address difficulty and length biases.
* **Deterministic Verifiable Rewards (RLVR):** Replaces learned neural reward models with a safe `ast` + `fractions.Fraction` mathematical evaluator and strict multiset checker—requiring zero extra model memory and closing arithmetic shortcut loopholes.
* **Exact Importance Sampling & $k_3$ KL Penalty:** Samples rollouts at $\mathrm{temperature} = 1.0, \mathrm{top\_p} = 1.0$ and recomputes $\log \pi_{\theta_{\text{old}}}$ and $\log \pi_{\text{ref}}$ via full uncached forward passes (`eval()`, `no_grad()`), guaranteeing $\rho_{i,t} = 1.0$ on the first inner iteration and using Schulman's clamped non-negative $k_3$ value estimator.
* **Fixed-Width Batched `StaticKVCache` Rollouts:** Uses space-padded fixed-width prompts (`prompt_len = 20`, zero `<pad>` tokens) so all $G \times B$ trajectories share `StaticKVCache.cur_len` cleanly without left-padding corruption, while including the first `EOS` token in the loss mask so the model learns when to stop.
* **Pinned `DecoderLLM` Dependency:** Imports `DecoderLLM` and `StaticKVCache` directly from `modern_decoder_llm` pinned at tag `v0.1.0` via `uv.lock`—zero duplicated model code and 100% reproducible clones.

---

## 📐 Architecture: R1-Style GRPO Loop

```text
Prompt q (Fixed 20 tokens: "T: 24|N: 2, 3, 4, 6\n")
       │
       ▼  Batched StaticKVCache Rollout (T=1.0, top_p=1.0)
┌──────────────────────────────────────────────────────────┐
│ Group of G=8 Completions: {o_1, o_2, ..., o_G}           │
│   <think>...</think><answer>(6-3)*(4*2)</answer><eos>    │
└──────────────────────────────────────────────────────────┘
       │
       ├──► Deterministic AST Verifier (Format + Multiset + Fraction == Target)
       │         └──► Rewards {r_1, ..., r_G} ──► Group Advantages {A_1, ..., A_G}
       │
       ├──► Full Forward Pass (Frozen Reference pi_ref) ──► log_pi_ref
       │
       └──► Full Forward Pass (Trainable Policy pi_theta) ──► log_pi_theta
                 │
                 ▼
       Clipped Surrogate Loss - beta * k3_KL  ──► AdamW Step on pi_theta
```

---

## ⚖️ PPO vs. GRPO

| Dimension | PPO (with Verifier / Reward Model) | GRPO (Group Relative Policy Optimization) |
| :--- | :--- | :--- |
| **Models in Memory** | **3 models** with a verifier ($\pi_\theta, \pi_{\mathrm{ref}}, V_\psi$); **4 models** with a learned Reward Model ($R_\phi$) | **2 models:** Policy ($\pi_\theta$) and frozen Reference ($\pi_{\mathrm{ref}}$) |
| **Baseline Estimation** | Learned **Critic network $V_\psi(s_t)$** trained via MSE against returns (GAE) | **Group mean** (and optional std) of rewards across $G$ sampled completions |
| **Sparse Terminal Rewards** | Harder to train token-level value attribution from a single terminal score | Naturally suited: one outcome reward $r_i$ yields one trajectory advantage $\hat{A}_i$ |
| **KL Regularization** | Typically folded into per-token reward | Added directly as a non-negative penalty in the loss ($k_3$ value estimator) |

---

## 📚 Mathematical Formulations

### 1. Group-Relative Advantage (GRPO vs. Dr. GRPO)
Given $G$ completions $\{o_1, \dots, o_G\}$ for prompt $q$ with verifier rewards $\{r_1, \dots, r_G\}$:

* **Dr. GRPO (`normalize_std = false`):** Removes $\mathrm{std}(r)$ division to prevent difficulty bias (exploding gradients on near-uniform groups):
  $$\hat{A}_i = r_i - \mathrm{mean}(\{r_1, \dots, r_G\})$$
* **Standard GRPO (`normalize_std = true`):**
  $$\hat{A}_i = \frac{r_i - \mathrm{mean}(\{r_1, \dots, r_G\})}{\mathrm{std}(\{r_1, \dots, r_G\}) + \epsilon_{\mathrm{std}}}$$

### 2. Clipped Surrogate Objective & Clamped $k_3$ KL Value Estimator
With token-level ratio $\rho_{i,t}(\theta) = \exp\big(\log \pi_\theta(o_{i,t} \mid q, o_{i,<t}) - \log \pi_{\theta_{\mathrm{old}}}(o_{i,t} \mid q, o_{i,<t})\big)$ and completion mask $m_{i,t} \in \{0, 1\}$ (covering generated tokens up to and including the first `EOS`):

$$\mathcal{J}_{\mathrm{GRPO}}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{Z_i} \sum_{t=1}^{|o_i|} m_{i,t} \left[ \min\Big(\rho_{i,t}(\theta)\hat{A}_i, \; \mathrm{clip}(\rho_{i,t}(\theta), 1-\epsilon_{\mathrm{clip}}, 1+\epsilon_{\mathrm{clip}})\hat{A}_i\Big) - \beta \, \hat{D}_{\mathrm{KL}}^{(k_3)}(i, t) \right]$$

Where $Z_i = L_{\max}$ (`"constant"` per Dr. GRPO) or $\sum_t m_{i,t}$ (`"sequence"`), and $\log r_{i,t} = \mathrm{clamp}(\log \pi_{\mathrm{ref}} - \log \pi_\theta, -20.0, 20.0)$:
$$\hat{D}_{\mathrm{KL}}^{(k_3)}(i, t) = \exp(\log r_{i,t}) - \log r_{i,t} - 1 \ge 0$$

---

## 📊 Empirical Results & Diagnostics

### Dataset & Space-Size Gate
Exhaustive enumeration over 4 operands (`1..12`) and targets (`1..100`) yields **67,526** solvable `(numbers, target)` pairs, partitioned into disjoint splits:
* `sft_train`: **2,000** | `grpo_train`: **2,000** | `eval`: **500** (`15.0x` total-to-split ratio).

### Held-Out Evaluation (`n = 500`, `DecoderLLM` `depth=4, hidden_dim=256`, CPU)

| Method | Seed | `Pass@1` (Greedy) | `Pass@8` | Format Rate | Mean Len | Truncation |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SFT Warmstart (`1200` steps)** | 42 | `0.038` | `0.120` | `0.998` | `45.9` | `0.000` |
| **SFT Compute-Matched (`+300` steps)** | 42 | `0.040` | `0.122` | `0.998` | `46.2` | `0.000` |
| **Dr. GRPO (`150` steps, `lr=1e-4, beta_kl=0.04`)** | 42 | `0.032` | `0.098` | `1.000` | `45.9` | `0.000` |
| **Dr. GRPO (`150` steps, `lr=1e-4, beta_kl=0.04`)** | 1337 | `0.042` | `0.086` | `0.998` | `46.1` | `0.000` |
| **Dr. GRPO Aggressive (`150` steps, `lr=3e-4, beta_kl=0.005, mu=4`)** | 42 | `0.002` | `0.020` | `0.994` | `45.8` | — |

### Diagnostic Findings
* **Format & Stopping Mastery:** SFT warmstart reaches `0.998` format compliance and `0.000` truncation (`MeanLen = 45.9 <= 96`), passing the learnability gate with a `>3x` exploration gap (`Pass@1 = 3.8%` vs. `Pass@8 = 12.0%`).
* **Zero-Variance Group Bottleneck (`z_std`):** At `Pass@8 = 12.0%` and `G = 8`, `~75%–94%` of prompt groups per batch produce uniform rewards (`z_std`), yielding zero advantage signal on those groups.
* **KL Regularization Sensitivity:** At `lr = 1e-4, beta_kl = 0.04, mu_iters = 2`, Dr. GRPO stayed stable (`Format = 0.998–1.000`, `Pass@1 = 0.032 / 0.042`), whereas weakening the KL anchor (`beta_kl = 0.005, lr = 3e-4, mu_iters = 4`) caused policy collapse (`Pass@1` dropped to `0.002`, `Pass@8` to `0.020`).

---

## 📁 Repository Structure

```text
grpo_rlvr/
├── README.md
├── pyproject.toml               # uv config pinning modern-decoder-llm @ v0.1.0
├── LICENSE
├── CONTRIBUTING.md
├── CITATION.cff
├── .pre-commit-config.yaml
├── configs/
│   ├── sft_warmstart.yaml
│   └── grpo.yaml
├── checkpoints/                 # Saved SFT & GRPO policy weights (git-ignored)
├── outputs/                     # Evaluation reports & plots (git-ignored)
├── src/grpo_rlvr/
│   ├── __init__.py
│   ├── tokenizer.py             # Deterministic arithmetic & CoT tag tokenizer
│   ├── data.py                  # Procedural Countdown generator, solver & disjoint splits
│   ├── verifier.py              # Safe AST Fraction evaluator & multiset reward verifier
│   ├── grpo_loss.py             # Dr. GRPO / GRPO objective & clamped k3 KL
│   ├── rollout.py               # Fixed-width StaticKVCache rollouts & full-forward log-probs
│   ├── train_sft.py             # SFT warmstart & compute-matched baseline trainer
│   ├── train_grpo.py            # GRPO RLVR loop & diagnostics logger
│   └── evaluate.py              # Pass@1, Pass@G, format rate & 95% Wilson CIs
└── tests/
    ├── __init__.py
    ├── test_tokenizer.py
    ├── test_data.py
    ├── test_verifier.py
    ├── test_grpo_loss.py
    ├── test_rollout.py
    └── test_grpo_step.py
```

---

## 🚀 Quickstart & Reproducibility

```bash
git clone https://github.com/HamzA-1729/grpo_rlvr.git
cd grpo_rlvr
uv sync
uv run pytest -v

# 1. Generate disjoint Countdown splits (data/*.json)
uv run python -m grpo_rlvr.data

# 2. Train & evaluate SFT warmstart
uv run python -m grpo_rlvr.train_sft
uv run python -m grpo_rlvr.evaluate --checkpoint checkpoints/sft_model.pt

# 3. Train Dr. GRPO RLVR policy
uv run python -m grpo_rlvr.train_grpo --seed 42
```

---

## 📖 References

1. **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models** — Shao et al., 2024 ([arXiv:2402.03300](https://arxiv.org/abs/2402.03300))
2. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning** — DeepSeek-AI, 2025 ([arXiv:2501.12948](https://arxiv.org/abs/2501.12948))
3. **Understanding R1-Zero-Like Training: A Critical Perspective (Dr. GRPO)** — Liu et al., 2025 ([arXiv:2503.20783](https://arxiv.org/abs/2503.20783))
4. **Approximating KL Divergence** — John Schulman, 2020 ([joschu.net](http://joschu.net/blog/kl-approx.html))

---

## 📜 License
Distributed under the [MIT License](LICENSE).
