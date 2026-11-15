# Contributing to `grpo_rlvr`

Contributions, bug reports, and mathematical verification improvements are welcome.

## Development Setup

1. Clone the repository and sync the isolated `uv` environment:
   ```bash
   git clone https://github.com/HamzA-1729/grpo_rlvr.git
   cd grpo_rlvr
   uv sync
   ```
2. Run the automated test suite before submitting changes:
   ```bash
   uv run pytest -v
   ```
3. Format and lint with `ruff`:
   ```bash
   uv run ruff check src tests
   ```

## Standards
- Keep implementations pure PyTorch without high-level RL wrappers (`trl`, `verl`).
- Every new mathematical invariant or verifier rule must include a deterministic unit test in `tests/`.
