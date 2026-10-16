# CLAUDE.md — instructions for coding agents in this repository

## Mission
Educational, reproducible, from-scratch implementation of the **Qwen3 dense
Transformer**: PyTorch first (M1–M2), then C++20 (M3–M4). Never wrap Hugging Face,
llama.cpp or another inference engine and call it "from scratch". Hugging Face is
allowed only as a *reference* for numerical verification (M2+).

## Milestones (do not skip; do not start a milestone without approval)
M0 foundation → M1 PyTorch modules → M2 HF parity, tokenizer, KV cache →
M3 C++20 CPU port → M4 GPU experiments/benchmarks. Status: `docs/roadmap.md`.

## Sources of truth
1. `configs/hf/Qwen3-0.6B-Base.config.json` (official config) and the matching
   `transformers` `modeling_qwen3.py` for exact checkpoint compatibility.
2. Raschka's *Build a Reasoning Model (From Scratch)* Appendix C as the
   educational reference. Known discrepancies: `docs/references.md`.

## Environment
- Fedora Linux, CPU only, conda env `qwen3-scratch`, Python 3.11, CPU torch.
- Never assume a GPU. Every unmarked test must run on CPU in seconds.
- Do not download pretrained checkpoints unless the task explicitly requires it
  (M2), and never commit them.

## Code rules
- Package code in `src/qwen3_scratch/`, one module per component (see
  `docs/architecture.md` → module map). Each module must have a C++ counterpart plan.
- Config field names = Hugging Face `config.json` keys. Never hardcode model
  dimensions; read them from `Qwen3Config`.
- `qwen3_scratch.config` must not import torch.
- Readability over speed until a component is verified. Type hints where useful.
- Every component ships with: math definition, shapes, implementation, synthetic
  example, unit tests (shapes, values, determinism, invalid inputs), reference
  comparison where possible, and "common mistakes" notes.

## Testing
- `pytest` (markers: `gpu`, `weights`, `slow`; GPU/weights tiers auto-skip).
- Fixed seeds; explicit `rtol`/`atol`; use `torch.testing.assert_close`.
- C++: CMake + CTest (`bash scripts/build_cpp.sh`).
- Quality: `ruff check .` and `ruff format --check .` must pass.
- Report results honestly as PASSED / FAILED / SKIPPED / NOT RUN. Never claim
  parity with Hugging Face without an executed comparison.

## Secrets
Never read, print, copy or commit credentials (Kaggle, GitHub, HF). Kaggle auth
lives in `~/.kaggle/` or env vars outside the repo. Never push a Kaggle kernel or
start a GPU session without an explicit experimental objective from the user.

## Git workflow (per task)
1. `git status` + branch. 2. Read related code/docs. 3. Short plan.
4. Smallest change. 5. Run tests/linters. 6. Review diff; check for secrets and
large files. 7. Update docs (`docs/progress.md`, roadmap, decisions).
8. Conventional Commit (`feat: implement rmsnorm`, `test: ...`, `docs: ...`),
   authored by the maintainer only: no `Co-Authored-By` or other agent trailers.
9. Push (never force-push, never rewrite history, never delete branches).
10. Report branch, commit SHA, push status.

## Communication
Explanations to the maintainer are in **Vietnamese**, keeping English
terminology, identifiers and math notation. Repository docs are in English.
