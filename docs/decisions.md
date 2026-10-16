# Decision log

Format: ID · date · decision · rationale · status.

## D-001 · 2026-10-16 · PyTorch reference first, C++20 second
Correctness is established in Python where tensors are easy to inspect and the
official HF implementation is available for parity; the C++ port then has a
verified oracle for every module. **Accepted.**

## D-002 · 2026-10-16 · Config field names = Hugging Face `config.json` keys
The official config is the source of truth for checkpoint compatibility, so
`Qwen3Config` uses `hidden_size`, `num_key_value_heads`, `head_dim`, ... rather than
the book's `emb_dim`, `n_kv_groups`, ... A mapping to Raschka's names is kept in
`RASCHKA_KEY_MAP` for cross-reading. **Accepted.**

## D-003 · 2026-10-16 · Config is framework-independent
`qwen3_scratch.config` does not import torch; dtype is a string. The same struct
is mirrored in C++ (`cpp/include/qwen3/config.hpp`) and both sides test the same
analytic parameter counts. **Accepted.**

## D-004 · 2026-10-16 · `max_position_embeddings` defaults to 32,768
R5 says 32,768; the educational reference uses 40,960. The value only sizes the
RoPE table and the maximum input, so the official value is the default and it
stays configurable. (Discrepancy X1.) **Accepted.**

## D-005 · 2026-10-16 · Module/parameter names follow the HF checkpoint
Using `q_proj`, `k_proj`, `gate_proj`, `input_layernorm`, ... makes weight loading
a near-identity mapping and parity debugging easier. (Discrepancy X10.)
**Proposed — confirm at M1.6/M2.**

## D-006 · 2026-10-16 · Tooling: setuptools + pip/conda, Ruff, pytest
Conda for the interpreter (project requirement), pip for Python packages
(official CPU torch index), setuptools build backend (same as R1). Ruff rule set
`E,F,W,I,B,UP,SIM,NPY` (broader than R1's `E4,E7,E9,F`) and Ruff formatter,
line length 100. **Accepted.**

## D-007 · 2026-10-16 · Three test tiers with automatic skipping
CPU (unmarked), `gpu`, `weights`, plus `slow`. Skipping is decided in
`tests/conftest.py` from `torch.cuda.is_available()` and `QWEN3_WEIGHTS_DIR`, so
CI and the laptop never need a GPU or weights. **Accepted.**

## D-008 · 2026-10-16 · Numerical precision order follows HF for parity
Where the educational and official code differ only in cast order (RMSNorm, RoPE,
softmax — discrepancies X5–X7), we follow HF so bf16 parity tests can be tight.
**Proposed — confirm per component in M1.**

## D-009 · 2026-10-16 · Real weight tying
When `tie_word_embeddings` is true the LM head will share the embedding
parameter (one tensor), not hold a copy. (Discrepancy X4.) **Proposed — M1.8.**

## D-010 · 2026-10-16 · License Apache-2.0
Matches the licenses of R1, R6 and the Qwen3 weights, and keeps attribution
simple. **Accepted, maintainer may change before first public release.**

## D-011 · 2026-10-16 · Do not commit local upstream reference copies
`documents/` (local copies of `modeling_qwen3.py` and `chC_main.ipynb`) is
git-ignored; we link to upstream instead of redistributing third-party code. The
official `config.json` (a small data file) is committed under `configs/hf/` with
attribution because tests depend on it. **Accepted.**

## D-012 · 2026-10-16 · C++ tests without a framework in M0
CTest runs a dependency-free executable. GoogleTest/Catch2 may be introduced in
M3 if test volume justifies a dependency (would need network at configure time).
**Accepted for M0.**
