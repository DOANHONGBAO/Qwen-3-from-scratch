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
**Proposed — confirm per component in M1.** RMSNorm: confirmed by D-013.

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

## D-013 · 2026-10-16 · RMSNorm follows the HF `Qwen3RMSNorm` operation order
Statistics in float32, cast the normalised activations back to the input dtype,
then multiply by `weight` (initialised to ones, no `1 + w` offset). This makes our
output bit-identical to HF in fp32/bf16/fp16 (tested), whereas the
"scale-then-cast" order of R2 and `torch.nn.RMSNorm` differs by 1 bf16 ulp on
~25 % of elements. The plain formula is kept separately as `rms_norm_textbook`
for teaching and as a float64 reference. **Accepted.**

## D-014 · 2026-10-16 · float64 inputs are not downcast in RMSNorm
Compute dtype is `promote_types(x.dtype, float32)`: fp16/bf16/fp32 → fp32,
float64 → float64. HF always uses float32 and would downcast float64. The
difference only affects float64 inputs (tests, `gradcheck`), never the model;
it is pinned by `tests/parity/test_rmsnorm_hf.py::test_documented_float64_difference`.
**Accepted.**

## D-015 · 2026-10-16 · HF parity tests are optional locally, required in CI
Parity tests import `transformers` via `pytest.importorskip` and construct
modules directly with random weights (no downloads). Locally they are SKIPPED
unless the `hf` extra is installed; CI installs `.[dev,hf]` so they always run
there. **Accepted.**

## D-016 · 2026-10-16 · Notebooks are executed by the test suite
`tests/test_notebooks.py` runs every notebook's code cells with `exec` (no Jupyter
dependency) and fails if a notebook is committed with outputs. Keeps notebooks in
sync with the package API. **Accepted.**

## D-017 · 2026-10-16 · SwiGLU module naming and bias policy
`SwiGLUFeedForward` uses HF parameter names (`gate_proj`, `up_proj`, `down_proj`)
so `Qwen3MLP` state dicts load with `strict=True`. `from_config` always builds it
without biases, matching HF (hardcoded `bias=False`, independent of
`attention_bias`). An explicit `bias` flag remains for teaching and tests.
**Accepted.**

## D-018 · 2026-10-16 · C++ parity fixtures are small exact JSON files
`scripts/export_fixtures.py` writes tiny fixtures (a few KB) to `tests/fixtures/`
as JSON whose numbers are exact float32 values. They are reviewable, need no
binary format, and are not blocked by the `*.npy`/`*.bin` ignore rules. They are
committed. `tests/test_fixtures.py` checks inputs and parameters bit-exactly and
outputs within the stored tolerance. Large fixtures for M3 will be generated at
test time instead. Supersedes the "generate as CTest step" note in
docs/components/rmsnorm.md for small fixtures. **Accepted.**
