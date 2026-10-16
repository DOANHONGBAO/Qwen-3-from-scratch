# References, licenses and reference analysis

## 1. Sources

| # | Source | Role | License |
|---|---|---|---|
| R1 | [rasbt/reasoning-from-scratch](https://github.com/rasbt/reasoning-from-scratch) | Educational reference (repo organisation, Appendix C Qwen3 code) | Apache-2.0 |
| R2 | [Appendix C notebook `chC_main.ipynb`](https://github.com/rasbt/reasoning-from-scratch/blob/main/chC/01_main-chapter-code/chC_main.ipynb) | Readable Qwen3 implementation | Apache-2.0 |
| R3 | [R1 `pyproject.toml`](https://github.com/rasbt/reasoning-from-scratch/blob/main/pyproject.toml) | Dependency/tooling reference | Apache-2.0 |
| R4 | [Qwen/Qwen3-0.6B-Base model card](https://huggingface.co/Qwen/Qwen3-0.6B-Base) | Target checkpoint | Apache-2.0 |
| R5 | [Qwen3-0.6B-Base `config.json`](https://huggingface.co/Qwen/Qwen3-0.6B-Base/blob/main/config.json) | **Source of truth** for hyperparameters; copied to `configs/hf/` | Apache-2.0 |
| R6 | [transformers `modeling_qwen3.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3/modeling_qwen3.py) | **Source of truth** for exact numerics | Apache-2.0 |
| R7 | [Kaggle/kaggle-cli](https://github.com/Kaggle/kaggle-cli) ([docs](https://github.com/Kaggle/kaggle-cli/blob/main/docs/README.md), [kernels](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md)) | Remote GPU workflow | Apache-2.0 |

Attribution policy: we do not copy substantial upstream code. Small, well-known
formulas (e.g. `rotate_half`) are re-derived and written independently; when a
snippet is adapted, the file carries a comment naming the source and its license.
The only verbatim third-party file in the repo is the official `config.json`
(`configs/hf/Qwen3-0.6B-Base.config.json`, Apache-2.0, © Qwen team), kept for
tests. Local reference copies of R2 and R6 (in `documents/`) are git-ignored.

Analysed on 2026-10-16. R6 was read from a local copy of the current `main`
branch (generated from `modular_qwen3.py`, uses `_tied_weights_keys` as a dict and
kernel-hub decorators, i.e. transformers v5-era code); R5 was produced with
`transformers_version 4.51.0`.

## 2. How R1 (Raschka) organises the project

- **Layout.** Chapter folders `ch02`–`ch08` and appendices `chC`–`chG`, each with
  a `*_main.ipynb` and exercise solutions; a pip-installable helper package
  `reasoning_from_scratch/` that the notebooks import from; a top-level `tests/`.
- **Dependencies** (`pyproject.toml`): setuptools build, `requires-python
  >=3.10,<3.15`, runtime `torch>=2.10,<3`, `tokenizers`, `jupyterlab`,
  `sympy`, `matplotlib`, `nbformat`; a `dev` dependency group (`pytest`, `ruff`,
  `pytest-ruff`, `transformers`, `huggingface-hub`, `safetensors`); optional
  `extra` (`datasets`, `chainlit`). Ruff selects only `E4,E7,E9,F`. `uv` is the
  suggested installer, with a CUDA 13 torch index only for Linux aarch64.
  `requirements.txt` is a thin list (`reasoning-from-scratch`, `torch`, `tokenizers`,
  `sympy`, `setuptools`). Hardware: chapters 2–4 run on CPU, GPU recommended for 5–6.
- **Model initialisation.** A plain dict config (`QWEN_CONFIG_06_B`) with
  book-specific keys (`emb_dim`, `n_heads`, `n_kv_groups`, `hidden_dim`,
  `rope_base`, `context_length`, `dtype`). `Qwen3Model(cfg)` builds
  `tok_emb`, `trf_blocks`, `final_norm`, `out_head`, and precomputes RoPE `cos`/`sin`
  buffers for the full context length (`persistent=False`).
- **Tokenizer.** `Qwen3Tokenizer` wraps the `tokenizers` library loaded from a
  `tokenizer-*.json`, splits the prompt on special tokens with a regex, maps them
  directly to IDs, and optionally wraps a chat template. EOS is `<|endoftext|>`
  for base models and `<|im_end|>` otherwise, inferred from the file name.
- **Weight loading.** `download_qwen3_small(kind="base")` fetches a
  pre-converted PyTorch state dict (`qwen3-0.6B-base.pth`) whose keys already use
  the book's names, so `model.load_state_dict(torch.load(...))` works directly.
  The HF→book name mapping is hidden in that conversion step.
- **Generation.** `generate_text_basic_stream_cache` (from `ch02`) streams tokens
  with a `KVCache` object (list of per-layer `(k, v)` tuples) and the model's own
  `current_pos` counter; `reset_kv_cache()` must be called between prompts.

What we adopt: installable package + tests, small explicit modules, a tiny
config for tests, the KV-cache-as-list-of-tuples idea. What we change: config keys
follow HF (D-002), module and parameter names follow the HF checkpoint (D-005),
weights will be loaded from the official safetensors with an explicit, tested
name mapping (M2), and lint rules are broader (D-006).

## 3. Discrepancies between R2 (educational) and R5/R6 (official)

These are recorded, not resolved, in M0. Each must be decided and tested when the
corresponding component is implemented.

| # | Topic | Raschka Appendix C (R2) | Official config / HF code (R5, R6) | Impact / plan |
|---|---|---|---|---|
| X1 | **Context length** | `context_length = 40_960` (prompt 8,192 + output 32,768, per the notebook) | `max_position_embeddings = 32768` | Not a weight difference; only affects the RoPE table size and max input. We use 32,768 from R5 by default; configurable. |
| X2 | **Head dimension** | `head_dim = 128` (explicit), falls back to `emb_dim // n_heads` if `None` | `head_dim = 128`; HF falls back to `hidden_size // num_attention_heads` | Consistent. Key lesson: `128 ≠ 1024/16`. Our config requires explicit `head_dim` and derives it only for HF configs that omit it. |
| X3 | **QK-Norm** | `qk_norm=True`; q/k norms hardcode `eps=1e-6`; applied after `view`+`transpose`, before RoPE | `q_norm`/`k_norm` use `config.rms_norm_eps`; applied after `view`, before `transpose`, before RoPE | Mathematically equivalent (norm over last dim). We will take ε from the config. |
| X4 | **Weight tying** | Separate `out_head` `nn.Linear`; no tying in the module (the pre-converted `.pth` presumably stores a copy of the embedding) | `tie_word_embeddings = true`; `_tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}` | Same numbers if copied, but untied training would diverge and memory doubles (+155.6M params). We will implement real tying when configured and test parameter identity. |
| X5 | **RMSNorm precision order** | Casts to fp32, normalises, multiplies by `scale` **in fp32**, then casts back | Casts to fp32, normalises, casts back to input dtype, **then** multiplies by `weight` | Identical in fp32; small differences in bf16. Follow HF order for parity; document. |
| X6 | **RoPE precision** | `cos`/`sin` kept in fp32; rotation computed in fp32 then cast to `x.dtype` | `cos`/`sin` cast to `x.dtype` (bf16) **before** the rotation | bf16 differences only. Follow HF for parity tests. |
| X7 | **Softmax precision** | `softmax` in the input dtype; scaling applied after masking | `softmax(..., dtype=float32)` then cast; scaling applied to scores before adding mask | bf16 differences only. Follow HF for parity. |
| X8 | **Masking** | Boolean `triu` mask + `masked_fill(-inf)`, sliced for cached decoding | Additive mask built by `create_causal_mask`, supports padding masks | Equivalent for unpadded input. Padding/batched generation is out of scope until M2. |
| X9 | **RMSNorm ε** | Default `1e-6` hardcoded in several places | `rms_norm_eps = 1e-6` from config | Same value; we read it from the config. |
| X10 | **Parameter naming** | `W_query`, `W_key`, `W_value`, `out_proj`, `fc1/fc2/fc3`, `norm1/norm2`, `tok_emb`, `final_norm`, `out_head`, `scale` | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj/up_proj/down_proj`, `input_layernorm/post_attention_layernorm`, `embed_tokens`, `norm`, `lm_head`, `weight` | We use HF names (D-005). |
| X11 | **dtype** | Config `dtype = bfloat16` for all `nn.Linear`/`nn.Embedding` | `torch_dtype = bfloat16` in checkpoint | Tests use fp32 by default; bf16 parity in M2. |
| X12 | **RoPE API** | Precomputed tables for the whole context, sliced with `offset` | Computed per forward from `position_ids` | Equivalent; we precompute (simpler for C++) and test against HF. |

Other R5 fields we deliberately do not model yet: `sliding_window`,
`max_window_layers`, `use_sliding_window=false` (dense model, no SWA),
`rope_scaling=null`, `attention_dropout=0.0`, `bos/eos_token_id=151643`
(used by the tokenizer/generation in M2).

## 4. Kaggle CLI (R7) facts used in `docs/kaggle.md`

- Install: `pip install kaggle`.
- Authentication options (current docs): `kaggle auth login` (OAuth);
  `KAGGLE_API_TOKEN` env var; token file `~/.kaggle/access_token`; legacy
  `~/.kaggle/kaggle.json` from "Create Legacy API Key".
- Kernels: `kaggle kernels init -p DIR`, `kaggle kernels push -p DIR
  [--accelerator ID] [-t SECONDS] [--no-run]`, `kaggle kernels status
  owner/slug`, `kaggle kernels output owner/slug -p PATH`, `kaggle kernels pull`.
- Accelerator IDs listed in the docs include `NvidiaTeslaT4`, `NvidiaL4`,
  `NvidiaTeslaA100`, `NvidiaH100` (some restricted).
- The kernels doc does not document `kernel-metadata.json` fields; our template
  uses the established fields (`id`, `title`, `code_file`, `language`,
  `kernel_type`, `is_private`, `enable_gpu`, `enable_tpu`, `enable_internet`,
  `*_sources`). **To verify** with `kaggle kernels init` once the CLI is installed.
