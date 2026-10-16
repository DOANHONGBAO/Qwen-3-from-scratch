# Notebooks

Explanatory notebooks that accompany each milestone. They import from the
`qwen3_scratch` package instead of redefining code, so the tested implementation
and the explanation never drift apart.

| Notebook | Milestone | Content |
|---|---|---|
| `00_environment_and_config.ipynb` | M0 | Environment report, `Qwen3Config`, parameter budget of Qwen3-0.6B |
| `01_rmsnorm.ipynb` | M1.1 | RMSNorm: intuition, math, manual example, shapes, LayerNorm vs RMSNorm, HF parity, gradients, precision |
| `02_swiglu.ipynb` | M1.2 | SwiGLU: FFN variants, SiLU vs ReLU/GELU plot, manual example, shapes, HF parity, two-branch gradients |
| `03_rope.ipynb` | M1.3 | RoPE: 2D rotation plot, frequencies vs theta, worked example, relative positions, split-half vs interleaved, KV-cache positions, HF parity, gradients |
| `04_qk_norm_gqa.ipynb` | M1.4 | Attention from scratch: causal mask heatmaps, MHA/GQA/MQA head mapping, KV-cache sizes, QK-Norm logit bound, full trace, leakage check, HF parity, gradients |
| `05_decoder_block.ipynb` | M1.5 | Decoder block: equations, shapes, residual identity, pre-norm vs post-norm depth experiment, gradients, HF parity |
| `06_tiny_qwen3.ipynb` | M1.6 | Full Tiny Qwen3: architecture, config, init, parameter budget plot, embeddings, stacked blocks, final norm, weight tying, next-token distribution plot, shifted loss, backprop, leakage check, greedy decoding, HF parity, why random outputs are meaningless |
| `07_pretrained_qwen3.ipynb` | M2.1 | Official weights: safetensors header by hand, name map, bf16 vs fp32, loading without random init, tied head, tokenizer, logits vs HF, next-token probabilities, completion, diagnosing a wrong mapping (synthetic checkpoint by default; real weights opt-in) |
| `08_kv_cache_generation.ipynb` | M2.2 | Autoregressive generation, full-prefix recomputation, prefill vs decode, KV cache shapes and GQA memory, RoPE offsets, masks with T_q ≠ T_k, temperature/top-k/top-p, cached vs uncached numerics, latency, common bugs (real model opt-in) |

Conventions:

- Install the extra once: `pip install -e ".[notebooks]"`, then `jupyter lab`.
  Plots need `matplotlib`; without it the notebooks print text tables instead.
- Commit notebooks **without outputs** (clear outputs before committing) to keep
  diffs reviewable and avoid leaking local paths. `tests/test_notebooks.py`
  executes every notebook's code cells and enforces this.
- Notebooks never download weights. 07 and 08 use a synthetic random-weight
  checkpoint or the Tiny model by default; the real Qwen3-0.6B-Base sections run only
  with `QWEN3_WEIGHTS_DIR=<checkpoint dir>` and `QWEN3_NB_PRETRAINED=1` (~4 GiB RAM).
- Numbering: `NN_topic.ipynb`, matching roadmap items (e.g. `01_rmsnorm.ipynb` for M1.1).
