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

Conventions:

- Install the extra once: `pip install -e ".[notebooks]"`, then `jupyter lab`.
  Plots need `matplotlib`; without it the notebooks print text tables instead.
- Commit notebooks **without outputs** (clear outputs before committing) to keep
  diffs reviewable and avoid leaking local paths. `tests/test_notebooks.py`
  executes every notebook's code cells and enforces this.
- Notebooks never download weights implicitly; any download cell (M2+) is
  explicit and writes outside Git-tracked folders.
- Numbering: `NN_topic.ipynb`, matching roadmap items (e.g. `01_rmsnorm.ipynb` for M1.1).
