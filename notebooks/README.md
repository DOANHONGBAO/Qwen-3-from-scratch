# Notebooks

Explanatory notebooks that accompany each milestone. They import from the
`qwen3_scratch` package instead of redefining code, so the tested implementation
and the explanation never drift apart.

| Notebook | Milestone | Content |
|---|---|---|
| `00_environment_and_config.ipynb` | M0 | Environment report, `Qwen3Config`, parameter budget of Qwen3-0.6B |

Conventions:

- Install the extra once: `pip install -e ".[notebooks]"`, then `jupyter lab`.
- Commit notebooks **without outputs** (clear outputs before committing) to keep
  diffs reviewable and avoid leaking local paths.
- Notebooks never download weights implicitly; any download cell (M2+) is
  explicit and writes outside Git-tracked folders.
- Numbering: `NN_topic.ipynb`, matching roadmap items (e.g. `01_rmsnorm.ipynb` for M1.1).
