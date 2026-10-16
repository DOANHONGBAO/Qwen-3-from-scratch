# Official weights, tokenizer and parity (M2.1)

Code: [`weight_loader.py`](../../src/qwen3_scratch/weight_loader.py),
[`pretrained.py`](../../src/qwen3_scratch/pretrained.py),
[`tokenizer.py`](../../src/qwen3_scratch/tokenizer.py).
Scripts: `scripts/verify_pretrained.py`, `scripts/run_pretrained_qwen3.py`,
`scripts/m2_local_run.sh`. Tests: `tests/test_weight_loader.py`,
`tests/test_tokenizer.py`, `tests/parity/test_pretrained_hf.py`,
`tests/test_pretrained_weights.py` (weights tier).
Notebook: [`07_pretrained_qwen3.ipynb`](../../notebooks/07_pretrained_qwen3.ipynb).
Results on real weights: [`docs/progress.md`](../progress.md) and `results/m2/`.

## 1. Checkpoint

| item | value |
|---|---|
| repository | `Qwen/Qwen3-0.6B-Base` (the **base** model; never silently replaced by the instruct `Qwen3-0.6B`) |
| pinned revision | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` (Hub `lastModified` 2025-07-26) |
| files used | `config.json`, `generation_config.json`, `model.safetensors`, `tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`, `LICENSE` |
| weights | one `model.safetensors`, 310 tensors, all **BF16**, 596,049,920 values (~1.11 GiB) |
| `lm_head.weight` | absent: tied to `model.embed_tokens.weight` (596,049,920 = tied parameter count) |
| config | `transformers_version` 4.51.0; top-level `rope_theta` = 1e6; no sliding window, no rope scaling |

`download_checkpoint()` uses `huggingface_hub.snapshot_download(repo, revision,
allow_patterns=...)` and is the **only** function that touches the network; tests never
call it. The files live in the normal HF cache (`~/.cache/huggingface`), never in the
repository (`*.safetensors` is git-ignored).

**Safetensors**: 8-byte header length, JSON header (`dtype`, `shape`, `data_offsets`
per tensor), raw bytes. No pickle, so loading cannot run code. `SafetensorsCheckpoint`
memory-maps the file(s) (single file or sharded index), reads one tensor at a time,
and refuses `.bin`/`.pt` pickles.

## 2. Name mapping and checks

The map is generated from the config (`build_name_map`, `expected_shapes`), not copied
from a `state_dict`:

```
model.embed_tokens.weight                         [V, H]
model.layers.{i}.input_layernorm.weight           [H]
model.layers.{i}.self_attn.{q,k,v}_proj.weight    [Nq·D | Nkv·D, H]   (nn.Linear: [out, in])
model.layers.{i}.self_attn.o_proj.weight          [H, Nq·D]
model.layers.{i}.self_attn.{q,k}_norm.weight      [D]
model.layers.{i}.post_attention_layernorm.weight  [H]
model.layers.{i}.mlp.{gate,up}_proj.weight        [I, H]
model.layers.{i}.mlp.down_proj.weight             [H, I]
model.norm.weight                                 [H]
lm_head.weight                                    [V, H]   (untied models only)
```

`load_into_model(model, checkpoint, strict=True)`:

* checks names and shapes **before** writing anything; reports `missing`,
  `unexpected`, `shape_mismatch` with the offending names and both shapes;
* non-floating tensors, duplicate names across shards, a parameter assigned twice,
  or any parameter left unwritten raise `CheckpointError`;
* `strict=False` only tolerates *unexpected extra* tensors; missing tensors and shape
  mismatches always fail (they would leave random weights behind);
* tied models: `lm_head.weight` may be absent; if present it must equal the embedding
  exactly; after loading `lm_head.weight is embed_tokens.weight`;
* dtype conversion is explicit (`tensor.to(param.dtype)`) and counted in
  `report.dtype_conversions` (e.g. `{"bfloat16->float32": 310}`). bf16 → fp32 is exact.

`load_pretrained(dir, dtype, device)` builds the model on the `meta` device,
allocates uninitialised storage with `to_empty`, rebuilds the RoPE buffers, re-ties the
head and fills every parameter from the checkpoint. No random initialisation is run, and
no second copy of the model is made. Measured on a random-weight model with the exact
0.6B shapes (sandbox, 2 vCPU): ~2 s, peak RSS ~4.0 GiB in float32 including ~1.1 GiB of
memory-mapped file pages that the kernel can drop; RSS after loading ~2.9–3.0 GiB.

`config.json` parsing accepts both the official format (top-level `rope_theta`) and the
transformers ≥ 5 format (`rope_parameters`, `layer_types`); anything else unsupported
(rope scaling, sliding windows, non-SiLU) raises.

## 3. Precision

* **float32** is the correctness reference on CPU (exact copy of the bf16 weights).
* **bfloat16** halves memory. It is *not* assumed to be faster on CPU: the target
  laptop (Ryzen 3 3250U) has AVX2/FMA but no native bf16 instructions.
* Tolerances (relative L2 error, `‖a − b‖ / ‖b‖`): float32 ≤ 1e-5 for every hidden
  state and the logits; bfloat16 ≤ 2e-2. Rationale: both implementations execute the
  same operations in the same order, so in float32 we expect rounding-level
  agreement (often exactly 0); a mapping, RoPE or masking bug produces errors of 1e-2…1.
  Tolerances are never loosened to make a run pass.

## 4. Tokenizer adapter

`Qwen3Tokenizer` wraps the checkpoint's `tokenizer.json` with the Rust `tokenizers`
library (the engine behind the HF fast tokenizer) and reads `tokenizer_config.json`.
It is separate from the model, which only sees ids. Facts (revision da87bfb):

| | |
|---|---|
| vocabulary | 151,669 ids (151,643 BPE + 26 added) < model `vocab_size` 151,936 (267 padded rows) |
| BOS | none (`add_bos_token: false`, `bos_token: null`; `config.json`'s `bos_token_id` is not added) |
| EOS / PAD | `<|endoftext|>` = 151643 |
| special | 151643–151656 (14 tokens: `<|endoftext|>`, `<|im_start|>`, `<|im_end|>`, …) |
| normaliser | NFC, so `decode(encode(s)) == NFC(s)` |

Special-token text in the input becomes the special id (as in transformers); pass
`allow_special_tokens=False` to reject it. `decode` rejects ids outside the tokenizer
vocabulary unless `skip_out_of_vocab=True` (a model can, in principle, pick a padded
row). `encode_batch(padding=True, padding_side=...)` returns ids and masks; padded
batches are not supported by the cached generation path.

## 5. Parity procedure (`scripts/verify_pretrained.py`)

Never two models in memory: two child processes.

1. **reference**: HF `Qwen3ForCausalLM.from_pretrained(dir, dtype, attn_implementation="eager")`,
   `eval()`; `AutoTokenizer` ids; forward hooks on `embed_tokens`, every decoder layer and
   the final norm; logits; greedy ids from `transformers.generate` (16 new tokens).
   Saved to `results/m2/reference_<dtype>.safetensors` (git-ignored) + JSON.
2. **ours**: `load_pretrained` + `Qwen3Tokenizer`; same hooks on the **same token ids**;
   per tensor: max/mean absolute error, RMS error, relative L2; first diverging tensor;
   top-1 agreement and top-5 overlap of the logits, with the reference top-2 margin of any
   disagreeing position (near-ties are reported, not hidden); greedy ids with the KV
   cache vs HF; cached vs uncached step logits.

Prompts (text completion; Qwen3-0.6B-Base is not a chat model):
`"Artificial intelligence is"`, `"The capital of France is"`, `"Hà Nội là thủ đô của"`,
`"def fibonacci(n):\n    "`, `"1, 1, 2, 3, 5, 8, 13,"`.

### Results on the real checkpoint (maintainer's Fedora laptop, 2026-10-16)

`results/m2/parity_float32.json`, `parity_bfloat16.json` (transformers 5.20.0,
torch 2.14.1+cpu, 2 threads):

| check | float32 | bfloat16 |
|---|---|---|
| mapping: 310/310 tensors, missing / unexpected / shape errors | 0 / 0 / 0 | 0 / 0 / 0 |
| parameters ours vs HF | 596,049,920 = 596,049,920 | same |
| tokenizer ids ours vs `AutoTokenizer` (5 prompts) | identical | identical |
| embeddings, all 28 layer outputs, final norm, logits: max \|Δ\| | **0** (bit-identical) | **0** |
| top-1 / top-5 agreement of the logits (every position) | 1.000 / 1.000 | 1.000 / 1.000 |
| smallest reference top-2 margin (near-tie check) | 0.0625 (Vietnamese prompt) | 0.125 |
| greedy ids (16 tokens, our KV cache) vs `transformers.generate` | identical, 5/5 prompts | identical |
| cached vs uncached step logits (8 steps), max \|Δ\| | 2.4e-5 – 3.0e-5 | 0 |
| verdict | **PASSED** | **PASSED** |

Greedy completions of our model (identical to HF):

* "Artificial intelligence is" → " a field that has seen rapid advancements in recent years, driven by the increasing availability"
* "The capital of France is" → " Paris. The capital of Germany is Berlin. The capital of Italy is Rome."
* "Hà Nội là thủ đô của" → " Việt Nam, là một trong những thành phố lớn nhất thế giới. Tuy"
* "def fibonacci(n):\n    " → "\n    if n == 0:\n        return 0\n    elif n"
* "1, 1, 2, 3, 5, 8, 13," → " 21, 34, 55, 89,"

These completions show the weights are correctly loaded; they are not a quality claim
about the base model. Weights-tier tests (`pytest -m "weights and not slow"`): 11 passed;
the `slow` fp32 test was NOT RUN on the real weights (the parity script covers it).

The sandbox rehearsal with random weights of the exact 0.6B shapes gave the same
result (bit-identical), which validated the scripts before the download.

## 6. Reproduce

```bash
conda activate qwen3-scratch
pip install -e ".[dev,hf,notebooks]"
bash scripts/m2_local_run.sh            # hardware, download, tests, parity, run, benchmark
# or step by step:
python scripts/verify_pretrained.py --download --dtype float32 --threads 2
python scripts/run_pretrained_qwen3.py --download --prompt "Artificial intelligence is" --max-new-tokens 32
QWEN3_WEIGHTS_DIR=<snapshot dir> python -m pytest -m weights
```

Dry run without downloading: `python scripts/make_synthetic_checkpoint.py --size tiny --out /tmp/q3`
then `python scripts/verify_pretrained.py --model-dir /tmp/q3` (random weights, toy tokenizer).

## 7. Limitations

No padded batches with the KV cache; no sliding-window or scaled RoPE; CPU-only
validation (GPU is M4); float16 loading is possible but untested against a reference.
