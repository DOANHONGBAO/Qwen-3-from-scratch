# Hardware assessment and execution plan (M2)

All values below were **measured** on the maintainer's machine by
`scripts/hardware_report.py` and the M2 scripts (raw data: `results/m2/hardware.json`,
`results/m2/*.json`), unless marked *estimate*. Date: 2026-10-16.

## 1. Machine

| item | value |
|---|---|
| OS | Fedora Linux 44, kernel 7.2.8 (x86_64) |
| CPU | AMD Ryzen 3 3250U: 2 cores / 4 threads, 1.4–2.6 GHz |
| SIMD | AVX, AVX2, FMA, F16C; **no** AVX-512, AVX512-BF16, AMX, AVX-VNNI |
| RAM | 5.66 GiB total; MemAvailable 2.1–3.1 GiB with the desktop session and apps open |
| swap | 5.66 GiB (Fedora zram), 0.8–1.0 GiB already in use |
| cgroup memory limit | none |
| disk | 207 GiB free in `/home` |
| HF cache before M2 | empty (8 KiB) |
| GPU | none (`torch.cuda.is_available() == False`) |
| Python / torch | 3.11.17 / 2.14.1+cpu, 2 intra-op threads by default |
| reference libraries | transformers 5.20.0, safetensors 0.8.0, tokenizers 0.23.3, huggingface_hub 2.2.0 |

## 2. Memory model for Qwen3-0.6B-Base

596,049,920 parameters (tied embeddings). Weight size is not peak memory:

| quantity | bf16 | fp32 |
|---|---|---|
| weights | 1.11 GiB | 2.22 GiB |
| largest tensor (embedding 151936 × 1024) | 0.29 GiB | 0.58 GiB |
| logits per 1,024 positions (always fp32 in reports) | 0.58 GiB | 0.58 GiB |
| KV cache per token | 112 KiB | 224 KiB |
| KV cache at 128 / 2,048 / 32,768 tokens | 14 MiB / 224 MiB / 3.5 GiB | 28 MiB / 448 MiB / 7 GiB |
| two models at once (HF + ours) *estimate* | ~3.0 GiB | ~5.5 GiB |

Loading overheads (measured): Python + torch + safetensors/tokenizers ≈ 0.3–0.5 GiB;
with `transformers` imported ≈ 0.75 GiB. The memory-mapped checkpoint (1.11 GiB) is
counted in RSS while it is read, but those pages are file-backed and reclaimable.
The loader copies tensor by tensor with an in-place dtype conversion, so no second
copy of the model and no converted temporary is created.

Observed on this machine with the real weights (`results/m2/parity_*.json`,
`benchmark_local_fp32.json`, run logs):

| process | dtype | peak RSS (VmHWM) | resident after loading |
|---|---|---|---|
| ours, `run_pretrained_qwen3.py` (prompt + 32 tokens) | fp32 | 2.69–3.35 GiB | 2.39 GiB |
| ours, parity phase | fp32 | 3.60 GiB | 2.49 GiB (2.38 anonymous + 0.12 file) |
| HF `transformers`, parity reference phase | fp32 | 3.52 GiB | 2.72 GiB (2.62 anonymous) |
| ours, parity phase | bf16 | 2.53 GiB | 1.57 GiB |
| HF, parity reference phase | bf16 | 1.59 GiB | 1.59 GiB (1.21 file-backed) |

Peak RSS includes the memory-mapped checkpoint pages read during loading; the
anonymous (non-reclaimable) memory of one fp32 model stays ≈ 2.4–2.6 GiB.
Load time of our model: 4.7–19 s (19 s on the first, cold-cache run; ~5 s once the
file is in the page cache). Download: 165 s for 1.19 GB.

## 3. Decision: local CPU, sequential, float32 for correctness

* **One fp32 model fits** (≈ 2.4 GiB resident + runtime) when ≥ 3.5 GiB are available,
  i.e. after closing the browser. **Two fp32 models do not fit** in 5.66 GiB, so the
  parity check runs HF and our model **in separate processes, one after the other**,
  exchanging only small reference tensors (hidden states and logits of short prompts).
* float32 is the correctness reference (bf16 → fp32 is exact; tolerances are tight).
  bf16 would halve memory, but this CPU has no native bf16 arithmetic, so bf16 is not
  assumed to be faster; it is kept as a fallback, not the default.
* Threads: 2 (= physical cores). SMT threads add little to dense matmuls and
  make timings noisier; the setting is recorded in every report.
* Short prompts (≤ 256 tokens) and bounded generation lengths; the uncached
  benchmark is limited to prompts ≤ 128 tokens because each step recomputes the
  whole sequence (≈ 1–3 s per token here).
* `verify_pretrained.py` refuses to start when the estimated per-phase memory exceeds
  85 % of `MemAvailable` (no reliance on swap). With the desktop session running it
  refused three times (2.98–3.07 GiB available). The bf16 check then ran normally
  (estimate 1.86 GiB, 3.31 GiB available); the fp32 check was started with `--force`
  after the maintainer agreed, at 3.55 GiB available (estimate 2.97 GiB, i.e. just above
  the 85 % threshold anyway). Both phases completed; the measured peaks are in §2.
* **Kaggle was not needed**: every M2 experiment ran on this machine. The fallback
  kernel (`kaggle/qwen3_inference/`) is prepared and documented but was **not**
  submitted (no kernel URL, no remote results).

## 4. Rehearsal (sandbox, not this machine)

Before touching the real checkpoint, the full pipeline was rehearsed in the
development sandbox (Intel Xeon, 2 vCPU, AVX-512, 7.8 GiB RAM, no swap) with a
**random-weight checkpoint of the exact 0.6B shapes** (`scripts/make_synthetic_checkpoint.py`):
our model was bit-identical to HF in fp32 (every hidden state and the logits), greedy
ids matched, peak RSS per phase ≈ 4.0 GiB including ≈ 1.1 GiB of mapped file pages,
These numbers only validated the scripts; all results
reported for M2 come from the maintainer's machine.
