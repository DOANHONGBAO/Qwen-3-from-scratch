# Kaggle (optional remote GPU)

Kaggle is used only for runs with an explicit experimental objective: GPU work in
M4, and in M2 as a **fallback** when the local CPU machine cannot run the
pretrained-model experiments safely (`kaggle/qwen3_inference/`). M2 itself ran
locally (see `docs/hardware_assessment.md`), so no M2 kernel was submitted. Kaggle authentication does **not** guarantee GPU availability (quota,
phone verification and queueing all apply).

## 1. Credentials (outside the repository)

Pick one mechanism from the official CLI docs; none of them touches this repo:

- `kaggle auth login` (OAuth in the browser), or
- environment variable `KAGGLE_API_TOKEN` (e.g. exported from your shell profile), or
- token file `~/.kaggle/access_token`, or
- legacy `~/.kaggle/kaggle.json` (from "Create Legacy API Key").

Keep token files private: `chmod 600 ~/.kaggle/access_token ~/.kaggle/kaggle.json`.
`.gitignore` blocks `kaggle.json`, `access_token` and `.kaggle/` as a second line
of defence. Never paste tokens into notebooks, scripts, chat or commits. Inside a
Kaggle notebook use Kaggle Secrets if a secret is ever needed (none is needed now).

## 2. Check authentication without exposing secrets

```bash
pip install -e ".[kaggle]"
bash scripts/kaggle_check_auth.sh
```

The script reports which mechanism is present and file permissions, then makes a
read-only API call with output suppressed. It never prints credential contents.

## 3. GPU smoke test (manual, deliberate)

```bash
python kaggle/smoke_test/kaggle_smoke_test.py                    # local CPU dry run
KAGGLE_USERNAME=<you> bash kaggle/prepare_kernel.sh smoke_test   # renders build/kaggle/smoke_test
kaggle kernels push -p build/kaggle/smoke_test --accelerator NvidiaTeslaT4
kaggle kernels status <you>/qwen3-scratch-gpu-smoke-test
kaggle kernels output <you>/qwen3-scratch-gpu-smoke-test -p build/kaggle/smoke_test/output
```

The kernel is private, has internet disabled, uses no datasets or weights, and
writes `smoke_test_report.json` (device, versions, fp32/bf16 matmul errors vs. an
fp64 CPU reference).

Verify the metadata fields once with `kaggle kernels init -p /tmp/k` and compare
with `kaggle/smoke_test/kernel-metadata.template.json` (the CLI docs do not list
them; see docs/references.md §4).

## 4. M2 fallback kernel (`kaggle/qwen3_inference/`)

Private CPU kernel with internet enabled. It clones this repository at the commit
that `prepare_kernel.sh` writes into the script (your local `HEAD`, which must be
pushed), installs pinned `transformers`/`safetensors`/`tokenizers`/`huggingface-hub`
versions, downloads the pinned checkpoint revision, runs `hardware_report.py`,
`verify_pretrained.py` (fp32) and `benchmark_generation.py`, and leaves only small
JSON/CSV/PNG files in its output. Inference runs on this repository's model; HF is the
reference only. Details: [`kaggle/qwen3_inference/README.md`](../kaggle/qwen3_inference/README.md).

## 5. Test tier separation

| Tier | Where | How it is selected |
|---|---|---|
| Local CPU tests | laptop, GitHub Actions | `pytest` (default; `gpu`/`weights` auto-skip) |
| Kaggle GPU tests | Kaggle kernel | `pytest -m gpu` inside a GPU kernel (M4) |
| Pretrained-weight tests | laptop or Kaggle | `QWEN3_WEIGHTS_DIR=/path pytest -m weights` (M2) |
