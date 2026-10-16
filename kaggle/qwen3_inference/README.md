# Kaggle fallback: M2 inference parity and benchmark

Use only when the local machine cannot run `scripts/m2_local_run.sh` safely (for
example: less than ~4.5 GiB free RAM for the float32 parity phases). The kernel runs
**this repository's** model at a pinned commit; Hugging Face is only the reference.

```bash
pip install -e ".[kaggle]"
bash scripts/kaggle_check_auth.sh                         # never prints secrets
git push                                                  # the kernel clones a PUSHED commit
KAGGLE_USERNAME=<you> bash kaggle/prepare_kernel.sh qwen3_inference
kaggle kernels push -p build/kaggle/qwen3_inference       # private, CPU, internet on
kaggle kernels status <you>/qwen3-scratch-m2-inference
kaggle kernels output <you>/qwen3-scratch-m2-inference -p build/kaggle/qwen3_inference/output
```

* Private kernel, no accelerator by default (the parity check is a CPU float32
  experiment). Add `--accelerator NvidiaTeslaT4` only for a GPU experiment.
* Internet is enabled because the kernel downloads the pinned checkpoint revision and
  pip packages. Nothing is uploaded from your machine except the kernel script.
* Outputs: `results/` (JSON, CSV, PNG) and `results.zip`. Reference tensors are deleted
  before the outputs are collected; weights are never part of the outputs.
* A run is complete only when `kaggle kernels status` reports `COMPLETE` **and**
  `parity_float32.json` is present in the downloaded output.

Local dry run (CPU, needs internet for the clone/download):
`python kaggle/qwen3_inference/run_qwen3_inference.py` after replacing `COMMIT_SHA`.
