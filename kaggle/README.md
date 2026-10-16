# Kaggle templates

Templates for optional remote GPU runs (milestone M4). Nothing here runs
automatically, and nothing here contains or reads credentials.

| Folder | Purpose | Needs weights | Needs internet |
|---|---|---|---|
| `smoke_test/` | Check that CUDA works and that fp32/bf16 matmuls agree with an fp64 CPU reference | No | No |

Workflow (details in [`docs/kaggle.md`](../docs/kaggle.md)):

```bash
pip install -e ".[kaggle]"                       # optional extra
bash scripts/kaggle_check_auth.sh                # reports auth status, never prints secrets
KAGGLE_USERNAME=<you> bash kaggle/prepare_kernel.sh smoke_test   # renders build/kaggle/smoke_test
# Manual, deliberate step - starts a GPU session that uses your Kaggle quota:
kaggle kernels push -p build/kaggle/smoke_test --accelerator NvidiaTeslaT4
```

The smoke test script also runs on CPU, so you can check it locally first:
`python kaggle/smoke_test/kaggle_smoke_test.py`.
