#!/usr/bin/env python
"""Text completion with OUR Qwen3 implementation and the official weights (M2.1).

    python scripts/run_pretrained_qwen3.py --model-dir <checkpoint dir> \
        --prompt "Artificial intelligence is" --max-new-tokens 32
    python scripts/run_pretrained_qwen3.py --download --prompt "Hà Nội là"

Qwen3-0.6B-Base is a BASE (pre-trained) model: it continues text. It is not an
instruction-following chat assistant, so prompts should be the beginning of a text.

The model and the (greedy, full-recomputation) decoding loop are this repository's
code; Hugging Face provides only the files (and the tokenizer engine via
``tokenizers``). The KV cache and sampling arrive in M2.2.
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from qwen3_scratch.generation import greedy_generate
from qwen3_scratch.pretrained import (
    QWEN3_0_6B_BASE_REPO,
    QWEN3_0_6B_BASE_REVISION,
    download_checkpoint,
    load_pretrained,
)
from qwen3_scratch.sysinfo import GiB, peak_rss_bytes, rss_bytes
from qwen3_scratch.tokenizer import Qwen3Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", help="local checkpoint directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help=f"use {QWEN3_0_6B_BASE_REPO}@{QWEN3_0_6B_BASE_REVISION[:7]}",
    )
    parser.add_argument("--prompt", default="Artificial intelligence is")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--no-eos", action="store_true", help="do not stop at <|endoftext|>")
    args = parser.parse_args()

    if args.download:
        args.model_dir = str(download_checkpoint())
    if not args.model_dir:
        parser.error("pass --model-dir or --download")
    torch.set_num_threads(args.threads)

    t0 = time.perf_counter()
    model, report = load_pretrained(
        args.model_dir, dtype=getattr(torch, args.dtype), device=args.device
    )
    tok = Qwen3Tokenizer.from_pretrained(args.model_dir)
    print(
        f"loaded {report.num_parameters:,} parameters ({args.dtype}, {args.device}) "
        f"in {time.perf_counter() - t0:.1f}s | {report.summary()}"
    )
    print(f"tokenizer: {tok}")
    ids = torch.tensor([tok.encode(args.prompt)], device=args.device)
    print(f"prompt: {args.prompt!r} -> {ids.shape[1]} tokens {ids[0].tolist()}")

    start = time.perf_counter()
    out = greedy_generate(
        model, ids, args.max_new_tokens, eos_token_id=None if args.no_eos else tok.eos_token_id
    )
    total = time.perf_counter() - start
    new = out[0, ids.shape[1] :].tolist()
    print("\n--- completion (greedy) ---")
    print(args.prompt + tok.decode(new, skip_out_of_vocab=True))
    print("---")
    print(f"new tokens: {len(new)} in {total:.2f}s (no KV cache: every step recomputes the prefix)")
    print(
        f"RSS {(rss_bytes() or 0) / GiB:.2f} GiB | peak RSS {(peak_rss_bytes() or 0) / GiB:.2f} GiB"
    )


if __name__ == "__main__":
    main()
