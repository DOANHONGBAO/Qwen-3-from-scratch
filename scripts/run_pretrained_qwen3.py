#!/usr/bin/env python
"""Text completion with OUR Qwen3 implementation and the official weights (M2.1/M2.2).

    python scripts/run_pretrained_qwen3.py --model-dir <checkpoint dir> \
        --prompt "Artificial intelligence is" --max-new-tokens 32
    python scripts/run_pretrained_qwen3.py --download --prompt "Hà Nội là" --do-sample \
        --temperature 0.7 --top-p 0.9 --seed 0

Qwen3-0.6B-Base is a BASE (pre-trained) model: it continues text. It is not an
instruction-following chat assistant, so prompts should be the beginning of a text.

The model, generation loop, KV cache and sampling are this repository's code;
Hugging Face provides only the files (and the tokenizer engine via ``tokenizers``).
"""

from __future__ import annotations

import argparse
import os
import time

import torch

from qwen3_scratch.generation import GenerationConfig, generate
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
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--no-cache", action="store_true", help="disable the KV cache")
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-eos", action="store_true", help="do not stop at <|endoftext|>")
    parser.add_argument("--stream", action="store_true", help="print tokens as they come")
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
    load_s = time.perf_counter() - t0
    print(
        f"loaded {report.num_parameters:,} parameters ({args.dtype}, {args.device}) "
        f"in {load_s:.1f}s | {report.summary()}"
    )
    print(f"tokenizer: {tok}")

    ids = torch.tensor([tok.encode(args.prompt)], device=args.device)
    print(f"prompt: {args.prompt!r} -> {ids.shape[1]} tokens {ids[0].tolist()}")
    cfg = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_token_id=None if args.no_eos else tok.eos_token_id,
        use_cache=not args.no_cache,
    )
    stamps: list[float] = []
    printed: list[int] = []

    def on_token(step, next_ids):
        stamps.append(time.perf_counter())
        if args.stream:
            printed.append(int(next_ids[0]))
            text = tok.decode(printed, skip_out_of_vocab=True)
            prev = tok.decode(printed[:-1], skip_out_of_vocab=True)
            print(text[len(prev) :], end="", flush=True)

    if args.stream:
        print("\n--- completion (streaming) ---\n" + args.prompt, end="", flush=True)
    start = time.perf_counter()
    out = generate(model, ids, cfg, seed=args.seed if args.do_sample else None, on_token=on_token)
    total = time.perf_counter() - start
    new = out.new_tokens[0].tolist()
    if args.stream:
        print()
    else:
        print("\n--- completion ---")
        print(args.prompt + tok.decode(new, skip_out_of_vocab=True))
    print("---")
    first = stamps[0] - start if stamps else float("nan")
    decode_n = max(len(stamps) - 1, 0)
    decode_s = (stamps[-1] - stamps[0]) if decode_n else 0.0
    print(
        f"new tokens: {len(new)} | stopped at EOS: {bool(out.finished[0])} | cache: {cfg.use_cache}"
    )
    print(
        f"time to first token (prefill): {first:.3f}s | decode: {decode_n} tokens in "
        f"{decode_s:.2f}s ({decode_n / decode_s if decode_s else float('nan'):.2f} tok/s) | "
        f"total {total:.2f}s"
    )
    if out.cache is not None:
        print(f"KV cache: {out.cache.seq_len} tokens, {out.cache.nbytes() / 2**20:.1f} MiB")
    print(
        f"RSS {(rss_bytes() or 0) / GiB:.2f} GiB | peak RSS {(peak_rss_bytes() or 0) / GiB:.2f} GiB"
    )


if __name__ == "__main__":
    main()
