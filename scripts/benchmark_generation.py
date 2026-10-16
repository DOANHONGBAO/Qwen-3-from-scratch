#!/usr/bin/env python
"""Benchmark OUR generation loop: KV cache vs. full recomputation (M2.2).

    # real checkpoint (download excluded from timing):
    python scripts/benchmark_generation.py --model-dir <ckpt> --tag fedora_fp32
    # tiny random model, seconds:
    python scripts/benchmark_generation.py --tiny --tag tiny

For every (prompt length P, new tokens N, mode) case:

* prefill  = first forward over the P prompt tokens (identical work in both modes);
* decode   = each later step. Cached: one token attends to the cache.
             Uncached: the WHOLE sequence (P + t tokens) is recomputed;
* decode tok/s = (N - 1) / decode time  (prefill excluded; never mixed with end-to-end);
* peak RSS of the case (VmHWM reset before each case), KV cache bytes at the end;
* cached vs. uncached: max |logit difference| per step and token agreement.

Greedy decoding, EOS ignored (fixed length), synthetic prompt ids. Uncached cases
are skipped beyond ``--uncached-max-prompt`` or after one exceeds the time budget
(they are reported as SKIPPED, never extrapolated).
Results: ``results/m2/benchmark_<tag>.json`` + ``.csv`` (+ PNG plots with ``--plot``).
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import torch

from qwen3_scratch.config import QWEN3_TINY
from qwen3_scratch.kv_cache import kv_cache_bytes
from qwen3_scratch.model import Qwen3ForCausalLM
from qwen3_scratch.sysinfo import GiB, peak_rss_bytes, reset_peak_rss, rss_bytes, system_summary

REPO = Path(__file__).resolve().parents[1]


def run_case(model, prompt: torch.Tensor, n_new: int, use_cache: bool) -> dict:
    """One greedy run with explicit timing of prefill and every decode step."""
    reset_peak_rss()
    step_times, logits_steps = [], []
    with torch.inference_mode():
        start = time.perf_counter()
        cache = model.new_cache() if use_cache else None
        seq = prompt
        out = model(prompt, past_key_values=cache, use_cache=use_cache, logits_to_keep=1)
        logits = out.logits[:, -1].float()
        next_id = logits.argmax(-1, keepdim=True)
        prefill = time.perf_counter() - start
        logits_steps.append(logits[0])
        for _ in range(n_new - 1):
            seq = torch.cat([seq, next_id], dim=1)
            t = time.perf_counter()
            if use_cache:
                out = model(next_id, past_key_values=cache, use_cache=True, logits_to_keep=1)
            else:
                out = model(seq, logits_to_keep=1)
            logits = out.logits[:, -1].float()
            next_id = logits.argmax(-1, keepdim=True)
            step_times.append(time.perf_counter() - t)
            logits_steps.append(logits[0])
        total = time.perf_counter() - start
    seq = torch.cat([seq, next_id], dim=1)
    decode = sum(step_times)
    return {
        "prefill_s": prefill,
        "decode_s": decode,
        "decode_step_median_s": statistics.median(step_times) if step_times else None,
        "decode_step_last_s": step_times[-1] if step_times else None,
        "decode_tok_per_s": (len(step_times) / decode) if decode > 0 else None,
        "total_s": total,
        "peak_rss_gib": round((peak_rss_bytes() or 0) / GiB, 3),
        "kv_cache_mib": round(cache.nbytes() / 2**20, 3) if cache is not None else 0.0,
        "tokens": seq[0, prompt.shape[1] :].tolist(),
        "_logits": torch.stack(logits_steps),
    }


def summarise(runs: list[dict]) -> dict:
    keys = ["prefill_s", "decode_s", "decode_step_median_s", "decode_tok_per_s", "total_s"]
    out = {
        k: statistics.median([r[k] for r in runs if r[k] is not None] or [float("nan")])
        for k in keys
    }
    out["peak_rss_gib"] = max(r["peak_rss_gib"] for r in runs)
    out["kv_cache_mib"] = runs[-1]["kv_cache_mib"]
    out["repeats"] = len(runs)
    out["tokens"] = runs[-1]["tokens"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--model-dir")
    src.add_argument("--tiny", action="store_true", help="random Tiny Qwen3 (no checkpoint)")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--prompt-lens", type=int, nargs="+", default=[16, 64, 128, 256])
    parser.add_argument("--new-tokens", type=int, nargs="+", default=[16, 32, 64])
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--uncached-max-prompt", type=int, default=128)
    parser.add_argument("--max-case-seconds", type=float, default=180.0)
    parser.add_argument("--tag", default="local")
    parser.add_argument("--out-dir", default=str(REPO / "results" / "m2"))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    if args.tiny:
        cfg = dataclasses.replace(QWEN3_TINY, max_position_embeddings=1024)
        model = Qwen3ForCausalLM(cfg, dtype=dtype).eval()
        source, revision = "random Tiny Qwen3 (QWEN3_TINY)", None
    else:
        from qwen3_scratch.pretrained import load_pretrained

        model, _ = load_pretrained(args.model_dir, dtype=dtype)
        cfg = model.config
        resolved = Path(args.model_dir).resolve()
        # no absolute paths in reports (they contain the user's home directory)
        source = f"{resolved.parent.parent.name}/{resolved.name}"  # models--Qwen--...-Base/<sha>
        revision = resolved.name  # in the HF cache the snapshot directory is the commit SHA
    load_s = time.perf_counter() - t0
    weights_rss = (rss_bytes() or 0) / GiB
    vocab_hi = min(cfg.vocab_size, 151_643)  # real-token range of the official tokenizer
    g = torch.Generator().manual_seed(0)
    max_p = max(args.prompt_lens)
    base_prompt = torch.randint(0, vocab_hi, (1, max_p), generator=g)

    run_case(model, base_prompt[:, :8], 4, True)  # warm-up (allocator, kernels)
    run_case(model, base_prompt[:, :8], 4, False)

    results, uncached_budget_hit = [], False
    for p in args.prompt_lens:
        for n in args.new_tokens:
            if p + n > cfg.max_position_embeddings:
                continue
            prompt = base_prompt[:, :p]
            case_logits = {}
            for use_cache in (True, False):
                mode = "cached" if use_cache else "uncached"
                row = {"prompt_len": p, "new_tokens": n, "mode": mode}
                if not use_cache and (p > args.uncached_max_prompt or uncached_budget_hit):
                    row["status"] = "SKIPPED (time budget)"
                    results.append(row)
                    print(f"P={p:4d} N={n:3d} {mode:8s} SKIPPED", flush=True)
                    continue
                runs = []
                for _ in range(args.repeats):
                    r = run_case(model, prompt, n, use_cache)
                    runs.append(r)
                    if r["total_s"] > args.max_case_seconds:
                        if not use_cache:
                            uncached_budget_hit = True
                        break
                case_logits[mode] = runs[-1]["_logits"]
                row.update(summarise(runs))
                row["status"] = "RUN"
                results.append(row)
                print(
                    f"P={p:4d} N={n:3d} {mode:8s} prefill {row['prefill_s'] * 1e3:8.1f} ms | "
                    f"decode {row['decode_tok_per_s'] or 0:7.2f} tok/s "
                    f"(median step {(row['decode_step_median_s'] or 0) * 1e3:7.1f} ms) | "
                    f"total {row['total_s']:6.2f} s | peak RSS {row['peak_rss_gib']:.2f} GiB | "
                    f"KV {row['kv_cache_mib']:.1f} MiB",
                    flush=True,
                )
            if len(case_logits) == 2:
                a, b = case_logits["cached"], case_logits["uncached"]
                cached_row = next(
                    r
                    for r in results
                    if r["prompt_len"] == p and r["new_tokens"] == n and r["mode"] == "cached"
                )
                uncached_row = results[-1]
                diff = (a - b).abs().amax(dim=-1)
                cached_row["max_logit_diff_vs_uncached"] = float(diff.max())
                cached_row["tokens_equal_uncached"] = cached_row["tokens"] == uncached_row["tokens"]
                speed = (
                    uncached_row["decode_s"] / cached_row["decode_s"]
                    if cached_row["decode_s"]
                    else None
                )
                cached_row["decode_speedup_vs_uncached"] = speed
                print(
                    f"           cached vs uncached: max |dlogit| {float(diff.max()):.3e}, "
                    f"tokens equal {cached_row['tokens_equal_uncached']}, decode speed-up "
                    f"{speed:.2f}x"
                    if speed
                    else "",
                    flush=True,
                )

    for r in results:
        r.pop("_logits", None)
    report = {
        "source": source,
        "revision_dir": revision,
        "dtype": args.dtype,
        "batch_size": 1,
        "threads": args.threads,
        "load_seconds": load_s,
        "rss_after_load_gib": round(weights_rss, 3),
        "config": cfg.to_dict(),
        "kv_cache_bytes_per_token": kv_cache_bytes(cfg, 1, bytes_per_element=dtype.itemsize),
        "system": system_summary(),
        # paths are reduced to their last component: no home directory in reports
        "settings": {
            k: (Path(v).name if k in ("model_dir", "out_dir") and v else v)
            for k, v in vars(args).items()
            if k != "plot"
        },
        "results": results,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"benchmark_{args.tag}"
    stem.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    fields = [
        "prompt_len",
        "new_tokens",
        "mode",
        "status",
        "prefill_s",
        "decode_step_median_s",
        "decode_tok_per_s",
        "total_s",
        "peak_rss_gib",
        "kv_cache_mib",
        "max_logit_diff_vs_uncached",
        "tokens_equal_uncached",
        "decode_speedup_vs_uncached",
    ]
    with stem.with_suffix(".csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(results)
    print(f"\nload {load_s:.1f}s | wrote {stem}.json and .csv")
    if args.plot:
        plot(report, stem)


def plot(report: dict, stem: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in report["results"] if r.get("status") == "RUN"]
    n_max = max(r["new_tokens"] for r in rows)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    title = f"{Path(report['source']).name} | {report['dtype']} | {report['threads']} threads"
    fig.suptitle(f"Qwen3 generation benchmark (this repo's implementation)\n{title}", fontsize=10)
    colors = {"cached": "#2a6fdb", "uncached": "#d9822b"}

    ax = axes[0, 0]
    for mode in ("cached", "uncached"):
        sel = sorted(
            (r for r in rows if r["mode"] == mode and r["new_tokens"] == n_max),
            key=lambda r: r["prompt_len"],
        )
        if sel:
            ax.plot(
                [r["prompt_len"] for r in sel],
                [r["decode_step_median_s"] * 1e3 for r in sel],
                "o-",
                color=colors[mode],
                label=mode,
            )
    ax.set(
        xlabel="prompt length (tokens)",
        ylabel="median decode step (ms)",
        title=f"Decode latency per token (N={n_max})",
    )
    ax.legend()

    ax = axes[0, 1]
    for mode in ("cached", "uncached"):
        sel = sorted(
            (r for r in rows if r["mode"] == mode and r["new_tokens"] == n_max),
            key=lambda r: r["prompt_len"],
        )
        if sel:
            ax.plot(
                [r["prompt_len"] for r in sel],
                [r["decode_tok_per_s"] for r in sel],
                "o-",
                color=colors[mode],
                label=mode,
            )
    ax.set(
        xlabel="prompt length (tokens)",
        ylabel="decode tokens / s",
        title="Decode throughput (prefill excluded)",
    )
    ax.legend()

    ax = axes[1, 0]
    for mode in ("cached", "uncached"):
        sel = sorted(
            (r for r in rows if r["mode"] == mode and r["new_tokens"] == n_max),
            key=lambda r: r["prompt_len"],
        )
        if sel:
            ax.plot(
                [r["prompt_len"] for r in sel],
                [r["peak_rss_gib"] for r in sel],
                "o-",
                color=colors[mode],
                label=mode,
            )
    ax.set(xlabel="prompt length (tokens)", ylabel="peak RSS (GiB)", title="Peak process memory")
    ax.legend()

    ax = axes[1, 1]
    per_tok = report["kv_cache_bytes_per_token"]
    lengths = [2**k for k in range(4, 16) if 2**k <= report["config"]["max_position_embeddings"]]
    ax.plot(lengths, [per_tok * t / 2**20 for t in lengths], "-", color="#555", label="analytic")
    meas = [r for r in rows if r["mode"] == "cached"]
    ax.plot(
        [r["prompt_len"] + r["new_tokens"] - 1 for r in meas],
        [r["kv_cache_mib"] for r in meas],
        "o",
        color=colors["cached"],
        label="measured",
    )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="cached tokens",
        ylabel="KV cache (MiB)",
        title="KV cache size vs context length",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=110)
    print(f"wrote {stem}.png")


if __name__ == "__main__":
    main()
