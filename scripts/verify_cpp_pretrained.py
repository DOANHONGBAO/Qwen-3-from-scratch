#!/usr/bin/env python
"""Parity of the C++ Qwen3 (``build/cpp/qwen3_run``) with a saved float32 reference (M3).

    bash scripts/build_cpp.sh                                   # builds qwen3_run
    python scripts/verify_cpp_pretrained.py --model-dir <checkpoint dir>
    python scripts/verify_cpp_pretrained.py --download          # pinned Qwen3-0.6B-Base

Reference (``--reference PREFIX``, default ``results/m2/reference_float32``): the
Hugging Face float32 run saved by ``scripts/verify_pretrained.py`` in M2
(``PREFIX.json`` with prompts / ids / greedy ids, ``PREFIX.safetensors`` with the
logits ``p<i>.logits [T, V]``). Nothing is recomputed in Python, so only the C++
model (~2.4 GiB in float32) is in memory. Without that file, ``--make-reference``
first computes the same data with OUR PyTorch model in a child process (useful
for random synthetic checkpoints, see ``scripts/make_synthetic_checkpoint.py``).

Checks per prompt (exit code 0 only if all pass):

* logits of every prompt position: max |C++ - reference| <= ``--atol`` and the
  same argmax at every position. The two sides sum in a different order (naive
  C++ loops vs. MKL/oneDNN), so values are close but not bit-identical.
* greedy token ids (prompt + new tokens, stop at EOS like ``transformers``)
  identical to the reference.

Output: ``results/m3/parity_cpp_float32.json`` (small; safe to commit).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_REFERENCE = REPO / "results" / "m2" / "reference_float32"
DEFAULT_OUT = REPO / "results" / "m3" / "parity_cpp_float32.json"
DEFAULT_BINARY = REPO / "build" / "cpp" / "qwen3_run"
PROMPTS = [
    "Artificial intelligence is",
    "The capital of France is",
    "Hà Nội là thủ đô của",
    "def fibonacci(n):\n    ",
    "1, 1, 2, 3, 5, 8, 13,",
]
GiB = 1024**3


def eos_ids(model_dir: Path) -> list[int]:
    """EOS ids used by ``transformers.generate`` (generation_config.json, else config.json)."""
    for name in ("generation_config.json", "config.json"):
        path = model_dir / name
        if path.is_file():
            eos = json.loads(path.read_text(encoding="utf-8")).get("eos_token_id")
            if eos is not None:
                return [eos] if isinstance(eos, int) else list(eos)
    return []


# ------------------------------------------------- phase: our PyTorch reference
def make_reference(args) -> None:
    """Child process: logits + greedy ids with OUR PyTorch model (float32)."""
    import torch
    from safetensors.torch import save_file

    from qwen3_scratch.generation import generate
    from qwen3_scratch.pretrained import load_pretrained
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    torch.set_num_threads(args.threads)
    model, _ = load_pretrained(args.model_dir, dtype=torch.float32)
    tok = Qwen3Tokenizer.from_pretrained(args.model_dir)
    eos = eos_ids(Path(args.model_dir))
    tensors, meta = {}, {"source": "qwen3_scratch (PyTorch, float32)", "prompts": []}
    with torch.inference_mode():
        for i, text in enumerate(args.prompts):
            ids = torch.tensor([tok.encode(text)])
            tensors[f"p{i}.logits"] = model(ids).logits[0].float().contiguous()
            out = generate(
                model,
                ids,
                max_new_tokens=args.max_new_tokens,
                eos_token_id=tuple(eos) or None,
            )
            meta["prompts"].append(
                {"text": text, "ids": ids[0].tolist(), "generated_ids": out.sequences[0].tolist()}
            )
    prefix = Path(args.reference)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(prefix.with_suffix(".safetensors")))
    prefix.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[reference] wrote {prefix}.json / .safetensors")


# ---------------------------------------------------------------- comparison
def compare(args) -> int:
    import torch
    from safetensors.torch import load_file

    from qwen3_scratch.config import Qwen3Config
    from qwen3_scratch.sysinfo import mem_info

    prefix = Path(args.reference)
    meta = json.loads(prefix.with_suffix(".json").read_text(encoding="utf-8"))
    ref = load_file(str(prefix.with_suffix(".safetensors")))
    prompts = meta["prompts"]
    max_new = max(len(p["generated_ids"]) - len(p["ids"]) for p in prompts)

    cfg = Qwen3Config.from_json(Path(args.model_dir) / "config.json")
    need = cfg.estimate_num_parameters() * 4 + int(0.5 * GiB)
    avail = mem_info().get("MemAvailable", 0)
    print(f"C++ model needs ~{need / GiB:.2f} GiB; MemAvailable {avail / GiB:.2f} GiB")
    if need > 0.85 * avail and not args.force:
        sys.exit("not enough free RAM (15% margin). Close other programs or pass --force.")

    binary = Path(args.binary)
    if not binary.is_file():
        sys.exit(f"{binary} not found: run `bash scripts/build_cpp.sh` first")
    logits_dir = Path(args.out).parent / "cpp_logits_tmp"
    logits_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "--model-dir", str(args.model_dir), "--max-new-tokens", str(max_new)]
    for e in eos_ids(Path(args.model_dir)):
        cmd += ["--eos", str(e)]
    for p in prompts:
        cmd += ["--ids", ",".join(str(i) for i in p["ids"])]
    cmd += ["--logits-out", str(logits_dir / "logits")]
    print("running:", " ".join(cmd[:6]), "...")
    t0 = time.perf_counter()
    run = subprocess.run(cmd, stdout=subprocess.PIPE, text=True, check=False)
    wall = time.perf_counter() - t0
    if run.returncode != 0:
        sys.exit(f"qwen3_run failed with exit code {run.returncode}")
    rows = [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]

    results, all_ok = [], True
    vocab = cfg.vocab_size
    for i, (p, row) in enumerate(zip(prompts, rows, strict=True)):
        path = logits_dir / f"logits_p{i}.bin"
        ours = torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.float32).view(-1, vocab)
        path.unlink()
        theirs = ref[f"p{i}.logits"].float()
        diff = (ours - theirs).abs()
        argmax_same = bool(torch.equal(ours.argmax(-1), theirs.argmax(-1)))
        top5 = set(ours[-1].topk(5).indices.tolist()) == set(theirs[-1].topk(5).indices.tolist())
        sequence = row["ids"] + row["generated"]
        tokens_same = sequence == p["generated_ids"]
        max_abs = float(diff.max())
        ok = argmax_same and tokens_same and max_abs <= args.atol
        all_ok &= ok
        results.append(
            {
                "text": p.get("text"),
                "num_prompt_tokens": len(p["ids"]),
                "max_abs_logit_diff": max_abs,
                "mean_abs_logit_diff": float(diff.mean()),
                "rel_l2_logit_diff": float((ours - theirs).norm() / theirs.norm()),
                "argmax_all_positions_equal": argmax_same,
                "top5_last_position_equal": top5,
                "greedy_ids_equal": tokens_same,
                "cpp_generated_ids": row["generated"],
                "reference_generated_ids": p["generated_ids"][len(p["ids"]) :],
                "cpp_prefill_seconds": row["prefill_seconds"],
                "cpp_decode_tokens_per_second": row["decode_tokens_per_second"],
                "status": "PASSED" if ok else "FAILED",
            }
        )
        print(
            f"[{results[-1]['status']}] prompt {i}: max|dlogit| {max_abs:.3e}, "
            f"argmax equal {argmax_same}, greedy equal {tokens_same}, "
            f"{row['decode_tokens_per_second']:.2f} tok/s"
        )
    logits_dir.rmdir()

    report = {
        "status": "PASSED" if all_ok else "FAILED",
        "reference": meta.get("source", "Hugging Face transformers (scripts/verify_pretrained.py)"),
        "reference_file": str(prefix.relative_to(REPO) if prefix.is_relative_to(REPO) else prefix),
        "model_dir_name": Path(args.model_dir).name,
        "atol_logits": args.atol,
        "cpp_load_seconds": rows[0]["load_seconds"] if rows else None,
        "cpp_peak_rss_gib": rows[-1]["peak_rss_gib"] if rows else None,
        "wall_seconds": round(wall, 1),
        "machine": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "prompts": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{report['status']}: wrote {out}")
    return 0 if all_ok else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", help="checkpoint directory (config.json + safetensors)")
    parser.add_argument("--download", action="store_true", help="use the pinned Qwen3-0.6B-Base")
    parser.add_argument("--binary", default=str(DEFAULT_BINARY))
    parser.add_argument("--reference", default=str(DEFAULT_REFERENCE), help="path prefix")
    parser.add_argument("--make-reference", action="store_true", help="compute it with PyTorch")
    parser.add_argument("--phase", choices=["compare", "reference"], default="compare")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="for --make-reference")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--atol", type=float, default=5e-2, help="max |C++ - ref| on logits")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--force", action="store_true", help="skip the free-RAM guard")
    parser.add_argument("--prompt", action="append", dest="prompts", help="repeatable")
    args = parser.parse_args()
    args.prompts = args.prompts or PROMPTS

    if args.download:
        from qwen3_scratch.pretrained import download_checkpoint

        args.model_dir = str(download_checkpoint())
    if not args.model_dir:
        parser.error("pass --model-dir or --download")

    if args.phase == "reference":
        return make_reference(args)
    if args.make_reference:  # separate process: PyTorch memory is freed before C++ runs
        child = [sys.executable, __file__, "--phase", "reference", "--model-dir", args.model_dir]
        child += ["--reference", args.reference, "--max-new-tokens", str(args.max_new_tokens)]
        child += ["--threads", str(args.threads)]
        for p in args.prompts:
            child += ["--prompt", p]
        subprocess.run(child, check=True)
    elif not Path(args.reference).with_suffix(".json").is_file():
        sys.exit(f"{args.reference}.json not found: run verify_pretrained.py or --make-reference")
    sys.exit(compare(args))


if __name__ == "__main__":
    main()
