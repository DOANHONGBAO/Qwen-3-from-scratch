#!/usr/bin/env python
"""Numerical parity of OUR Qwen3 with Hugging Face on the official checkpoint (M2.1).

    python scripts/verify_pretrained.py --model-dir <dir with config.json + model.safetensors>
    python scripts/verify_pretrained.py --download          # pinned Qwen/Qwen3-0.6B-Base

The two models are never in memory together. The script runs two child processes:

1. ``--phase reference``: Hugging Face ``Qwen3ForCausalLM`` (eager attention, eval)
   tokenises the prompts with ``AutoTokenizer``, records embeddings, every decoder
   layer output, the final RMSNorm, the logits (forward hooks) and greedy token ids
   from ``transformers.generate``; saves them to ``<out>/reference_<dtype>.safetensors``.
2. ``--phase ours``: OUR model (``load_pretrained``) and OUR tokenizer adapter do the
   same and compare against the saved reference; then our greedy generation is
   compared with HF's token ids.

Output: ``<out>/parity_<dtype>.json`` (all metrics) and a printed summary.
Exit code 0 only if every check passes. Tolerances (rel = ||a-b|| / ||b||):

* float32: rel <= 1e-5 for every hidden state and the logits. Both sides run the
  same operations on the same CPU; anything above ~1e-6 indicates a real
  difference (mapping, RoPE, masking...), not rounding.
* bfloat16: rel <= 2e-2 (bf16 has 8 mantissa bits; ~4e-3 per op, 28 layers).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO / "results" / "m2"
PROMPTS = [
    "Artificial intelligence is",
    "The capital of France is",
    "Hà Nội là thủ đô của",
    "def fibonacci(n):\n    ",
    "1, 1, 2, 3, 5, 8, 13,",
]
TOLERANCE = {"float32": 1e-5, "bfloat16": 2e-2}
GiB = 1024**3


# ---------------------------------------------------------------- helpers
def _capture(model, n_layers):
    """Forward hooks on embed_tokens, every decoder layer and the final norm."""
    store: dict[str, object] = {}
    handles = []

    def hook(name):
        def fn(_module, _inputs, output):
            out = output[0] if isinstance(output, tuple) else output
            store[name] = out.detach().float().clone()

        return fn

    base = model.model
    handles.append(base.embed_tokens.register_forward_hook(hook("embed")))
    for i in range(n_layers):
        handles.append(base.layers[i].register_forward_hook(hook(f"layer{i:02d}")))
    handles.append(base.norm.register_forward_hook(hook("norm")))
    return store, handles


def _metrics(a, b) -> dict:
    import torch

    a, b = a.double(), b.double()
    diff = a - b
    denom = torch.linalg.vector_norm(b).clamp_min(1e-30)
    return {
        "max_abs": float(diff.abs().max()),
        "mean_abs": float(diff.abs().mean()),
        "rms": float(diff.pow(2).mean().sqrt()),
        "rel_l2": float(torch.linalg.vector_norm(diff) / denom),
        "ref_rms": float(b.pow(2).mean().sqrt()),
    }


def _logit_agreement(ours, ref, k=5) -> dict:
    import torch

    top1_o, top1_r = ours.argmax(-1), ref.argmax(-1)
    agree = (top1_o == top1_r).float()
    top2 = torch.topk(ref.double(), 2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    tk_o = torch.topk(ours, k, dim=-1).indices
    tk_r = torch.topk(ref, k, dim=-1).indices
    overlap = [
        len(set(x.tolist()) & set(y.tolist())) / k
        for x, y in zip(tk_o.reshape(-1, k), tk_r.reshape(-1, k), strict=True)
    ]
    disagreements = [
        {"position": int(i), "ref_margin": float(margin.reshape(-1)[i])}
        for i in torch.nonzero(agree.reshape(-1) == 0).flatten().tolist()
    ]
    return {
        "top1_agreement": float(agree.mean()),
        f"top{k}_overlap": sum(overlap) / len(overlap),
        "min_ref_top2_margin": float(margin.min()),
        "disagreements": disagreements,  # near-ties are listed, not hidden
    }


def _rss_gib():
    from qwen3_scratch.sysinfo import rss_breakdown

    b = rss_breakdown()
    gib = lambda v: round((v or 0) / GiB, 3)  # noqa: E731
    return {
        "rss_gib": gib(b["VmRSS"]),
        "peak_rss_gib": gib(b["VmHWM"]),
        "rss_anon_gib": gib(b["RssAnon"]),
        "rss_file_gib": gib(b["RssFile"]),
    }


# ------------------------------------------------------------ phase: HF
def phase_reference(args) -> None:
    import torch
    from safetensors.torch import save_file
    from transformers import AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = Qwen3ForCausalLM.from_pretrained(
        args.model_dir, dtype=dtype, attn_implementation="eager"
    ).eval()
    load_s = time.perf_counter() - t0
    n_layers = model.config.num_hidden_layers
    tensors, meta = {}, {"prompts": [], "load_seconds": load_s}
    store, handles = _capture(model, n_layers)
    with torch.inference_mode():
        for p_idx, text in enumerate(args.prompts):
            ids = tok(text, return_tensors="pt", add_special_tokens=True).input_ids
            store.clear()
            logits = model(ids).logits
            for name, t in store.items():
                tensors[f"p{p_idx}.{name}"] = t[0].contiguous()
            tensors[f"p{p_idx}.logits"] = logits[0].float().contiguous()
            tensors[f"p{p_idx}.ids"] = ids[0].contiguous()
            for h in handles:
                h.remove()
            gen = model.generate(
                ids,
                attention_mask=torch.ones_like(ids),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            store, handles = _capture(model, n_layers)
            tensors[f"p{p_idx}.generated"] = gen[0].contiguous()
            meta["prompts"].append(
                {
                    "text": text,
                    "ids": ids[0].tolist(),
                    "generated_ids": gen[0].tolist(),
                    "generated_text": tok.decode(gen[0, ids.shape[1] :]),
                }
            )
    meta.update(_rss_gib())
    import transformers

    meta["transformers"] = transformers.__version__
    meta["num_parameters"] = sum(p.numel() for p in model.parameters())
    out = Path(args.out)
    save_file(tensors, str(out / f"reference_{args.dtype}.safetensors"))
    (out / f"reference_{args.dtype}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    print(f"[reference] done: load {load_s:.1f}s, peak RSS {meta['peak_rss_gib']} GiB")


# ----------------------------------------------------------- phase: ours
def phase_ours(args) -> None:
    import torch
    from safetensors.torch import load_file

    from qwen3_scratch.generation import greedy_generate
    from qwen3_scratch.pretrained import load_pretrained
    from qwen3_scratch.tokenizer import Qwen3Tokenizer

    torch.set_num_threads(args.threads)
    dtype = getattr(torch, args.dtype)
    out = Path(args.out)
    ref = load_file(str(out / f"reference_{args.dtype}.safetensors"))
    ref_meta = json.loads((out / f"reference_{args.dtype}.json").read_text())
    tol = TOLERANCE[args.dtype]

    t0 = time.perf_counter()
    model, report = load_pretrained(args.model_dir, dtype=dtype)
    load_s = time.perf_counter() - t0
    tok = Qwen3Tokenizer.from_pretrained(args.model_dir)
    cfg = model.config
    result = {
        "dtype": args.dtype,
        "tolerance_rel_l2": tol,
        "load_seconds": load_s,
        "load_report": {
            "ok": report.ok,
            "mapped": len(report.mapped),
            "missing": report.missing,
            "unexpected": report.unexpected,
            "shape_mismatch": {k: list(map(list, v)) for k, v in report.shape_mismatch.items()},
            "dtype_conversions": report.dtype_conversions,
            "tied": report.tied,
            "lm_head_in_checkpoint": report.lm_head_in_checkpoint,
            "num_parameters": report.num_parameters,
            "hf_num_parameters": ref_meta["num_parameters"],
        },
        "prompts": [],
    }
    checks = {
        "mapping_complete": report.ok and report.num_parameters == ref_meta["num_parameters"],
    }
    store, _ = _capture(model, cfg.num_hidden_layers)
    with torch.inference_mode():
        for p_idx, text in enumerate(args.prompts):
            ids = torch.tensor([tok.encode(text)])
            entry = {
                "text": text,
                "tokenizer_ids_match": ids[0].tolist() == ref[f"p{p_idx}.ids"].tolist(),
            }
            checks.setdefault("tokenizer_ids_match", True)
            checks["tokenizer_ids_match"] &= entry["tokenizer_ids_match"]
            ids = ref[f"p{p_idx}.ids"][None]  # always compare the model on identical ids
            store.clear()
            logits = model(ids).logits[0].float()
            tensors = {}
            first_bad = None
            names = ["embed"] + [f"layer{i:02d}" for i in range(cfg.num_hidden_layers)] + ["norm"]
            for name in names:
                m = _metrics(store[name][0], ref[f"p{p_idx}.{name}"])
                tensors[name] = m
                if first_bad is None and m["rel_l2"] > tol:
                    first_bad = name
            tensors["logits"] = _metrics(logits, ref[f"p{p_idx}.logits"])
            if first_bad is None and tensors["logits"]["rel_l2"] > tol:
                first_bad = "logits"
            entry["tensors"] = tensors
            entry["first_divergence"] = first_bad
            entry["logits_agreement"] = _logit_agreement(logits, ref[f"p{p_idx}.logits"])
            entry["next_token"] = {
                "ours": int(logits[-1].argmax()),
                "hf": int(ref[f"p{p_idx}.logits"][-1].argmax()),
                "text": tok.decode([int(logits[-1].argmax())], skip_out_of_vocab=True),
            }
            # greedy generation (full recomputation each step) vs transformers.generate
            hf_gen = ref[f"p{p_idx}.generated"]
            ours_gen = greedy_generate(
                model, ids, args.max_new_tokens, eos_token_id=tok.eos_token_id
            )[0]
            n_cmp = min(len(ours_gen), len(hf_gen))
            entry["generation"] = {
                "ours_ids": ours_gen.tolist(),
                "hf_ids": hf_gen.tolist(),
                "ids_equal_hf": ours_gen.tolist() == hf_gen.tolist(),
                "first_mismatch": next(
                    (i for i in range(n_cmp) if int(ours_gen[i]) != int(hf_gen[i])), None
                ),
                "text_ours": tok.decode(ours_gen[ids.shape[1] :], skip_out_of_vocab=True),
                "text_hf": ref_meta["prompts"][p_idx]["generated_text"],
            }
            result["prompts"].append(entry)
            for key, ok in (
                ("hidden_and_logits_within_tolerance", first_bad is None),
                ("greedy_ids_equal_hf", entry["generation"]["ids_equal_hf"]),
            ):
                checks[key] = checks.get(key, True) and ok
    result["checks"] = checks
    result["passed"] = all(checks.values())
    result["memory"] = _rss_gib()
    result["reference_memory"] = {
        k: ref_meta.get(k) for k in ("rss_gib", "peak_rss_gib", "rss_anon_gib", "rss_file_gib")
    }
    from qwen3_scratch.pretrained import QWEN3_0_6B_BASE_REVISION
    from qwen3_scratch.sysinfo import system_summary

    result["system"] = system_summary()
    result["revision_expected"] = QWEN3_0_6B_BASE_REVISION
    result["threads"] = args.threads
    path = out / f"parity_{args.dtype}.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    print(f"\n== parity ({args.dtype}, tolerance rel_l2 <= {tol:g}) ==")
    print(f"load: {report.summary()}")
    for e in result["prompts"]:
        worst = max(e["tensors"].items(), key=lambda kv: kv[1]["rel_l2"])
        lg = e["tensors"]["logits"]
        print(
            f"- {e['text']!r}: logits max_abs={lg['max_abs']:.3e} rel={lg['rel_l2']:.3e} | "
            f"worst {worst[0]} rel={worst[1]['rel_l2']:.3e} | top1 "
            f"{e['logits_agreement']['top1_agreement']:.3f} | greedy==HF "
            f"{e['generation']['ids_equal_hf']} | first divergence {e['first_divergence']}"
        )
        print(f"    ours: {e['generation']['text_ours']!r}")
    print(f"checks: {checks}\nPASSED: {result['passed']}   (wrote {path})")
    print(
        f"peak RSS ours {result['memory']['peak_rss_gib']} GiB, "
        f"HF {result['reference_memory']['peak_rss_gib']} GiB"
    )
    if not result["passed"]:
        sys.exit(1)


# ---------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-dir", help="local checkpoint directory")
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the pinned Qwen/Qwen3-0.6B-Base revision first",
    )
    parser.add_argument("--dtype", choices=sorted(TOLERANCE), default="float32")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--phase", choices=["all", "reference", "ours"], default="all")
    parser.add_argument("--force", action="store_true", help="skip the free-RAM guard")
    parser.add_argument(
        "--prompt", action="append", dest="prompts", help="override prompts (repeatable)"
    )
    args = parser.parse_args()
    args.prompts = args.prompts or PROMPTS
    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.download:
        from qwen3_scratch.pretrained import download_checkpoint

        args.model_dir = str(download_checkpoint())
        print(f"checkpoint: {args.model_dir}")
    if not args.model_dir:
        parser.error("pass --model-dir or --download")

    if args.phase == "reference":
        return phase_reference(args)
    if args.phase == "ours":
        return phase_ours(args)

    from qwen3_scratch.config import Qwen3Config
    from qwen3_scratch.sysinfo import mem_info

    cfg = Qwen3Config.from_json(Path(args.model_dir) / "config.json")
    bytes_per = 4 if args.dtype == "float32" else 2
    # Anonymous memory per phase: weights + runtime/libraries/activations (~0.75 GiB
    # measured with transformers imported). Memory-mapped checkpoint pages are
    # file-backed and reclaimable, so they are not counted.
    need = cfg.estimate_num_parameters() * bytes_per + int(0.75 * GiB)
    avail = mem_info().get("MemAvailable", 0)
    print(f"estimated peak per phase {need / GiB:.2f} GiB; MemAvailable {avail / GiB:.2f} GiB")
    if need > 0.85 * avail and not args.force:
        sys.exit(
            f"not enough free RAM for a safe run: need ~{need / 0.85 / GiB:.1f} GiB available "
            f"(15% margin), have {avail / GiB:.1f} GiB. Close other programs (browser, IDE) "
            "and retry, or use --dtype bfloat16 (half the memory, looser tolerance). "
            "--force skips this check (the run may then use swap)."
        )
    base = [
        sys.executable,
        __file__,
        "--model-dir",
        args.model_dir,
        "--dtype",
        args.dtype,
        "--threads",
        str(args.threads),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--out",
        args.out,
    ]
    for p in args.prompts:
        base += ["--prompt", p]
    for phase in ("reference", "ours"):  # separate processes: memory is fully released
        print(f"\n--- phase {phase} ---", flush=True)
        code = subprocess.call([*base, "--phase", phase])
        if code:
            sys.exit(code)


if __name__ == "__main__":
    main()
