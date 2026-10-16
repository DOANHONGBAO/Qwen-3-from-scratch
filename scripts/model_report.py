#!/usr/bin/env python
"""End-to-end numerical report: Tiny Qwen3 vs Hugging Face Qwen3ForCausalLM (M1.6).

    python scripts/model_report.py

Builds HF Qwen3ForCausalLM from the Tiny config (configs/qwen3_tiny.json; no
download), copies ALL its parameters into our model with load_state_dict(strict=True),
and compares embeddings, every layer output, the final norm, logits, loss and
gradients on CPU float32 (eager attention, eval mode). Also compares our float32
model with the same weights in float64. Requires `transformers` for the HF part.
Metrics: max_abs, mean_abs, rel_l2 = ||a-b|| / ||b||.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import QWEN3_TINY as CFG
from qwen3_scratch.model import Qwen3ForCausalLM


def metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    a, b = a.detach().double(), b.detach().double()
    d = (a - b).abs()
    return d.max().item(), d.mean().item(), (d.norm() / b.norm()).item()


def main() -> None:
    try:
        import transformers
        from transformers.models.qwen3 import modeling_qwen3 as hf_mod
    except ImportError:
        print("transformers not installed: HF comparison NOT RUN (pip install -e '.[hf]')")
        return

    torch.manual_seed(0)
    hf_cfg = transformers.Qwen3Config(
        vocab_size=CFG.vocab_size,
        hidden_size=CFG.hidden_size,
        intermediate_size=CFG.intermediate_size,
        num_hidden_layers=CFG.num_hidden_layers,
        num_attention_heads=CFG.num_attention_heads,
        num_key_value_heads=CFG.num_key_value_heads,
        head_dim=CFG.head_dim,
        max_position_embeddings=CFG.max_position_embeddings,
        rope_theta=CFG.rope_theta,
        rms_norm_eps=CFG.rms_norm_eps,
        tie_word_embeddings=CFG.tie_word_embeddings,
        attn_implementation="eager",
    )
    hf = hf_mod.Qwen3ForCausalLM(hf_cfg).eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn(p.shape))
    ours = Qwen3ForCausalLM(CFG).eval()
    ours.load_state_dict(hf.state_dict(), strict=True)
    ours64 = Qwen3ForCausalLM(CFG, dtype=torch.float64).eval()
    ours64.load_state_dict({k: v.double() for k, v in hf.state_dict().items()}, strict=True)

    ids = torch.randint(0, CFG.vocab_size, (4, 64), generator=torch.Generator().manual_seed(1))
    seen = {"layers": []}
    hooks = [
        hf.model.embed_tokens.register_forward_hook(lambda m, i, o: seen.__setitem__("emb", o))
    ]
    for layer in hf.model.layers:
        hooks.append(layer.register_forward_hook(lambda m, i, o: seen["layers"].append(o)))
    hooks.append(hf.model.norm.register_forward_hook(lambda m, i, o: seen.__setitem__("norm", o)))
    ref = hf(input_ids=ids, labels=ids)
    for h in hooks:
        h.remove()
    ref.loss.backward()
    out = ours(ids, labels=ids, output_hidden_states=True)
    out.loss.backward()
    out64 = ours64(ids, labels=ids, output_hidden_states=True)

    print(f"torch {torch.__version__}; transformers {transformers.__version__}")
    print(
        f"Tiny Qwen3: {ours.num_parameters():,} parameters (HF: "
        f"{sum(p.numel() for p in hf.parameters()):,}); input ids (4, 64)\n"
    )
    print("| ours (float32) vs | tensor | max_abs | mean_abs | rel_l2 |")
    print("|---|---|---|---|---|")
    rows = [("HF", "embeddings", out.hidden_states[0], seen["emb"])]
    for i, h in enumerate(seen["layers"]):
        h = h[0] if isinstance(h, tuple) else h
        rows.append(("HF", f"layer {i} output", out.hidden_states[i + 1], h))
    rows += [
        ("HF", "final norm", out.last_hidden_state, seen["norm"]),
        ("HF", "logits", out.logits, ref.logits),
        ("HF", "loss", out.loss, ref.loss),
    ]
    hf_params = dict(hf.named_parameters())
    for name in (
        "model.embed_tokens.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.q_norm.weight",
        "model.layers.1.mlp.down_proj.weight",
        "model.norm.weight",
    ):
        rows.append(("HF", f"grad {name}", ours.get_parameter(name).grad, hf_params[name].grad))
    rows += [
        ("float64 model", "logits", out.logits, out64.logits),
        ("float64 model", "loss", out.loss, out64.loss),
    ]
    for ref_name, tensor, a, b in rows:
        mx, mean, rel = metrics(a, b)
        print(f"| {ref_name} | {tensor} | {mx:.3e} | {mean:.3e} | {rel:.3e} |")
    print(
        f"\nloss ours = {out.loss.item():.7f}, HF = {ref.loss.item():.7f}, "
        f"ln(V) = {torch.log(torch.tensor(float(CFG.vocab_size))).item():.7f}"
    )


if __name__ == "__main__":
    main()
