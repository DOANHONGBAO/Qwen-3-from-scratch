#!/usr/bin/env python
"""CPU demo of the randomly initialised Tiny Qwen3 (M1.6).

    python scripts/demo_tiny_qwen3.py
    python scripts/demo_tiny_qwen3.py --config configs/qwen3_tiny.json --seed 0 \
        --prompt "Hello" --new-tokens 12

The model has RANDOM weights: it has never been trained, so its predictions and
generated text are meaningless. The demo shows shapes, the loss of an untrained
model (~ ln V) and the mechanics of greedy decoding - not language ability.

Toy tokenizer: vocab_size is 256, so we use UTF-8 BYTES as token ids (one byte =
one token). This is a deliberately simple stand-in; the real Qwen3 BPE tokenizer
(151,936 tokens) arrives in M2.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.generation import greedy_generate
from qwen3_scratch.model import Qwen3ForCausalLM

REPO = Path(__file__).resolve().parents[1]


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8", errors="replace")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny Qwen3 CPU demo (random weights)")
    parser.add_argument("--config", default=str(REPO / "configs" / "qwen3_tiny.json"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="Hello, Qwen3!")
    parser.add_argument("--new-tokens", type=int, default=12)
    args = parser.parse_args()

    cfg = Qwen3Config.from_json(args.config)
    if cfg.vocab_size != 256:
        raise SystemExit("the byte-level toy tokenizer needs vocab_size == 256")
    torch.manual_seed(args.seed)
    model = Qwen3ForCausalLM(cfg).eval()

    print("== Tiny Qwen3 (educational, RANDOM weights, not a pretrained model) ==")
    print(f"config: {Path(args.config).name}")
    for key, value in cfg.to_dict().items():
        print(f"  {key:24s} {value}")
    total, trainable = model.num_parameters(), model.num_parameters(trainable_only=True)
    print(
        f"parameters: total {total:,} | trainable {trainable:,} "
        f"| tied embeddings: {model.weights_tied} "
        f"(untied would add {cfg.vocab_size * cfg.hidden_size:,})"
    )

    ids = torch.tensor([encode(args.prompt)])
    print(f"\nprompt {args.prompt!r} -> {ids.shape[1]} byte tokens: {ids[0].tolist()}")
    with torch.no_grad():
        out = model(ids, labels=ids, output_hidden_states=True)
    print("\n== shapes ==")
    print(f"  input_ids            {tuple(ids.shape)}")
    print(f"  embeddings           {tuple(out.hidden_states[0].shape)}")
    for i, h in enumerate(out.hidden_states[1:]):
        print(f"  after layer {i}        {tuple(h.shape)}")
    print(f"  final RMSNorm        {tuple(out.last_hidden_state.shape)}")
    print(f"  logits               {tuple(out.logits.shape)}")

    probs = torch.softmax(out.logits[0, -1], dim=-1)
    top = torch.topk(probs, 5)
    print("\n== next-token prediction after the prompt (meaningless: random weights) ==")
    for p, i in zip(top.values.tolist(), top.indices.tolist(), strict=True):
        print(f"  id {i:3d} {decode([i])!r:8s} p = {p:.4f}")
    print(f"  (uniform would be 1/{cfg.vocab_size} = {1 / cfg.vocab_size:.4f})")

    print("\n== causal LM loss with labels = input_ids ==")
    print(
        f"  loss = {out.loss.item():.4f}   ln(V) = {math.log(cfg.vocab_size):.4f} "
        "(an untrained model is close to uniform)"
    )

    n = min(args.new_tokens, cfg.max_position_embeddings - ids.shape[1])
    gen = greedy_generate(model, ids, n)
    new = gen[0, ids.shape[1] :].tolist()
    print(f"\n== greedy generation, {n} new tokens (no KV cache) ==")
    print(f"  new ids: {new}")
    print(f"  as bytes: {decode(new)!r}")
    print("  -> gibberish is expected: the weights were never trained.")
    print("     (Tied embeddings often make an untrained model repeat the last token: the")
    print("      residual stream still resembles that token's embedding, and the LM head")
    print("      scores tokens by similarity to the embedding matrix.)")
    again = greedy_generate(model, ids, n)
    print(f"  deterministic (same result twice): {torch.equal(gen, again)}")


if __name__ == "__main__":
    main()
