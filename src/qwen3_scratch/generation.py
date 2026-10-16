"""Minimal greedy decoding (educational; no KV cache, no sampling).

Every step re-runs the full prefix through the model - O(T^2) work per step, fine
for tiny models and short sequences. A KV cache (M2) removes the recomputation.

With randomly initialised weights the generated ids are meaningless: the model
has never seen data, so its "preferences" are an accident of initialisation.
"""

from __future__ import annotations

import torch

from qwen3_scratch.model import Qwen3ForCausalLM

__all__ = ["greedy_generate"]


def greedy_generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Append up to ``max_new_tokens`` argmax tokens to each sequence.

    Args:
        model: a :class:`Qwen3ForCausalLM`; temporarily put in ``eval()`` mode.
        input_ids: ``[T]`` or ``[B, T]`` prompt ids (no padding: all prompts in a
            batch must have the same length; the last position is the last real token).
        max_new_tokens: number of tokens to generate (>= 0).
        eos_token_id: optional stop token. A sequence that produced it keeps
            receiving ``eos_token_id``; generation stops early once all have.

    Returns:
        ``[T + n]`` or ``[B, T + n]`` ids (``n <= max_new_tokens``), same rank as the input.

    Raises:
        ValueError: if ``T + max_new_tokens`` exceeds ``max_position_embeddings``
            (no silent truncation or out-of-range positions).
    """
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError(f"max_new_tokens must be a non-negative int, got {max_new_tokens!r}")
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() not in (1, 2):
        raise ValueError("input_ids must be a tensor of shape [T] or [B, T]")
    squeeze = input_ids.dim() == 1
    ids = input_ids[None] if squeeze else input_ids
    if ids.shape[1] == 0:
        raise ValueError("the prompt must contain at least one token")
    limit = model.config.max_position_embeddings
    if ids.shape[1] + max_new_tokens > limit:
        raise ValueError(
            f"prompt length {ids.shape[1]} + max_new_tokens {max_new_tokens} exceeds "
            f"max_position_embeddings={limit}"
        )
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError(f"eos_token_id must be in [0, {model.config.vocab_size})")

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            finished = torch.zeros(ids.shape[0], dtype=torch.bool, device=ids.device)
            for _ in range(max_new_tokens):
                logits = model(ids).logits[:, -1, :]  # [B, V] at the last position
                next_ids = logits.argmax(dim=-1)  # greedy; ties -> lowest id
                if eos_token_id is not None:
                    next_ids = torch.where(finished, eos_token_id, next_ids)
                    finished |= next_ids == eos_token_id
                ids = torch.cat([ids, next_ids[:, None]], dim=1)
                if eos_token_id is not None and bool(finished.all()):
                    break
    finally:
        model.train(was_training)
    return ids[0] if squeeze else ids
