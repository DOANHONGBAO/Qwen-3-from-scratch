"""Autoregressive text generation: greedy and sampling, with or without a KV cache.

Two phases when the cache is enabled::

    prefill : model(prompt [B, T0], cache)       -> logits of the last prompt token,
              cache now holds T0 keys/values per layer
    decode  : model(next_token [B, 1], cache)    -> logits for the next token,
              cache grows by 1 per step (positions T0, T0+1, ...)

Without the cache every step re-runs the whole sequence ``[B, T0 + t]`` (the M1.6
behaviour): same result up to floating-point rounding, but O(T) more work per step.

Token selection (``select_next_token``), applied to float32 logits of the last
position, in this order (the same order as Hugging Face's logits processors):

1. ``do_sample=False`` or ``temperature == 0``  -> ``argmax`` (greedy; ties -> lowest id).
2. ``logits / temperature``                       (temperature > 0).
3. top-k: keep the ``k`` largest logits, others -> -inf (ties at the k-th value are kept).
4. top-p (nucleus): keep the smallest set of most-likely tokens whose probability
   mass reaches ``top_p`` (the top-1 token is always kept).
5. ``softmax`` (subtracts the max, numerically stable) and ``torch.multinomial``
   with an optional ``torch.Generator`` for reproducibility.

With random (untrained) weights the generated ids are meaningless.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import torch

from qwen3_scratch.kv_cache import KVCache
from qwen3_scratch.model import Qwen3ForCausalLM

__all__ = [
    "GenerationConfig",
    "GenerationOutput",
    "apply_top_k",
    "apply_top_p",
    "generate",
    "greedy_generate",
    "select_next_token",
]


# ------------------------------------------------------------------ settings
@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings (validated on construction)."""

    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float = 1.0
    top_k: int | None = None
    top_p: float | None = None
    eos_token_id: int | tuple[int, ...] | None = None
    use_cache: bool = True

    def __post_init__(self) -> None:
        n = self.max_new_tokens
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError(f"max_new_tokens must be a non-negative int, got {n!r}")
        t = self.temperature
        if isinstance(t, bool) or not isinstance(t, int | float) or not math.isfinite(t) or t < 0:
            raise ValueError(f"temperature must be a finite number >= 0, got {t!r}")
        k = self.top_k
        if k is not None and (isinstance(k, bool) or not isinstance(k, int) or k < 1):
            raise ValueError(f"top_k must be None or an int >= 1, got {k!r}")
        p = self.top_p
        if p is not None and (
            isinstance(p, bool) or not isinstance(p, int | float) or not 0.0 < p <= 1.0
        ):
            raise ValueError(f"top_p must be None or in (0, 1], got {p!r}")
        if not self.do_sample and (k is not None or p is not None):
            raise ValueError("top_k / top_p only apply with do_sample=True")
        eos = self.eos_token_id
        if eos is not None:
            ids = (eos,) if isinstance(eos, int) and not isinstance(eos, bool) else eos
            if (
                not isinstance(ids, tuple)
                or not ids
                or not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in ids)
            ):
                raise ValueError(
                    f"eos_token_id must be an int >= 0 or a tuple of them, got {eos!r}"
                )

    @property
    def greedy(self) -> bool:
        """``temperature == 0`` is defined as greedy decoding (no randomness)."""
        return not self.do_sample or self.temperature == 0

    @property
    def eos_ids(self) -> tuple[int, ...]:
        eos = self.eos_token_id
        if eos is None:
            return ()
        return (eos,) if isinstance(eos, int) else tuple(eos)


@dataclass
class GenerationOutput:
    sequences: torch.Tensor  # [B, T0 + n] prompt followed by generated ids
    new_tokens: torch.Tensor  # [B, n]
    num_prompt_tokens: int
    # per-step logits of the last position, float32 [B, V] each (only if requested)
    step_logits: list[torch.Tensor] = field(default_factory=list)
    cache: KVCache | None = None  # final cache (use_cache=True)
    finished: torch.Tensor | None = None  # [B] bool, True where an EOS was produced


# ------------------------------------------------------------ token selection
def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Keep the ``k`` largest logits per row (ties with the k-th value kept); rest -> -inf."""
    if k >= logits.shape[-1]:
        return logits
    kth = torch.topk(logits, k, dim=-1).values[..., -1:]  # [B, 1]
    return logits.masked_fill(logits < kth, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus filter: keep the smallest prefix (by probability) with mass >= ``p``.

    A token is removed when the probability mass of the tokens strictly more likely
    than it already reaches ``p``; hence the most likely token always survives.
    """
    if p >= 1.0:
        return logits
    sorted_logits, order = torch.sort(logits, dim=-1, descending=True)
    probs = torch.softmax(sorted_logits, dim=-1)
    mass_before = torch.cumsum(probs, dim=-1) - probs  # mass of strictly better tokens
    remove_sorted = mass_before >= p
    remove_sorted[..., 0] = False
    remove = torch.zeros_like(remove_sorted).scatter(-1, order, remove_sorted)
    return logits.masked_fill(remove, float("-inf"))


def select_next_token(
    logits: torch.Tensor, config: GenerationConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """``[B, V]`` logits -> ``[B]`` next token ids according to ``config``."""
    if logits.dim() != 2:
        raise ValueError(f"logits must be [B, V], got {tuple(logits.shape)}")
    logits = logits.float()
    if config.greedy:
        return logits.argmax(dim=-1)
    logits = logits / config.temperature
    if config.top_k is not None:
        logits = apply_top_k(logits, config.top_k)
    if config.top_p is not None:
        logits = apply_top_p(logits, config.top_p)
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


# ---------------------------------------------------------------- main loop
def _prepare_ids(model: Qwen3ForCausalLM, input_ids: torch.Tensor, n_new: int):
    if not isinstance(input_ids, torch.Tensor) or input_ids.dim() not in (1, 2):
        raise ValueError("input_ids must be a tensor of shape [T] or [B, T]")
    if input_ids.dtype.is_floating_point or input_ids.dtype == torch.bool:
        raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
    squeeze = input_ids.dim() == 1
    ids = input_ids[None] if squeeze else input_ids
    if ids.shape[1] == 0:
        raise ValueError("the prompt must contain at least one token")
    limit = model.config.max_position_embeddings
    if ids.shape[1] + n_new > limit:
        raise ValueError(
            f"prompt length {ids.shape[1]} + max_new_tokens {n_new} exceeds "
            f"max_position_embeddings={limit}"
        )
    return ids, squeeze


def generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    config: GenerationConfig | None = None,
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    return_step_logits: bool = False,
    on_token: Callable[[int, torch.Tensor], None] | None = None,
    **overrides,
) -> GenerationOutput:
    """Generate up to ``max_new_tokens`` tokens after ``input_ids`` ``[T]`` or ``[B, T]``.

    Args:
        model: our :class:`Qwen3ForCausalLM` (put in ``eval()`` temporarily).
        input_ids: prompt ids; all rows have the same length (no padding).
        config: a :class:`GenerationConfig`; keyword ``overrides`` (e.g.
            ``max_new_tokens=8, do_sample=True``) build or replace fields.
        generator / seed: randomness source for sampling (``seed`` creates a CPU
            generator; pass a generator for other devices). Greedy ignores both.
        return_step_logits: keep the float32 ``[B, V]`` logits of every step.
        on_token: callback ``(step, next_ids [B])`` called as tokens are produced
            (token-by-token streaming).

    EOS: a row that produced an id in ``eos_token_id`` is finished and then keeps
    receiving the first EOS id; the loop stops early once every row is finished.
    """
    if config is None:
        config = GenerationConfig(**overrides)
    elif overrides:
        config = GenerationConfig(**{**config.__dict__, **overrides})
    if not isinstance(config, GenerationConfig):
        raise TypeError("config must be a GenerationConfig")
    ids, _ = _prepare_ids(model, input_ids, config.max_new_tokens)
    vocab = model.config.vocab_size
    if any(i >= vocab for i in config.eos_ids):
        raise ValueError(f"eos_token_id must be < vocab_size={vocab}")
    if generator is not None and seed is not None:
        raise ValueError("pass either generator or seed, not both")
    if seed is not None:
        generator = torch.Generator(device="cpu").manual_seed(seed)

    batch, n_prompt = ids.shape
    eos = torch.tensor(config.eos_ids, device=ids.device, dtype=ids.dtype)
    finished = torch.zeros(batch, dtype=torch.bool, device=ids.device)
    cache = model.new_cache() if config.use_cache else None
    new_tokens: list[torch.Tensor] = []
    step_logits: list[torch.Tensor] = []

    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            next_input = ids  # prefill: the whole prompt
            for step in range(config.max_new_tokens):
                if config.use_cache:
                    out = model(next_input, past_key_values=cache, use_cache=True, logits_to_keep=1)
                else:
                    seq = torch.cat([ids, *[t[:, None] for t in new_tokens]], dim=1)
                    out = model(seq, logits_to_keep=1)
                logits = out.logits[:, -1, :].float()  # [B, V]
                if return_step_logits:
                    step_logits.append(logits.clone())
                next_ids = select_next_token(logits, config, generator)
                if eos.numel():
                    next_ids = torch.where(finished, eos[0], next_ids)
                    finished |= torch.isin(next_ids, eos)
                new_tokens.append(next_ids)
                if on_token is not None:
                    on_token(step, next_ids)
                if eos.numel() and bool(finished.all()):
                    break
                next_input = next_ids[:, None]  # decode: one new token per row
    finally:
        model.train(was_training)

    gen = (
        torch.stack(new_tokens, dim=1)
        if new_tokens
        else torch.empty(batch, 0, dtype=ids.dtype, device=ids.device)
    )
    return GenerationOutput(
        sequences=torch.cat([ids, gen], dim=1),
        new_tokens=gen,
        num_prompt_tokens=n_prompt,
        step_logits=step_logits,
        cache=cache,
        finished=finished,
    )


def greedy_generate(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    *,
    eos_token_id: int | Sequence[int] | None = None,
    use_cache: bool = False,
) -> torch.Tensor:
    """M1.6 API: greedy ids ``[T + n]`` or ``[B, T + n]`` (same rank as the input).

    ``use_cache=False`` (default) keeps the original full-recompute behaviour.
    """
    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or max_new_tokens < 0
    ):
        raise ValueError(f"max_new_tokens must be a non-negative int, got {max_new_tokens!r}")
    squeeze = isinstance(input_ids, torch.Tensor) and input_ids.dim() == 1
    if eos_token_id is not None and not isinstance(eos_token_id, int):
        eos_token_id = tuple(eos_token_id)
    out = generate(
        model,
        input_ids,
        GenerationConfig(
            max_new_tokens=max_new_tokens, eos_token_id=eos_token_id, use_cache=use_cache
        ),
    )
    return out.sequences[0] if squeeze else out.sequences
