"""Generation with sampling and KV cache (M2.2). CPU tier, Tiny Qwen3."""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_TINY
from qwen3_scratch.generation import (
    GenerationConfig,
    apply_top_k,
    apply_top_p,
    generate,
    greedy_generate,
    select_next_token,
)
from qwen3_scratch.model import Qwen3ForCausalLM


def _model(dtype=torch.float32, seed=0, **overrides):
    cfg = dataclasses.replace(QWEN3_TINY, **overrides) if overrides else QWEN3_TINY
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg, dtype=dtype).eval()


PROMPT = torch.tensor([[3, 17, 42, 9, 100], [5, 6, 7, 8, 200]])


# ------------------------------------------------------------ filters
def test_top_k_keeps_k_largest() -> None:
    logits = torch.tensor([[1.0, 5.0, 3.0, 4.0, 2.0]])
    out = apply_top_k(logits, 2)
    assert torch.isfinite(out).tolist() == [[False, True, False, True, False]]
    assert torch.equal(apply_top_k(logits, 10), logits)
    ties = apply_top_k(torch.tensor([[1.0, 3.0, 3.0, 0.0]]), 1)
    assert torch.isfinite(ties).sum() == 2  # ties at the k-th value are kept


def test_top_p_minimal_nucleus() -> None:
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    logits = probs.log()
    keep = lambda p: torch.isfinite(apply_top_p(logits, p)).tolist()[0]  # noqa: E731
    assert keep(0.5) == [True, False, False, False]
    assert keep(0.6) == [True, True, False, False]
    assert keep(0.8) == [True, True, False, False]  # 0.5+0.3 reaches 0.8
    assert keep(0.81) == [True, True, True, False]
    assert keep(1.0) == [True] * 4
    assert keep(1e-6) == [True, False, False, False]  # top-1 always kept
    shuffled = logits[:, [2, 0, 3, 1]]  # unsorted input: same decision per token
    assert torch.isfinite(apply_top_p(shuffled, 0.6)).tolist()[0] == [False, True, False, True]


def test_greedy_and_temperature_zero_are_argmax() -> None:
    logits = torch.randn(3, 50, generator=torch.Generator().manual_seed(0))
    for cfg in (GenerationConfig(), GenerationConfig(do_sample=True, temperature=0.0)):
        assert torch.equal(select_next_token(logits, cfg), logits.argmax(-1))


def test_sampling_respects_filters() -> None:
    logits = torch.randn(1, 100, generator=torch.Generator().manual_seed(1))
    top5 = set(torch.topk(logits, 5).indices[0].tolist())
    cfg = GenerationConfig(do_sample=True, top_k=5)
    g = torch.Generator().manual_seed(0)
    picks = {int(select_next_token(logits, cfg, g)) for _ in range(300)}
    assert picks <= top5 and len(picks) > 1
    cfg1 = GenerationConfig(do_sample=True, top_k=1)
    assert int(select_next_token(logits, cfg1, g)) == int(logits.argmax())


def test_sampling_frequencies_follow_softmax() -> None:
    logits = torch.tensor([[2.0, 1.0, 0.0]])
    cfg = GenerationConfig(do_sample=True, temperature=1.0)
    g = torch.Generator().manual_seed(0)
    n = 20_000
    counts = torch.bincount(
        torch.stack([select_next_token(logits, cfg, g) for _ in range(n)]).flatten(), minlength=3
    )
    freq = counts.double() / n
    assert_close(freq, torch.softmax(logits.double(), -1)[0], atol=0.015, rtol=0)


def test_low_temperature_sharpens() -> None:
    logits = torch.tensor([[2.0, 1.9, 0.0]])
    cfg = GenerationConfig(do_sample=True, temperature=0.05)
    g = torch.Generator().manual_seed(0)
    picks = torch.stack([select_next_token(logits, cfg, g) for _ in range(200)])
    assert (picks == 0).float().mean() > 0.8


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": -1},
        {"max_new_tokens": 1.5},
        {"temperature": -0.1},
        {"temperature": float("nan")},
        {"temperature": float("inf")},
        {"do_sample": True, "top_k": 0},
        {"do_sample": True, "top_k": 2.0},
        {"do_sample": True, "top_p": 0.0},
        {"do_sample": True, "top_p": 1.5},
        {"top_k": 5},  # filters without sampling are a configuration error
        {"top_p": 0.9},
        {"eos_token_id": -1},
        {"eos_token_id": ()},
        {"eos_token_id": True},
    ],
)
def test_invalid_generation_config(kwargs) -> None:
    with pytest.raises(ValueError):
        GenerationConfig(**kwargs)


# ------------------------------------------------------- full generation
def test_cached_and_uncached_greedy_agree() -> None:
    m = _model(dtype=torch.float64)
    a = generate(m, PROMPT, max_new_tokens=10, use_cache=True, return_step_logits=True)
    b = generate(m, PROMPT, max_new_tokens=10, use_cache=False, return_step_logits=True)
    assert torch.equal(a.sequences, b.sequences)
    for la, lb in zip(a.step_logits, b.step_logits, strict=True):
        assert_close(la, lb, atol=1e-5, rtol=1e-5)  # float32 copies of float64 logits
    assert a.cache.seq_len == PROMPT.shape[1] + 10 - 1  # last token is never fed back
    assert b.cache is None


def test_cached_and_uncached_float32_logits_close() -> None:
    m = _model()
    a = generate(m, PROMPT, max_new_tokens=8, use_cache=True, return_step_logits=True)
    b = generate(m, PROMPT, max_new_tokens=8, use_cache=False, return_step_logits=True)
    worst = max(
        float((x - y).abs().max()) for x, y in zip(a.step_logits, b.step_logits, strict=True)
    )
    assert worst < 1e-4
    assert torch.equal(a.sequences, b.sequences)


def test_greedy_matches_argmax_of_full_recompute() -> None:
    m = _model(dtype=torch.float64)
    out = generate(m, PROMPT[:1], max_new_tokens=6).sequences
    with torch.no_grad():
        for t in range(5, 11):
            assert out[0, t] == m(out[:, :t]).logits[0, -1].argmax()


def test_sampling_reproducible_with_seed_and_generator() -> None:
    m = _model()
    kw = {"max_new_tokens": 12, "do_sample": True, "temperature": 1.3, "top_k": 50, "top_p": 0.95}
    a = generate(m, PROMPT, seed=123, **kw).sequences
    b = generate(m, PROMPT, seed=123, **kw).sequences
    c = generate(m, PROMPT, generator=torch.Generator().manual_seed(123), **kw).sequences
    d = generate(m, PROMPT, seed=7, **kw).sequences
    assert torch.equal(a, b) and torch.equal(a, c) and not torch.equal(a, d)
    e = generate(m, PROMPT, seed=123, use_cache=False, **kw).sequences
    assert torch.equal(a, e)  # same random stream, (near-)identical logits
    with pytest.raises(ValueError, match="either"):
        generate(m, PROMPT, seed=1, generator=torch.Generator(), **kw)


def test_max_new_tokens_and_zero() -> None:
    m = _model()
    out = generate(m, PROMPT, max_new_tokens=4)
    assert out.new_tokens.shape == (2, 4) and out.sequences.shape == (2, 9)
    assert out.num_prompt_tokens == 5
    zero = generate(m, PROMPT, max_new_tokens=0)
    assert torch.equal(zero.sequences, PROMPT) and zero.new_tokens.shape == (2, 0)


@pytest.mark.parametrize("use_cache", [True, False])
def test_eos_stopping(use_cache) -> None:
    m = _model()
    free = generate(m, PROMPT, max_new_tokens=6, use_cache=use_cache).new_tokens
    eos = int(free[0, 1])  # second token of row 0
    out = generate(m, PROMPT, max_new_tokens=6, eos_token_id=eos, use_cache=use_cache)
    row0 = out.new_tokens[0].tolist()
    first = row0.index(eos)
    assert all(t == eos for t in row0[first:])  # finished rows keep emitting EOS
    assert bool(out.finished[0])
    if out.finished.all():
        assert out.new_tokens.shape[1] < 6
    multi = generate(m, PROMPT, max_new_tokens=6, eos_token_id=(9999 % 256, eos))
    assert bool(multi.finished[0])


def test_eos_stops_early_when_all_finished() -> None:
    m = _model()
    first = generate(m, PROMPT, max_new_tokens=1).new_tokens[:, 0]
    out = generate(m, PROMPT, max_new_tokens=10, eos_token_id=tuple(set(first.tolist())))
    assert out.new_tokens.shape[1] == 1 and bool(out.finished.all())


def test_on_token_streaming_callback() -> None:
    m = _model()
    seen = []
    out = generate(
        m, PROMPT, max_new_tokens=5, on_token=lambda s, ids: seen.append((s, ids.clone()))
    )
    assert [s for s, _ in seen] == list(range(5))
    assert torch.equal(torch.stack([ids for _, ids in seen], 1), out.new_tokens)


def test_context_limit_and_input_validation() -> None:
    m = _model(max_position_embeddings=10)
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        generate(m, PROMPT, max_new_tokens=6)
    generate(m, PROMPT, max_new_tokens=5)  # exactly at the limit is fine
    with pytest.raises(TypeError):
        generate(m, PROMPT.float(), max_new_tokens=1)
    with pytest.raises(ValueError, match="at least one"):
        generate(m, torch.zeros(1, 0, dtype=torch.long), max_new_tokens=1)
    with pytest.raises(ValueError, match="vocab_size"):
        generate(m, PROMPT, max_new_tokens=1, eos_token_id=256)


def test_generation_is_gradient_free_and_restores_mode() -> None:
    m = _model().train()
    out = generate(m, PROMPT, max_new_tokens=3, return_step_logits=True)
    assert m.training
    assert not out.sequences.requires_grad and not out.step_logits[0].requires_grad


def test_each_call_uses_a_fresh_cache() -> None:
    m = _model()
    a = generate(m, PROMPT, max_new_tokens=4)
    b = generate(m, PROMPT, max_new_tokens=4)
    assert a.cache is not b.cache and torch.equal(a.sequences, b.sequences)


def test_greedy_generate_wrapper_with_cache() -> None:
    m = _model(dtype=torch.float64)
    assert torch.equal(greedy_generate(m, PROMPT, 6, use_cache=True), greedy_generate(m, PROMPT, 6))
    one = greedy_generate(m, PROMPT[0], 3, use_cache=True)
    assert one.dim() == 1 and one.shape == (8,)


def test_config_object_and_overrides() -> None:
    m = _model()
    cfg = GenerationConfig(max_new_tokens=3)
    assert generate(m, PROMPT, cfg).new_tokens.shape[1] == 3
    assert generate(m, PROMPT, cfg, max_new_tokens=2).new_tokens.shape[1] == 2
    with pytest.raises(TypeError):
        generate(m, PROMPT, {"max_new_tokens": 2})
