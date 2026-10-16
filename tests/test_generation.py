"""Tests for greedy generation without KV cache (M1.6). CPU tier."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from qwen3_scratch.config import QWEN3_TINY
from qwen3_scratch.generation import greedy_generate
from qwen3_scratch.model import Qwen3ForCausalLM


def _model(seed=0, **overrides):
    cfg = dataclasses.replace(QWEN3_TINY, **overrides) if overrides else QWEN3_TINY
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg)


def test_greedy_steps_are_argmax_of_full_recomputation() -> None:
    m = _model()
    prompt = torch.tensor([[3, 17, 42, 9]])
    out = greedy_generate(m, prompt, max_new_tokens=5)
    assert out.shape == (1, 9) and torch.equal(out[:, :4], prompt)
    with torch.no_grad():
        for t in range(4, 9):
            expected = m(out[:, :t]).logits[0, -1].argmax()
            assert out[0, t] == expected


def test_deterministic_and_batched() -> None:
    m = _model()
    prompts = torch.tensor([[1, 2, 3], [7, 8, 9]])
    a = greedy_generate(m, prompts, 6)
    b = greedy_generate(m, prompts, 6)
    assert torch.equal(a, b)
    for i in range(2):  # batch rows equal individual runs
        assert torch.equal(a[i], greedy_generate(m, prompts[i], 6))


def test_one_dimensional_prompt_and_zero_tokens() -> None:
    m = _model()
    p = torch.tensor([5, 6])
    out = greedy_generate(m, p, 3)
    assert out.dim() == 1 and out.shape == (5,)
    assert torch.equal(greedy_generate(m, p, 0), p)


def test_restores_training_mode_and_does_not_track_grad() -> None:
    m = _model().train()
    out = greedy_generate(m, torch.tensor([[1, 2]]), 2)
    assert m.training and not out.requires_grad


def test_eos_stops_early_and_pads_finished_rows() -> None:
    m = _model()
    prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
    free = greedy_generate(m, prompt, 4)
    eos = int(free[0, 3])  # the first token row 0 generates
    out = greedy_generate(m, prompt, 4, eos_token_id=eos)
    assert out[0, 3] == eos and (out[0, 3:] == eos).all()
    if (free[1, 3:] != eos).all():  # row 1 never hits eos -> runs to the end
        assert out.shape[1] == 7 and torch.equal(out[1], free[1])
    single = greedy_generate(m, prompt[:1], 4, eos_token_id=eos)
    assert single.shape == (1, 4)  # stopped right after eos


def test_context_limit_is_enforced() -> None:
    m = _model(max_position_embeddings=10)
    greedy_generate(m, torch.zeros(1, 6, dtype=torch.long), 4)  # exactly 10: allowed
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        greedy_generate(m, torch.zeros(1, 6, dtype=torch.long), 5)
    with pytest.raises(ValueError, match="exceeds max_position_embeddings"):
        greedy_generate(m, torch.zeros(1, 11, dtype=torch.long), 0)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((torch.tensor([[1]]), -1), {}),
        ((torch.tensor([[1]]), 1.5), {}),
        ((torch.zeros(1, 0, dtype=torch.long), 1), {}),
        ((torch.zeros(1, 2, 3, dtype=torch.long), 1), {}),
        ((torch.tensor([[1]]), 1), {"eos_token_id": 256}),
    ],
)
def test_invalid_arguments(args, kwargs) -> None:
    with pytest.raises(ValueError):
        greedy_generate(_model(), *args, **kwargs)


def test_invalid_token_ids_rejected() -> None:
    with pytest.raises(ValueError):
        greedy_generate(_model(), torch.tensor([[300]]), 1)
