"""Tests for the causal language-modelling loss (M1.6). CPU tier."""

from __future__ import annotations

import math

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_TINY
from qwen3_scratch.loss import IGNORE_INDEX, NoValidTargetsError, causal_lm_loss, shift_labels
from qwen3_scratch.model import Qwen3ForCausalLM


def _manual_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Independent reference: explicit log-softmax and loop over valid targets."""
    log_probs = torch.log_softmax(logits.double(), dim=-1)
    total, count = 0.0, 0
    b, t, _ = logits.shape
    for i in range(b):
        for j in range(t - 1):
            target = int(labels[i, j + 1])
            if target != IGNORE_INDEX:
                total -= log_probs[i, j, target]
                count += 1
    return total / count


def _logits(b, t, v=11, seed=0):
    return torch.randn(b, t, v, generator=torch.Generator().manual_seed(seed))


def test_tiny_example_by_hand() -> None:
    # 1 sequence, 3 tokens, vocab 2. Position 0 predicts token 1, position 1 predicts token 2.
    logits = torch.tensor([[[2.0, 0.0], [0.0, 1.0], [5.0, 5.0]]])
    labels = torch.tensor([[1, 0, 1]])  # targets: labels[1]=0 for pos 0, labels[2]=1 for pos 1
    p0 = math.exp(2) / (math.exp(2) + 1)  # P(token 0 | pos 0)
    p1 = math.exp(1) / (math.exp(1) + 1)  # P(token 1 | pos 1)
    expected = -(math.log(p0) + math.log(p1)) / 2
    assert abs(causal_lm_loss(logits, labels).item() - expected) < 1e-6


@pytest.mark.parametrize(("b", "t"), [(1, 2), (2, 5), (4, 9)])
def test_matches_manual_reference(b, t) -> None:
    logits = _logits(b, t)
    labels = torch.randint(0, 11, (b, t), generator=torch.Generator().manual_seed(1))
    assert_close(causal_lm_loss(logits, labels).double(), _manual_loss(logits, labels))


def test_shift_happens_exactly_once() -> None:
    logits = _logits(2, 6)
    labels = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(2))
    loss = causal_lm_loss(logits, labels)
    unshifted = torch.nn.functional.cross_entropy(logits.reshape(-1, 11), labels.reshape(-1))
    double_shift = causal_lm_loss(logits[:, :-1], labels[:, 1:])
    assert not torch.allclose(loss, unshifted) and not torch.allclose(loss, double_shift)
    aligned = shift_labels(labels)
    assert aligned[:, :-1].tolist() == labels[:, 1:].tolist()
    assert (aligned[:, -1] == IGNORE_INDEX).all()
    via_shift = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11), aligned.reshape(-1), ignore_index=IGNORE_INDEX
    )
    assert_close(loss, via_shift)


def test_ignored_labels_are_excluded() -> None:
    logits = _logits(2, 6)
    labels = torch.randint(0, 11, (2, 6), generator=torch.Generator().manual_seed(3))
    masked = labels.clone()
    masked[0, 2:4] = IGNORE_INDEX
    masked[1, 5] = IGNORE_INDEX
    assert_close(causal_lm_loss(logits, masked).double(), _manual_loss(logits, masked))
    # Changing logits that predict ignored targets does not change the loss.
    other = logits.clone()
    other[0, 1:3] += 100.0  # positions 1, 2 predict labels[0, 2:4] (ignored)
    assert_close(causal_lm_loss(other, masked), causal_lm_loss(logits, masked))


def test_first_label_is_never_a_target() -> None:
    logits = _logits(1, 4)
    labels = torch.tensor([[IGNORE_INDEX, 1, 2, 3]])
    other = labels.clone()
    other[0, 0] = 7
    assert_close(causal_lm_loss(logits, labels), causal_lm_loss(logits, other))


@pytest.mark.parametrize(
    "labels",
    [torch.tensor([[5]]), torch.full((2, 4), IGNORE_INDEX), torch.tensor([[3, IGNORE_INDEX]])],
    ids=["T=1", "all-ignored", "only-target-ignored"],
)
def test_no_valid_targets_raises_instead_of_nan(labels) -> None:
    with pytest.raises(NoValidTargetsError):
        causal_lm_loss(_logits(*labels.shape), labels)


@pytest.mark.parametrize(
    ("labels", "exc"),
    [
        (torch.tensor([[1, 2, 11]]), ValueError),  # >= vocab
        (torch.tensor([[1, -2, 3]]), ValueError),  # negative, not ignore_index
        (torch.tensor([[1.0, 2.0, 3.0]]), TypeError),
        (torch.tensor([[1, 2]]), ValueError),  # shape mismatch
        (torch.tensor([1, 2, 3]), ValueError),  # [T]
    ],
)
def test_invalid_labels(labels, exc) -> None:
    with pytest.raises(exc):
        causal_lm_loss(_logits(1, 3), labels)


def test_loss_is_float32_and_differentiable_for_bf16_logits() -> None:
    logits = _logits(2, 5).bfloat16().requires_grad_()
    labels = torch.randint(0, 11, (2, 5), generator=torch.Generator().manual_seed(4))
    loss = causal_lm_loss(logits, labels)
    assert loss.dtype == torch.float32
    loss.backward()
    assert logits.grad is not None and logits.grad.dtype == torch.bfloat16
    assert logits.grad[:, -1].abs().max() == 0  # last position is never scored


def test_gradient_is_softmax_minus_one_hot() -> None:
    logits = _logits(1, 3, v=5, seed=5).double().requires_grad_()
    labels = torch.tensor([[0, 2, 4]])
    causal_lm_loss(logits, labels).backward()
    probs = torch.softmax(logits.detach(), dim=-1)
    expected = torch.zeros_like(probs)
    for t, target in [(0, 2), (1, 4)]:
        expected[0, t] = probs[0, t]
        expected[0, t, target] -= 1
    assert_close(logits.grad, expected / 2)  # mean over 2 targets


def test_model_loss_equals_function_and_random_init_is_near_log_v() -> None:
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(QWEN3_TINY).eval()
    ids = torch.randint(0, 256, (4, 32), generator=torch.Generator().manual_seed(6))
    out = m(ids, labels=ids)
    assert torch.equal(out.loss, causal_lm_loss(out.logits, ids))
    # Untrained model ~ uniform over the vocabulary: loss close to ln(256) = 5.545.
    assert abs(out.loss.item() - math.log(256)) < 0.2
    with pytest.raises(NoValidTargetsError):
        m(ids[:, :1], labels=ids[:, :1])
