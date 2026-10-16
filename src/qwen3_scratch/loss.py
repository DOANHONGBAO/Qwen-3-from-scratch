"""Causal language-modelling (next-token) cross-entropy loss.

Label convention (same as Hugging Face): ``labels`` has the SAME shape ``[B, T]``
as ``input_ids`` - usually ``labels = input_ids`` - and the shift happens here,
exactly once:

    logits[:, t]  (prediction made after reading tokens 0..t)  is scored against
    labels[:, t + 1]                                            for t = 0 .. T-2.

The last position has no next token and is never scored. Targets equal to
``ignore_index`` (-100) are skipped (padding, prompt tokens, ...). The loss is the
mean over the remaining targets, computed in float32 (float64 for float64 logits) from raw logits
(``F.cross_entropy`` applies log-softmax internally, which is numerically stable).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["IGNORE_INDEX", "NoValidTargetsError", "causal_lm_loss", "shift_labels"]

IGNORE_INDEX = -100


class NoValidTargetsError(ValueError):
    """Raised when every next-token target is ignored (the mean would be 0 / 0 = NaN)."""


def _check(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> None:
    if not isinstance(logits, torch.Tensor) or logits.dim() != 3:
        raise ValueError(f"logits must have shape [B, T, V], got {getattr(logits, 'shape', None)}")
    if not isinstance(labels, torch.Tensor) or labels.dim() != 2:
        raise ValueError(f"labels must have shape [B, T], got {getattr(labels, 'shape', None)}")
    if labels.dtype.is_floating_point or labels.dtype == torch.bool:
        raise TypeError(f"labels must be an integer tensor, got {labels.dtype}")
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels {tuple(labels.shape)} must match logits[:2] {tuple(logits.shape[:2])} "
            "(labels are NOT pre-shifted; the shift happens inside the loss)"
        )
    vocab = logits.shape[-1]
    bad = (labels != ignore_index) & ((labels < 0) | (labels >= vocab))
    if bad.any():
        raise ValueError(
            f"labels must be in [0, {vocab}) or equal to ignore_index={ignore_index}; "
            f"found {labels[bad][:5].tolist()}"
        )


def shift_labels(labels: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    """Targets aligned with every logit position: ``out[:, t] = labels[:, t+1]``, last = ignore."""
    pad = torch.full_like(labels[:, :1], ignore_index)
    return torch.cat([labels[:, 1:], pad], dim=1)


def causal_lm_loss(
    logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = IGNORE_INDEX
) -> torch.Tensor:
    """Mean next-token cross-entropy over valid targets (scalar, float32).

    Raises :class:`NoValidTargetsError` if no target is valid (e.g. ``T == 1`` or
    all labels ignored) instead of silently returning NaN.
    """
    _check(logits, labels, ignore_index)
    # targets[:, t] = labels[:, t + 1]; the last position gets ignore_index. Scoring all
    # T positions (the last one is ignored) keeps the same row order as HF's loss, so
    # the float32 reduction is bit-identical; the result equals using logits[:, :-1].
    targets = shift_labels(labels, ignore_index)  # [B, T]
    n_valid = int((targets != ignore_index).sum())
    if n_valid == 0:
        raise NoValidTargetsError(
            "no valid next-token targets: sequence length 1 or all labels == ignore_index"
        )
    vocab = logits.shape[-1]
    compute_dtype = torch.promote_types(logits.dtype, torch.float32)  # fp32, or fp64 for fp64
    return F.cross_entropy(
        logits.to(compute_dtype).reshape(-1, vocab),
        targets.reshape(-1),
        ignore_index=ignore_index,
        reduction="mean",
    )
