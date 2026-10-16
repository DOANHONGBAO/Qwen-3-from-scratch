"""Educational dynamic KV cache for autoregressive decoding.

Why a cache? In causal attention the key ``k_j`` and value ``v_j`` of token ``j``
depend only on tokens ``0..j``. Once computed they never change, so when token
``t+1`` is generated we can reuse ``k_0..k_t`` / ``v_0..v_t`` instead of running
the whole prefix through the model again. Queries are NOT cached: only the newest
token's query is needed, because older tokens' outputs were already produced.

Layout (one entry per decoder layer, native KV heads, AFTER QK-Norm and RoPE)::

    keys[layer]    [B, Nkv, T_cached, D]   rotated keys (RoPE already applied)
    values[layer]  [B, Nkv, T_cached, D]   values (never normalised or rotated)

* Keys are stored post-RoPE, so a cached key is never rotated again.
* Only ``Nkv`` heads are stored; grouped-query expansion to ``Nq`` heads happens
  inside attention at compute time (Qwen3-0.6B: 8 instead of 16 heads -> 2x smaller).
* Memory: ``2 (K,V) * L * B * Nkv * T * D * bytes``. Qwen3-0.6B (L=28, Nkv=8,
  D=128) needs 112 KiB per token in bfloat16 (229,376 bytes in float32).

Ownership and lifecycle: the caller creates a cache (``KVCache(num_layers)`` or
``KVCache.from_config(config)``), passes it to the model with ``use_cache=True``,
and the model APPENDS to it in place. One cache = one inference session with a
fixed batch. ``reset()`` empties it for a new session. The model never stores a
cache on itself, so unrelated sessions cannot share state by accident.

Simple concatenation (``torch.cat``) is used for clarity: every append copies the
cache (O(T) per step). Pre-allocated or paged caches are later optimisations.
"""

from __future__ import annotations

import torch

from qwen3_scratch.config import Qwen3Config

__all__ = ["KVCache", "kv_cache_bytes"]


def kv_cache_bytes(
    config: Qwen3Config, seq_len: int, *, batch_size: int = 1, bytes_per_element: int = 2
) -> int:
    """Analytic KV cache size: ``2 * L * B * Nkv * T * D * bytes_per_element``."""
    return (
        2
        * config.num_hidden_layers
        * batch_size
        * config.num_key_value_heads
        * seq_len
        * config.head_dim
        * bytes_per_element
    )


class KVCache:
    """Per-layer key/value storage that grows by concatenation along the time axis.

    Args:
        num_layers: number of decoder layers (one K/V pair each).
        max_length: optional hard limit on cached tokens (usually
            ``config.max_position_embeddings``); exceeding it raises ``ValueError``.

    Invariants checked on every :meth:`update`:

    * layer index in range;
    * new K/V are ``[B, Nkv, T_new, D]`` with ``T_new >= 1`` and identical shapes;
    * batch size, KV heads, head_dim, dtype and device match what is already cached
      (first append fixes them for the whole cache);
    * each layer is appended exactly once per model forward: a layer may only be
      updated while its length equals :attr:`seq_len` (prevents double appends and
      out-of-order layers that would corrupt positions).
    """

    def __init__(self, num_layers: int, *, max_length: int | None = None) -> None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be a positive int, got {num_layers!r}")
        if max_length is not None and (
            isinstance(max_length, bool) or not isinstance(max_length, int) or max_length <= 0
        ):
            raise ValueError(f"max_length must be a positive int or None, got {max_length!r}")
        self.num_layers = num_layers
        self.max_length = max_length
        self.reset()

    @classmethod
    def from_config(cls, config: Qwen3Config) -> KVCache:
        return cls(config.num_hidden_layers, max_length=config.max_position_embeddings)

    # ------------------------------------------------------------- state
    def reset(self) -> None:
        """Forget everything (new inference session; batch/dtype may change)."""
        self.keys: list[torch.Tensor | None] = [None] * self.num_layers
        self.values: list[torch.Tensor | None] = [None] * self.num_layers
        self._seq_len = 0  # tokens committed by completed model forwards

    @property
    def seq_len(self) -> int:
        """Number of tokens cached by every layer (= absolute position of the next token)."""
        return self._seq_len

    def layer_length(self, layer_idx: int) -> int:
        self._check_layer(layer_idx)
        k = self.keys[layer_idx]
        return 0 if k is None else k.shape[2]

    @property
    def is_empty(self) -> bool:
        return self._seq_len == 0 and all(k is None for k in self.keys)

    def _reference(self) -> torch.Tensor | None:
        return next((k for k in self.keys if k is not None), None)

    @property
    def batch_size(self) -> int | None:
        ref = self._reference()
        return None if ref is None else ref.shape[0]

    def nbytes(self) -> int:
        """Bytes held by all cached K and V tensors."""
        return sum(
            t.numel() * t.element_size() for t in (*self.keys, *self.values) if t is not None
        )

    # ------------------------------------------------------------ update
    def _check_layer(self, layer_idx: int) -> None:
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError(f"layer_idx must be an int, got {layer_idx!r}")
        if not 0 <= layer_idx < self.num_layers:
            raise IndexError(f"layer_idx {layer_idx} out of range [0, {self.num_layers})")

    def check_can_append(self, num_new: int) -> None:
        """Raise before any layer is touched if ``num_new`` more tokens would not fit."""
        if self.max_length is not None and self._seq_len + num_new > self.max_length:
            raise ValueError(
                f"KV cache overflow: {self._seq_len} cached + {num_new} new tokens > "
                f"max_length={self.max_length}"
            )

    def update(
        self, layer_idx: int, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Append ``key``/``value`` ``[B, Nkv, T_new, D]`` to ``layer_idx``; return full K/V.

        The returned tensors are ``[B, Nkv, T_cached + T_new, D]`` and are what the
        attention of this layer must attend over.
        """
        self._check_layer(layer_idx)
        for name, t in (("key", key), ("value", value)):
            if not isinstance(t, torch.Tensor) or t.dim() != 4:
                raise ValueError(
                    f"{name} must have shape [B, Nkv, T, D], got {getattr(t, 'shape', None)}"
                )
        if key.shape != value.shape:
            raise ValueError(f"key {tuple(key.shape)} and value {tuple(value.shape)} differ")
        if key.dtype != value.dtype or key.device != value.device:
            raise ValueError("key and value must share dtype and device")
        if key.shape[2] == 0:
            raise ValueError("cannot append zero tokens")
        if self.layer_length(layer_idx) != self._seq_len:
            raise RuntimeError(
                f"layer {layer_idx} already holds {self.layer_length(layer_idx)} tokens but the "
                f"cache has committed {self._seq_len}: each layer must be updated exactly once "
                "per forward (call commit() after the last layer)"
            )
        ref = self._reference()
        if ref is not None:
            b, nkv, _, d = ref.shape
            if (key.shape[0], key.shape[1], key.shape[3]) != (b, nkv, d):
                raise ValueError(
                    f"new K/V {tuple(key.shape)} incompatible with cached [B={b}, Nkv={nkv}, "
                    f"T, D={d}] (different batch? call reset() for a new session)"
                )
            if key.dtype != ref.dtype or key.device != ref.device:
                raise ValueError(
                    f"cache holds {ref.dtype} on {ref.device}, got {key.dtype} on {key.device}"
                )
        self.check_can_append(key.shape[2])
        old_k, old_v = self.keys[layer_idx], self.values[layer_idx]
        new_k = key if old_k is None else torch.cat([old_k, key], dim=2)
        new_v = value if old_v is None else torch.cat([old_v, value], dim=2)
        self.keys[layer_idx], self.values[layer_idx] = new_k, new_v
        return new_k, new_v

    def commit(self, num_new: int) -> None:
        """Advance :attr:`seq_len` after ALL layers appended ``num_new`` tokens."""
        target = self._seq_len + num_new
        lengths = [self.layer_length(i) for i in range(self.num_layers)]
        if any(n != target for n in lengths):
            raise RuntimeError(f"cannot commit {num_new} tokens: per-layer lengths are {lengths}")
        self._seq_len = target

    def rollback(self) -> None:
        """Drop uncommitted appends (used when a forward fails half-way through the layers)."""
        n = self._seq_len
        for i in range(self.num_layers):
            if self.keys[i] is not None and self.keys[i].shape[2] != n:
                self.keys[i] = self.keys[i][:, :, :n] if n else None
                self.values[i] = self.values[i][:, :, :n] if n else None

    def get(self, layer_idx: int) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        self._check_layer(layer_idx)
        return self.keys[layer_idx], self.values[layer_idx]

    def __repr__(self) -> str:
        ref = self._reference()
        shape = None if ref is None else (ref.shape[0], ref.shape[1], "T", ref.shape[3])
        return (
            f"KVCache(num_layers={self.num_layers}, seq_len={self._seq_len}, "
            f"max_length={self.max_length}, layout={shape}, nbytes={self.nbytes()})"
        )
