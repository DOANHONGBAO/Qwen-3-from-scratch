"""Qwen3 decoder-only language model built from the verified components.

    input_ids [B, T]
      -> embed_tokens                         [B, T, H]   (no absolute position embedding)
      -> N x Qwen3DecoderLayer                [B, T, H]   (one RoPE table shared by all layers)
      -> norm (final RMSNorm)                 [B, T, H]   = last_hidden_state
      -> lm_head (tied to embed_tokens)       [B, T, V]   = logits
      -> causal_lm_loss(logits, labels)       scalar      (optional)

Parameter names follow the HF checkpoint (``model.embed_tokens``,
``model.layers.{i}.*``, ``model.norm``, ``lm_head``), so
``Qwen3ForCausalLM.load_state_dict(hf_model.state_dict())`` maps 1:1.

Not implemented on purpose (later milestones): KV cache, pretrained checkpoint
loading (M2), tokenizer (M2), dropout/training loop, sliding-window attention.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from qwen3_scratch.config import Qwen3Config
from qwen3_scratch.layers.attention import causal_mask
from qwen3_scratch.layers.decoder import Qwen3DecoderLayer
from qwen3_scratch.layers.rmsnorm import RMSNorm
from qwen3_scratch.layers.rope import RotaryEmbedding
from qwen3_scratch.loss import IGNORE_INDEX, causal_lm_loss

__all__ = ["CausalLMOutput", "Qwen3ForCausalLM", "Qwen3Model", "Qwen3ModelOutput"]

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


@dataclass
class Qwen3ModelOutput:
    last_hidden_state: torch.Tensor  # [B, T, H], after the final RMSNorm
    # (embeddings, output of layer 0, ..., output of layer N-1), each [B, T, H],
    # all BEFORE the final norm; only when output_hidden_states=True.
    hidden_states: tuple[torch.Tensor, ...] | None = None


@dataclass
class CausalLMOutput:
    logits: torch.Tensor  # [B, T, V]
    loss: torch.Tensor | None = None  # scalar, only when labels are given
    last_hidden_state: torch.Tensor | None = None  # [B, T, H]
    hidden_states: tuple[torch.Tensor, ...] | None = None


def _padding_to_allowed(attention_mask: torch.Tensor, batch: int, seq_len: int) -> torch.Tensor:
    """Combine a [B, T] padding mask (1 = real token, 0 = padding) with causality."""
    if not isinstance(attention_mask, torch.Tensor):
        raise TypeError("attention_mask must be a tensor of shape [B, T] (1 = token, 0 = padding)")
    if attention_mask.dtype.is_floating_point:
        raise TypeError(
            "attention_mask must be a boolean or 0/1 integer PADDING mask; additive float "
            "attention biases are not accepted (pass 1/0, not 0/-inf)"
        )
    if attention_mask.shape != (batch, seq_len):
        raise ValueError(
            f"attention_mask must have shape {(batch, seq_len)}, got {tuple(attention_mask.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        if ((attention_mask != 0) & (attention_mask != 1)).any():
            raise ValueError("attention_mask must contain only 0 and 1")
        attention_mask = attention_mask.bool()
    idx = torch.arange(seq_len, device=attention_mask.device)
    # Causality is by token INDEX (order in the sequence), independent of position_ids.
    return causal_mask(idx, idx)[None] & attention_mask[:, None, :]  # [B, T, T]


class Qwen3Model(nn.Module):
    """Embedding + decoder stack + final RMSNorm. Returns hidden states, not logits."""

    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if config.attention_dropout != 0.0:
            raise NotImplementedError("attention dropout is not implemented (Qwen3 uses 0.0)")
        self.config = config
        factory = {"device": device, "dtype": dtype}
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, **factory)
        self.layers = nn.ModuleList(
            Qwen3DecoderLayer.from_config(config, layer_idx=i, **factory)
            for i in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, **factory)
        self.rotary_emb = RotaryEmbedding(config.head_dim, config.rope_theta, device=device)

    # ----------------------------------------------------------- validation
    def _embed(self, input_ids, inputs_embeds) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("pass exactly one of input_ids or inputs_embeds")
        if inputs_embeds is not None:
            if inputs_embeds.dim() != 3 or inputs_embeds.shape[-1] != self.config.hidden_size:
                raise ValueError(
                    f"inputs_embeds must have shape [B, T, {self.config.hidden_size}], "
                    f"got {tuple(inputs_embeds.shape)}"
                )
            return inputs_embeds
        if not isinstance(input_ids, torch.Tensor):
            raise TypeError("input_ids must be a torch.Tensor of shape [B, T]")
        if input_ids.dtype.is_floating_point or input_ids.dtype == torch.bool:
            raise TypeError(f"input_ids must be an integer tensor, got {input_ids.dtype}")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must have shape [B, T], got {tuple(input_ids.shape)}")
        if input_ids.numel() and (
            int(input_ids.min()) < 0 or int(input_ids.max()) >= self.config.vocab_size
        ):
            raise ValueError(f"token ids must be in [0, {self.config.vocab_size})")
        return self.embed_tokens(input_ids)

    def _positions(self, position_ids, batch: int, seq_len: int, device) -> torch.Tensor:
        max_pos = self.config.max_position_embeddings
        if position_ids is None:
            if seq_len > max_pos:
                raise ValueError(
                    f"sequence length {seq_len} exceeds max_position_embeddings={max_pos}"
                )
            return torch.arange(seq_len, device=device)[None]  # [1, T]
        if position_ids.dtype.is_floating_point or position_ids.dtype == torch.bool:
            raise TypeError("position_ids must be an integer tensor")
        if position_ids.dim() == 1:
            position_ids = position_ids[None]
        if position_ids.dim() != 2 or position_ids.shape[-1] != seq_len:
            raise ValueError(f"position_ids must have shape [T] or [B, T] with T={seq_len}")
        if position_ids.shape[0] not in (1, batch):
            raise ValueError(f"position_ids batch {position_ids.shape[0]} != {batch}")
        if int(position_ids.min()) < 0 or int(position_ids.max()) >= max_pos:
            raise ValueError(f"position_ids must be in [0, {max_pos})")
        return position_ids

    # -------------------------------------------------------------- forward
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
    ) -> Qwen3ModelOutput:
        """Args:
        input_ids: ``[B, T]`` token ids in ``[0, vocab_size)``.
        attention_mask: optional ``[B, T]`` PADDING mask (1/True = real token,
            0/False = padding). Combined with the causal mask; padding keys are
            never attended. Pad token ids are NOT masked automatically.
        position_ids: optional ``[T]`` or ``[B, T]``; default ``0..T-1``. Pass
            them explicitly for left-padded batches (e.g. ``cumsum(mask) - 1``).
        inputs_embeds: alternative to ``input_ids``, ``[B, T, H]``.
        output_hidden_states: also return embeddings and every layer output.
        """
        hidden = self._embed(input_ids, inputs_embeds)  # [B, T, H]
        batch, seq_len, _ = hidden.shape
        if seq_len == 0:
            raise ValueError("sequence length must be at least 1")
        positions = self._positions(position_ids, batch, seq_len, hidden.device)
        cos, sin = self.rotary_emb(positions, dtype=hidden.dtype)  # [1|B, T, D], shared
        allowed = None  # None -> plain causal mask inside attention
        if attention_mask is not None:
            allowed = _padding_to_allowed(attention_mask, batch, seq_len)

        all_hidden = [hidden] if output_hidden_states else None
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, allowed)
            if all_hidden is not None:
                all_hidden.append(hidden)
        hidden = self.norm(hidden)
        return Qwen3ModelOutput(hidden, tuple(all_hidden) if all_hidden is not None else None)


class Qwen3ForCausalLM(nn.Module):
    """``Qwen3Model`` + LM head (no bias), optional weight tying and causal-LM loss.

    With ``config.tie_word_embeddings`` the LM head *is* the embedding matrix: the
    same ``nn.Parameter`` object is registered under both names, so there is one
    tensor, one gradient and ``vocab_size * hidden_size`` fewer parameters.
    """

    def __init__(
        self,
        config: Qwen3Config,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        init_weights: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config, device=device, dtype=dtype)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False, device=device, dtype=dtype
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight  # share, do not copy
        if init_weights and device != "meta":
            self.init_weights()

    @classmethod
    def from_config(cls, config: Qwen3Config, **kwargs) -> Qwen3ForCausalLM:
        """Build with the dtype named in ``config.dtype``."""
        return cls(config, dtype=_DTYPES[config.dtype], **kwargs)

    @torch.no_grad()
    def init_weights(self) -> None:
        """HF-style init: Linear/Embedding ~ N(0, initializer_range), RMSNorm = 1.

        Uses the global torch RNG (seed it with ``torch.manual_seed``). A tied
        matrix is initialised once (``parameters()`` yields shared tensors once).
        """
        std = self.config.initializer_range
        for module in self.modules():
            if isinstance(module, nn.Linear | nn.Embedding) and not (
                module is self.lm_head and self.config.tie_word_embeddings
            ):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if getattr(module, "bias", None) is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                module.reset_parameters()

    @property
    def weights_tied(self) -> bool:
        return self.lm_head.weight is self.model.embed_tokens.weight

    def num_parameters(self, trainable_only: bool = False) -> int:
        """Unique parameter count (a tied matrix counts once)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        output_hidden_states: bool = False,
        ignore_index: int = IGNORE_INDEX,
    ) -> CausalLMOutput:
        """Logits ``[B, T, V]`` and, if ``labels`` ``[B, T]`` are given, the mean
        next-token loss (labels are shifted inside; see :mod:`qwen3_scratch.loss`)."""
        out = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
        )
        logits = self.lm_head(out.last_hidden_state)  # [B, T, V]
        loss = None
        if labels is not None:
            loss = causal_lm_loss(logits, labels, ignore_index=ignore_index)
        return CausalLMOutput(logits, loss, out.last_hidden_state, out.hidden_states)
