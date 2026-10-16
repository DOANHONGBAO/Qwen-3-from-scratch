"""Adapter around the OFFICIAL Qwen3 tokenizer files (kept separate from the model).

The model only ever sees integer ids; this module turns text into ids and back.
It reads the checkpoint's ``tokenizer.json`` with the Rust ``tokenizers`` library
(the same engine behind ``transformers``' fast tokenizer) and the special-token
settings from ``tokenizer_config.json``. We do not re-implement byte-level BPE in
this milestone.

Facts for Qwen/Qwen3-0.6B-Base (tokenizer_config.json, revision da87bfb):

* ``add_bos_token = false`` and ``bos_token = null``: no BOS is prepended.
  (``config.json``'s ``bos_token_id = 151643`` is NOT added by the tokenizer.)
* ``eos_token = pad_token = "<|endoftext|>"`` (id 151643).
* 26 added tokens, ids 151643..151668; 151643..151656 are "special"
  (``<|endoftext|>``, ``<|im_start|>``, ``<|im_end|>``, ...). Only 22 of them are in
  ``tokenizer.json``; the other four (``<tool_response>``, ``</tool_response>``,
  ``<think>``, ``</think>``) are declared only in ``tokenizer_config.json`` and are
  registered here, as ``transformers`` does (found by the weights-tier test).
* Tokenizer vocabulary = 151,669 ids < model ``vocab_size`` = 151,936: the last
  267 embedding rows are padding and never produced by the tokenizer.
* The normalizer is Unicode NFC, so ``decode(encode(s)) == unicodedata.normalize("NFC", s)``.

Special-token text (e.g. the literal string ``"<|endoftext|>"``) in the input IS
turned into the special id, as in ``transformers`` (``split_special_tokens=false``).
Pass ``allow_special_tokens=False`` to reject such input instead.
"""

from __future__ import annotations

import json
from pathlib import Path

__all__ = ["Qwen3Tokenizer"]


class Qwen3Tokenizer:
    """Text <-> ids using ``tokenizer.json`` (+ optional ``tokenizer_config.json``)."""

    def __init__(self, tokenizer_file: str | Path, config_file: str | Path | None = None) -> None:
        from tokenizers import Tokenizer  # optional dependency ([hf] extra)

        tokenizer_file = Path(tokenizer_file)
        if not tokenizer_file.is_file():
            raise FileNotFoundError(f"{tokenizer_file} not found")
        self._tok = Tokenizer.from_file(str(tokenizer_file))
        cfg: dict = {}
        if config_file is not None:
            cfg = json.loads(Path(config_file).read_text(encoding="utf-8"))
        self.config = cfg
        self.add_bos_token = bool(cfg.get("add_bos_token", False))
        self.bos_token = cfg.get("bos_token")
        self.eos_token = cfg.get("eos_token")
        self.pad_token = cfg.get("pad_token")
        added = cfg.get("added_tokens_decoder", {})
        self.registered_from_config = self._register_config_tokens(added)
        self.special_tokens: dict[str, int] = {
            entry["content"]: int(i) for i, entry in added.items() if entry.get("special")
        }
        if not self.special_tokens:  # fall back to tokenizer.json's added tokens
            self.special_tokens = {
                t.content: i for i, t in self._tok.get_added_tokens_decoder().items() if t.special
            }

    def _register_config_tokens(self, added: dict) -> list[str]:
        """Add tokens declared in ``tokenizer_config.json`` but missing from ``tokenizer.json``.

        Qwen3-0.6B-Base (da87bfb): ``tokenizer.json`` lists 22 added tokens
        (151643..151664) while ``tokenizer_config.json`` declares 26; ``<tool_response>``,
        ``</tool_response>``, ``<think>`` and ``</think>`` (151665..151668) exist only
        in the config. ``transformers`` registers them when loading, so we do the same
        and verify that each gets exactly its declared id.
        """
        from tokenizers import AddedToken

        registered = []
        for idx, entry in sorted(added.items(), key=lambda kv: int(kv[0])):
            idx, content = int(idx), entry["content"]
            existing = self._tok.token_to_id(content)
            if existing is not None:
                if existing != idx:
                    raise ValueError(
                        f"{content!r} has id {existing} in tokenizer.json but {idx} in "
                        "tokenizer_config.json"
                    )
                continue
            token = AddedToken(
                content,
                single_word=entry.get("single_word", False),
                lstrip=entry.get("lstrip", False),
                rstrip=entry.get("rstrip", False),
                normalized=entry.get("normalized", not entry.get("special", False)),
                special=entry.get("special", False),
            )
            if entry.get("special"):
                self._tok.add_special_tokens([token])
            else:
                self._tok.add_tokens([token])
            got = self._tok.token_to_id(content)
            if got != idx:
                raise ValueError(
                    f"added token {content!r} received id {got}, tokenizer_config.json "
                    f"declares {idx} (ids must be contiguous after the base vocabulary)"
                )
            registered.append(content)
        return registered

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> Qwen3Tokenizer:
        model_dir = Path(model_dir)
        cfg = model_dir / "tokenizer_config.json"
        return cls(model_dir / "tokenizer.json", cfg if cfg.is_file() else None)

    # ------------------------------------------------------------ ids
    def token_to_id(self, token: str | None) -> int | None:
        if token is None:
            return None
        idx = self._tok.token_to_id(token)
        if idx is None:
            raise KeyError(f"token {token!r} is not in the vocabulary")
        return idx

    @property
    def eos_token_id(self) -> int | None:
        return self.token_to_id(self.eos_token)

    @property
    def pad_token_id(self) -> int | None:
        return self.token_to_id(self.pad_token)

    @property
    def bos_token_id(self) -> int | None:
        return self.token_to_id(self.bos_token)

    @property
    def vocab_size(self) -> int:
        """Number of ids the tokenizer can produce (base vocab + added tokens)."""
        return self._tok.get_vocab_size(with_added_tokens=True)

    # --------------------------------------------------------- encode
    def _check_text(self, text: str, allow_special_tokens: bool) -> None:
        if not isinstance(text, str):
            raise TypeError(f"text must be str, got {type(text).__name__}")
        if not allow_special_tokens:
            found = [t for t in self.special_tokens if t in text]
            if found:
                raise ValueError(f"text contains special-token strings {found}")

    def encode(self, text: str, *, allow_special_tokens: bool = True) -> list[int]:
        """Text -> ids (no BOS for Qwen3; adds BOS only if ``add_bos_token`` is true)."""
        self._check_text(text, allow_special_tokens)
        ids = self._tok.encode(text, add_special_tokens=False).ids
        if self.add_bos_token and self.bos_token_id is not None:
            ids = [self.bos_token_id, *ids]
        return ids

    def encode_batch(
        self,
        texts: list[str],
        *,
        padding: bool = False,
        padding_side: str = "left",
        allow_special_tokens: bool = True,
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode several texts. Returns ``(ids, attention_mask)`` as Python lists.

        With ``padding=True`` every row is padded with ``pad_token_id`` to the longest
        length (``padding_side`` ``"left"`` or ``"right"``); mask = 1 for real tokens.
        Note: our KV-cache generation does not accept padded batches.
        """
        if padding_side not in ("left", "right"):
            raise ValueError("padding_side must be 'left' or 'right'")
        rows = [self.encode(t, allow_special_tokens=allow_special_tokens) for t in texts]
        masks = [[1] * len(r) for r in rows]
        if padding and rows:
            pad = self.pad_token_id
            if pad is None:
                raise ValueError("tokenizer has no pad token")
            width = max(len(r) for r in rows)
            for i, r in enumerate(rows):
                n = width - len(r)
                if padding_side == "left":
                    rows[i], masks[i] = [pad] * n + r, [0] * n + masks[i]
                else:
                    rows[i], masks[i] = r + [pad] * n, masks[i] + [0] * n
        return rows, masks

    # --------------------------------------------------------- decode
    def decode(
        self, ids, *, skip_special_tokens: bool = False, skip_out_of_vocab: bool = False
    ) -> str:
        """Ids -> text. Accepts a list/tuple of ints or a 1-D integer tensor.

        Ids outside the tokenizer vocabulary (e.g. the model's padded embedding rows
        151669..151935, which an untrained or unlucky model could still pick) raise
        ``ValueError`` unless ``skip_out_of_vocab=True`` drops them.
        """
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        ids = list(ids)
        if any(isinstance(i, bool) or not isinstance(i, int) for i in ids):
            raise TypeError("ids must be integers")
        bad = [i for i in ids if not 0 <= i < self.vocab_size]
        if bad and skip_out_of_vocab:
            ids = [i for i in ids if 0 <= i < self.vocab_size]
        elif bad:
            raise ValueError(
                f"ids {bad[:5]} are outside the tokenizer vocabulary [0, {self.vocab_size}) "
                "(the model's padded embedding rows have no text)"
            )
        return self._tok.decode(ids, skip_special_tokens=skip_special_tokens)

    def id_to_token(self, idx: int) -> str | None:
        return self._tok.id_to_token(idx)

    def __repr__(self) -> str:
        return (
            f"Qwen3Tokenizer(vocab_size={self.vocab_size}, eos={self.eos_token!r}, "
            f"pad={self.pad_token!r}, add_bos_token={self.add_bos_token})"
        )
