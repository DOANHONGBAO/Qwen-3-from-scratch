"""Load official Qwen3 checkpoint tensors into OUR model, with an explicit name map.

Checkpoint format: Hugging Face ``*.safetensors`` (a JSON header with name, dtype,
shape and byte offsets, followed by raw little-endian tensor bytes; no pickled
Python code, so loading cannot execute anything). Large models are split into
shards listed in ``model.safetensors.index.json``; Qwen3-0.6B-Base ships a single
``model.safetensors`` (bfloat16, 596,049,920 values, no ``lm_head.weight`` because
the LM head is tied to ``model.embed_tokens.weight``).

Name map (checkpoint name -> our parameter, ``i`` = layer index)::

    model.embed_tokens.weight                         [V, H]
    model.layers.i.input_layernorm.weight             [H]
    model.layers.i.self_attn.q_proj.weight            [Nq*D, H]
    model.layers.i.self_attn.k_proj.weight            [Nkv*D, H]
    model.layers.i.self_attn.v_proj.weight            [Nkv*D, H]
    model.layers.i.self_attn.o_proj.weight            [H, Nq*D]
    model.layers.i.self_attn.q_norm.weight            [D]
    model.layers.i.self_attn.k_norm.weight            [D]
    model.layers.i.post_attention_layernorm.weight    [H]
    model.layers.i.mlp.gate_proj.weight               [I, H]
    model.layers.i.mlp.up_proj.weight                 [I, H]
    model.layers.i.mlp.down_proj.weight               [H, I]
    model.norm.weight                                 [H]
    lm_head.weight                                    [V, H]  (only when NOT tied)

Our module tree deliberately uses the same names, but the map is still built
explicitly from the config (not by trusting ``state_dict`` keys), so a renamed or
missing module is reported instead of silently skipped.

Rules: strict by default (missing / unexpected / shape mismatch -> error listing
the offending names), every parameter must be written exactly once (no random
weights survive), dtype conversion is explicit and recorded, tying is preserved.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from qwen3_scratch.config import Qwen3Config

__all__ = [
    "CheckpointError",
    "LoadReport",
    "SafetensorsCheckpoint",
    "build_name_map",
    "check_checkpoint",
    "expected_shapes",
    "load_into_model",
]


class CheckpointError(ValueError):
    """Checkpoint does not match the model (names, shapes, dtypes or structure)."""


_FLOAT_DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.float64)


# ------------------------------------------------------------------ name map
def _layer_names(i: int) -> list[str]:
    p = f"model.layers.{i}."
    return [
        p + "input_layernorm.weight",
        p + "self_attn.q_proj.weight",
        p + "self_attn.k_proj.weight",
        p + "self_attn.v_proj.weight",
        p + "self_attn.o_proj.weight",
        p + "self_attn.q_norm.weight",
        p + "self_attn.k_norm.weight",
        p + "post_attention_layernorm.weight",
        p + "mlp.gate_proj.weight",
        p + "mlp.up_proj.weight",
        p + "mlp.down_proj.weight",
    ]


def build_name_map(config: Qwen3Config) -> dict[str, str]:
    """Checkpoint tensor name -> our parameter name, for exactly this config.

    Only the configurations this project supports are mapped: no attention bias,
    QK-Norm on. ``lm_head.weight`` is included only for untied models; for tied
    models it is optional in the checkpoint (see :func:`load_into_model`).
    """
    if config.attention_bias:
        raise NotImplementedError("attention_bias=True checkpoints are not mapped (Qwen3: False)")
    if not config.qk_norm:
        raise NotImplementedError("qk_norm=False checkpoints are not mapped (Qwen3: True)")
    names = ["model.embed_tokens.weight"]
    for i in range(config.num_hidden_layers):
        names += _layer_names(i)
    names.append("model.norm.weight")
    if not config.tie_word_embeddings:
        names.append("lm_head.weight")
    return {n: n for n in names}  # identical names by design; kept explicit


def expected_shapes(config: Qwen3Config) -> dict[str, tuple[int, ...]]:
    """Shape of every mapped tensor, derived from the config (not from the model)."""
    h, inter, v, d = (
        config.hidden_size,
        config.intermediate_size,
        config.vocab_size,
        config.head_dim,
    )
    q, kv = config.q_proj_dim, config.kv_proj_dim
    per_layer = {
        "input_layernorm.weight": (h,),
        "self_attn.q_proj.weight": (q, h),
        "self_attn.k_proj.weight": (kv, h),
        "self_attn.v_proj.weight": (kv, h),
        "self_attn.o_proj.weight": (h, q),
        "self_attn.q_norm.weight": (d,),
        "self_attn.k_norm.weight": (d,),
        "post_attention_layernorm.weight": (h,),
        "mlp.gate_proj.weight": (inter, h),
        "mlp.up_proj.weight": (inter, h),
        "mlp.down_proj.weight": (h, inter),
    }
    shapes: dict[str, tuple[int, ...]] = {"model.embed_tokens.weight": (v, h)}
    for i in range(config.num_hidden_layers):
        for suffix, shape in per_layer.items():
            shapes[f"model.layers.{i}.{suffix}"] = shape
    shapes["model.norm.weight"] = (h,)
    if not config.tie_word_embeddings:
        shapes["lm_head.weight"] = (v, h)
    return shapes


# ------------------------------------------------------------- checkpoint IO
class SafetensorsCheckpoint(Mapping[str, torch.Tensor]):
    """Lazy, read-only view of a safetensors checkpoint (single file or sharded).

    Tensors are read on access (memory-mapped by ``safetensors``), so the whole
    checkpoint is never materialised at once. Only ``.safetensors`` is accepted:
    pickle-based ``.bin``/``.pt`` files can execute code when loaded.
    """

    def __init__(self, path: str | Path) -> None:
        from safetensors import safe_open  # optional dependency ([hf] extra)

        path = Path(path)
        if path.is_dir():
            index = path / "model.safetensors.index.json"
            single = path / "model.safetensors"
            if index.is_file():
                files = self._from_index(index)
            elif single.is_file():
                files = {None: single}
            else:
                pickles = sorted(p.name for p in path.glob("*.bin")) + sorted(
                    p.name for p in path.glob("*.pt")
                )
                hint = f" (found pickle files {pickles}; refusing to load them)" if pickles else ""
                raise CheckpointError(f"no model.safetensors(.index.json) in {path}{hint}")
        elif path.suffix == ".safetensors" and path.is_file():
            files = {None: path}
        else:
            raise CheckpointError(f"expected a .safetensors file or a model directory: {path}")

        self._handles = {}
        self._where: dict[str, Path] = {}
        for f in dict.fromkeys(files.values()):
            handle = safe_open(str(f), framework="pt", device="cpu")
            self._handles[f] = handle
            for name in handle.keys():  # noqa: SIM118 - safe_open is not a dict
                if name in self._where:
                    raise CheckpointError(f"tensor {name!r} appears in more than one shard")
                self._where[name] = f
        self.metadata = {str(f): self._handles[f].metadata() for f in self._handles}

    @staticmethod
    def _from_index(index: Path) -> dict[str, Path]:
        data = json.loads(index.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise CheckpointError(f"{index} has no weight_map")
        return {name: index.parent / shard for name, shard in weight_map.items()}

    def __getitem__(self, name: str) -> torch.Tensor:
        if name not in self._where:
            raise KeyError(name)
        return self._handles[self._where[name]].get_tensor(name)

    def shape_of(self, name: str) -> tuple[int, ...]:
        return tuple(self._handles[self._where[name]].get_slice(name).get_shape())

    def dtype_of(self, name: str) -> str:
        return self._handles[self._where[name]].get_slice(name).get_dtype()

    def __iter__(self) -> Iterator[str]:
        return iter(self._where)

    def __len__(self) -> int:
        return len(self._where)


# --------------------------------------------------------------------- load
@dataclass
class LoadReport:
    mapped: list[str] = field(default_factory=list)  # checkpoint names copied into the model
    missing: list[str] = field(default_factory=list)  # expected but absent
    unexpected: list[str] = field(default_factory=list)  # present but not in the map
    shape_mismatch: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=dict
    )  # name -> (checkpoint shape, expected shape)
    dtype_conversions: dict[str, str] = field(default_factory=dict)  # "bfloat16->float32": count
    tied: bool = False
    lm_head_in_checkpoint: bool = False
    num_parameters: int = 0  # unique parameters of the loaded model
    num_checkpoint_values: int = 0  # elements read from the checkpoint

    @property
    def ok(self) -> bool:
        return not (self.missing or self.unexpected or self.shape_mismatch)

    def summary(self) -> str:
        return (
            f"mapped={len(self.mapped)} missing={len(self.missing)} "
            f"unexpected={len(self.unexpected)} shape_mismatch={len(self.shape_mismatch)} "
            f"tied={self.tied} params={self.num_parameters:,} "
            f"checkpoint_values={self.num_checkpoint_values:,} conversions={self.dtype_conversions}"
        )


def _shape(ckpt: Mapping[str, torch.Tensor], name: str) -> tuple[int, ...]:
    if isinstance(ckpt, SafetensorsCheckpoint):
        return ckpt.shape_of(name)
    return tuple(ckpt[name].shape)


def check_checkpoint(ckpt: Mapping[str, torch.Tensor], config: Qwen3Config) -> LoadReport:
    """Compare names and shapes against the config WITHOUT reading tensor data."""
    if not isinstance(ckpt, Mapping):
        raise CheckpointError(f"checkpoint must be a mapping name -> tensor, got {type(ckpt)}")
    name_map = build_name_map(config)
    shapes = expected_shapes(config)
    report = LoadReport(tied=config.tie_word_embeddings)
    keys = list(ckpt.keys())
    if len(set(keys)) != len(keys):
        raise CheckpointError("checkpoint contains duplicate tensor names")
    if any(not isinstance(k, str) for k in keys):
        raise CheckpointError("checkpoint keys must be strings")
    report.lm_head_in_checkpoint = "lm_head.weight" in ckpt
    allowed_extra = {"lm_head.weight"} if config.tie_word_embeddings else set()
    report.missing = [n for n in name_map if n not in ckpt]
    report.unexpected = [k for k in keys if k not in name_map and k not in allowed_extra]
    for name in name_map:
        if name in ckpt:
            got = _shape(ckpt, name)
            if got != shapes[name]:
                report.shape_mismatch[name] = (got, shapes[name])
    if report.lm_head_in_checkpoint and config.tie_word_embeddings:
        got = _shape(ckpt, "lm_head.weight")
        if got != shapes["model.embed_tokens.weight"]:
            report.shape_mismatch["lm_head.weight"] = (got, shapes["model.embed_tokens.weight"])
    return report


def _format_problems(report: LoadReport, limit: int = 10) -> str:
    parts = []
    if report.missing:
        parts.append(f"missing {len(report.missing)}: {report.missing[:limit]}")
    if report.unexpected:
        parts.append(f"unexpected {len(report.unexpected)}: {report.unexpected[:limit]}")
    if report.shape_mismatch:
        items = list(report.shape_mismatch.items())[:limit]
        parts.append(
            f"shape mismatch {len(report.shape_mismatch)}: "
            + ", ".join(f"{n}: checkpoint {a} vs expected {b}" for n, (a, b) in items)
        )
    return "; ".join(parts)


@torch.no_grad()
def load_into_model(
    model: torch.nn.Module,
    ckpt: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> LoadReport:
    """Copy checkpoint tensors into ``model`` (a :class:`Qwen3ForCausalLM`) in place.

    Each tensor is read (memory-mapped for safetensors) and written with
    ``param.copy_(tensor)``, which converts the dtype element by element (bf16 ->
    fp32 is exact); conversions are counted in ``report.dtype_conversions``. No
    converted temporary is created, so peak memory ~= the model itself.

    Strict mode (default) raises :class:`CheckpointError` on any missing,
    unexpected or wrongly shaped tensor BEFORE touching the model. Non-strict mode
    still refuses shape mismatches and missing tensors (they would leave random
    weights behind); it only tolerates unexpected extra tensors.

    Tied models: ``lm_head.weight`` may be absent (Qwen3-0.6B) or present; if present
    it must equal ``model.embed_tokens.weight`` exactly, otherwise tying would
    silently discard trained weights.
    """
    config: Qwen3Config = model.config
    report = check_checkpoint(ckpt, config)
    if report.missing or report.shape_mismatch or (strict and report.unexpected):
        raise CheckpointError("checkpoint does not match the model: " + _format_problems(report))

    params = dict(model.named_parameters(remove_duplicate=False))
    name_map = build_name_map(config)
    written: set[int] = set()
    for ckpt_name, our_name in name_map.items():
        if our_name not in params:
            raise CheckpointError(f"model has no parameter {our_name!r} (mapped from {ckpt_name})")
        param = params[our_name]
        if id(param) in written:
            raise CheckpointError(f"parameter {our_name!r} would be assigned twice")
        tensor = ckpt[ckpt_name]  # memory-mapped view for safetensors checkpoints
        if tensor.dtype not in _FLOAT_DTYPES:
            raise CheckpointError(f"{ckpt_name} has non-floating dtype {tensor.dtype}")
        if tensor.dtype != param.dtype:
            src, dst = (str(d).removeprefix("torch.") for d in (tensor.dtype, param.dtype))
            key = f"{src}->{dst}"
            report.dtype_conversions[key] = report.dtype_conversions.get(key, 0) + 1
        # copy_ converts dtype element-wise while copying: no temporary converted
        # tensor (the 151936 x 1024 embedding would otherwise need +0.58 GiB in fp32).
        param.copy_(tensor)
        written.add(id(param))
        report.mapped.append(ckpt_name)
        report.num_checkpoint_values += tensor.numel()
        del tensor

    if config.tie_word_embeddings:
        if model.lm_head.weight is not model.model.embed_tokens.weight:
            raise CheckpointError("lm_head and embed_tokens are not tied in the model")
        if report.lm_head_in_checkpoint:
            head = ckpt["lm_head.weight"].to(model.lm_head.weight.dtype)
            if not torch.equal(head, model.model.embed_tokens.weight):
                raise CheckpointError(
                    "config ties embeddings but checkpoint lm_head.weight differs from "
                    "model.embed_tokens.weight"
                )

    unique = {id(p): p for p in model.parameters()}
    not_written = [n for n, p in params.items() if id(p) not in written]
    if not_written:  # tied lm_head shares the embedding's id, so it is never listed
        raise CheckpointError(f"parameters left uninitialised: {not_written[:10]}")
    report.num_parameters = sum(p.numel() for p in unique.values())
    return report
