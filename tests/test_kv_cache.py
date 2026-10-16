"""KV cache tests (M2.2). CPU tier, Tiny Qwen3 configs, seconds to run.

Tolerances. Cached and uncached forwards execute the same operations on the same
values, but matmuls with different shapes ([B, Nq, 1, T] vs [B, Nq, T, T]) may
sum in a different order. In float64 we therefore require agreement to 1e-12
(rounding only); in float32 to ``atol=1e-5, rtol=1e-5`` (logits of the tiny model
are O(1), float32 eps = 1.2e-7, 2 layers). A real bug (wrong position, missing or
duplicated key, leaked future token) produces errors of order 1e-2 .. 1.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch
from torch.testing import assert_close

from qwen3_scratch.config import QWEN3_TINY, TINY_QWEN3
from qwen3_scratch.kv_cache import KVCache, kv_cache_bytes
from qwen3_scratch.layers.attention import GroupedQueryAttention
from qwen3_scratch.layers.rope import RotaryEmbedding
from qwen3_scratch.model import Qwen3ForCausalLM

F64 = {"atol": 1e-12, "rtol": 1e-12}
F32 = {"atol": 1e-5, "rtol": 1e-5}


def _model(cfg=QWEN3_TINY, dtype=torch.float64, seed=0):
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(cfg, dtype=dtype).eval()


def _ids(b, t, vocab, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, vocab, (b, t), generator=g)


# ----------------------------------------------------------- KVCache unit
def _kv(b=2, nkv=2, t=3, d=4, dtype=torch.float32, fill=0.0):
    return torch.full((b, nkv, t, d), fill, dtype=dtype), torch.full(
        (b, nkv, t, d), -fill, dtype=dtype
    )


def test_init_and_empty_state() -> None:
    c = KVCache(3, max_length=10)
    assert c.seq_len == 0 and c.is_empty and c.nbytes() == 0 and c.batch_size is None
    assert [c.layer_length(i) for i in range(3)] == [0, 0, 0]
    with pytest.raises(ValueError):
        KVCache(0)
    with pytest.raises(ValueError):
        KVCache(2, max_length=0)


def test_update_appends_and_commit_tracks_length() -> None:
    c = KVCache(2)
    for layer in range(2):
        k, v = c.update(layer, *_kv(t=3, fill=1.0))
        assert k.shape == (2, 2, 3, 4)
    c.commit(3)
    assert c.seq_len == 3
    for layer in range(2):
        k, v = c.update(layer, *_kv(t=1, fill=2.0))
        assert k.shape == (2, 2, 4, 4)
        assert torch.all(k[:, :, :3] == 1.0) and torch.all(k[:, :, 3] == 2.0)  # order kept
        assert torch.all(v[:, :, 3] == -2.0)
    c.commit(1)
    assert c.seq_len == 4 and c.nbytes() == 2 * 2 * (2 * 2 * 4 * 4) * 4


def test_layers_are_isolated() -> None:
    c = KVCache(2)
    c.update(0, *_kv(fill=1.0))
    c.update(1, *_kv(fill=5.0))
    assert torch.all(c.get(0)[0] == 1.0) and torch.all(c.get(1)[0] == 5.0)


def test_double_append_and_bad_commit_are_rejected() -> None:
    c = KVCache(2)
    c.update(0, *_kv())
    with pytest.raises(RuntimeError, match="exactly once"):
        c.update(0, *_kv())  # same layer twice in one forward
    with pytest.raises(RuntimeError, match="cannot commit"):
        c.commit(3)  # layer 1 never appended


def test_rollback_drops_uncommitted() -> None:
    c = KVCache(2)
    for i in range(2):
        c.update(i, *_kv(t=2))
    c.commit(2)
    c.update(0, *_kv(t=1))
    c.rollback()
    assert [c.layer_length(i) for i in range(2)] == [2, 2] and c.seq_len == 2
    c2 = KVCache(1)
    c2.update(0, *_kv(t=2))
    c2.rollback()
    assert c2.is_empty


def test_reset_allows_new_session() -> None:
    c = KVCache(1)
    c.update(0, *_kv(b=2))
    c.commit(3)
    c.reset()
    assert c.is_empty
    c.update(0, *_kv(b=5, dtype=torch.float64))  # different batch / dtype is fine now
    assert c.batch_size == 5


@pytest.mark.parametrize(
    ("k_shape", "match"),
    [((3, 2, 1, 4), "batch"), ((2, 4, 1, 4), "incompatible"), ((2, 2, 1, 8), "incompatible")],
)
def test_incompatible_appends(k_shape, match) -> None:
    c = KVCache(1)
    c.update(0, *_kv())
    c.commit(3)
    k = torch.zeros(k_shape)
    with pytest.raises(ValueError, match=match):
        c.update(0, k, k.clone())


def test_dtype_and_shape_validation() -> None:
    c = KVCache(1)
    c.update(0, *_kv())
    c.commit(3)
    k, v = _kv(t=1, dtype=torch.float64)
    with pytest.raises(ValueError, match="cache holds"):
        c.update(0, k, v.double())
    with pytest.raises(ValueError, match=r"\[B, Nkv, T, D\]"):
        c.update(0, torch.zeros(2, 2, 4), torch.zeros(2, 2, 4))
    with pytest.raises(ValueError, match="differ"):
        c.update(0, torch.zeros(2, 2, 1, 4), torch.zeros(2, 2, 2, 4))
    with pytest.raises(ValueError, match="zero tokens"):
        c.update(0, torch.zeros(2, 2, 0, 4), torch.zeros(2, 2, 0, 4))
    with pytest.raises(IndexError):
        c.update(1, *_kv(t=1))
    with pytest.raises(TypeError):
        c.update(True, *_kv(t=1))


def test_overflow_raises() -> None:
    c = KVCache(1, max_length=4)
    c.update(0, *_kv(t=3))
    c.commit(3)
    with pytest.raises(ValueError, match="overflow"):
        c.update(0, *_kv(t=2))


def test_analytic_size_matches_real_cache() -> None:
    m = _model(dtype=torch.float32)
    out = m(_ids(2, 9, QWEN3_TINY.vocab_size), use_cache=True)
    assert out.past_key_values.nbytes() == kv_cache_bytes(
        QWEN3_TINY, 9, batch_size=2, bytes_per_element=4
    )


def test_qwen3_0_6b_cache_size_per_token() -> None:
    from qwen3_scratch.config import QWEN3_0_6B_BASE

    assert kv_cache_bytes(QWEN3_0_6B_BASE, 1, bytes_per_element=2) == 114_688  # 112 KiB


# -------------------------------------------------- attention-level checks
def test_attention_stores_native_kv_heads_post_rope() -> None:
    torch.manual_seed(0)
    cfg = TINY_QWEN3
    attn = GroupedQueryAttention.from_config(cfg, dtype=torch.float64)
    rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
    x = torch.randn(1, 5, cfg.hidden_size, dtype=torch.float64)
    cos, sin = rope(torch.arange(5)[None], dtype=torch.float64)
    cache = KVCache(1)
    _, trace = attn(x, cos, sin, kv_cache=cache, layer_idx=0, return_trace=True)
    k_cached, v_cached = cache.get(0)
    assert k_cached.shape == (1, cfg.num_key_value_heads, 5, cfg.head_dim)  # Nkv, not Nq
    assert torch.equal(k_cached, trace.k_rope)  # post-QK-Norm, post-RoPE keys
    v_ref = trace.v_proj.view(1, 5, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
    assert torch.equal(v_cached, v_ref)  # values: no norm, no rotation


def test_attention_cached_step_matches_full() -> None:
    torch.manual_seed(0)
    cfg = TINY_QWEN3
    attn = GroupedQueryAttention.from_config(cfg, dtype=torch.float64)
    rope = RotaryEmbedding(cfg.head_dim, cfg.rope_theta)
    x = torch.randn(2, 6, cfg.hidden_size, dtype=torch.float64)
    cos, sin = rope(torch.arange(6)[None], dtype=torch.float64)
    full = attn(x, cos, sin)
    cache = KVCache(1)
    pre = attn(x[:, :4], cos[:, :4], sin[:, :4], kv_cache=cache, layer_idx=0)
    cache.commit(4)
    step = attn(x[:, 4:6], cos[:, 4:6], sin[:, 4:6], kv_cache=cache, layer_idx=0)
    assert_close(torch.cat([pre, step], 1), full, **F64)


def test_attention_requires_layer_idx_with_cache() -> None:
    attn = GroupedQueryAttention.from_config(TINY_QWEN3)
    cos, sin = RotaryEmbedding(32)(torch.arange(2)[None])
    with pytest.raises(ValueError, match="layer_idx"):
        attn(torch.randn(1, 2, 64), cos, sin, kv_cache=KVCache(1))


# ------------------------------------------------------ model-level checks
@pytest.mark.parametrize("cfg", [QWEN3_TINY, TINY_QWEN3], ids=["qwen3_tiny", "tiny_gqa"])
def test_full_forward_equals_prefill_plus_token_by_token(cfg) -> None:
    """MANDATORY: logits of one full pass == prefill + cached single-token decoding."""
    m = _model(cfg)
    ids = _ids(2, 12, cfg.vocab_size)
    with torch.no_grad():
        full = m(ids).logits  # [2, 12, V]
        cache = m.new_cache()
        parts = [m(ids[:, :5], past_key_values=cache, use_cache=True).logits]
        assert cache.seq_len == 5
        for t in range(5, 12):
            out = m(ids[:, t : t + 1], past_key_values=cache, use_cache=True)
            assert out.past_key_values is cache and cache.seq_len == t + 1
            parts.append(out.logits)
    assert_close(torch.cat(parts, 1), full, **F64)


def test_full_forward_equals_cached_float32() -> None:
    m = _model(dtype=torch.float32)
    ids = _ids(1, 10, QWEN3_TINY.vocab_size)
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_cache()
        parts = [m(ids[:, :1], past_key_values=cache, use_cache=True).logits]  # 1-token prefill
        parts += [
            m(ids[:, t : t + 1], past_key_values=cache, use_cache=True).logits for t in range(1, 10)
        ]
    assert_close(torch.cat(parts, 1), full, **F32)


@pytest.mark.parametrize("chunks", [(4, 3, 5), (1, 11), (6, 6), (2, 2, 2, 2, 2, 2)])
def test_chunked_decoding_matches_full(chunks) -> None:
    m = _model()
    ids = _ids(2, sum(chunks), QWEN3_TINY.vocab_size, seed=3)
    with torch.no_grad():
        full = m(ids).logits
        cache = m.new_cache()
        parts, start = [], 0
        for n in chunks:
            parts.append(m(ids[:, start : start + n], past_key_values=cache, use_cache=True).logits)
            start += n
    assert cache.seq_len == sum(chunks)
    assert_close(torch.cat(parts, 1), full, **F64)


def test_no_future_leakage_inside_a_chunk() -> None:
    """Changing a later token of a chunk must not change earlier logits of that chunk."""
    m = _model()
    ids = _ids(1, 9, QWEN3_TINY.vocab_size)
    alt = ids.clone()
    alt[0, 8] = (alt[0, 8] + 1) % QWEN3_TINY.vocab_size
    with torch.no_grad():
        outs = []
        for x in (ids, alt):
            cache = m.new_cache()
            m(x[:, :4], past_key_values=cache, use_cache=True)
            outs.append(m(x[:, 4:], past_key_values=cache, use_cache=True).logits)  # chunk 4..8
    assert torch.equal(outs[0][:, :4], outs[1][:, :4])  # positions 4..7 unaffected
    assert not torch.equal(outs[0][:, 4], outs[1][:, 4])


def test_cached_decode_uses_nonzero_positions() -> None:
    """A single cached token at index 7 must use RoPE position 7, not 0."""
    m = _model()
    ids = _ids(1, 8, QWEN3_TINY.vocab_size)
    with torch.no_grad():
        cache = m.new_cache()
        m(ids[:, :7], past_key_values=cache, use_cache=True)
        right = m(ids[:, 7:], past_key_values=cache, use_cache=True).logits
        cache2 = m.new_cache()
        m(ids[:, :7], past_key_values=cache2, use_cache=True)
        wrong = m(
            ids[:, 7:], position_ids=torch.tensor([[0]]), past_key_values=cache2, use_cache=True
        ).logits
        full = m(ids).logits[:, 7:]
    assert_close(right, full, **F64)
    assert (wrong - full).abs().max() > 1e-3  # position 0 gives a different answer


def test_cache_does_not_reprocess_old_tokens() -> None:
    m = _model()
    cache = m.new_cache()
    with torch.no_grad():
        m(_ids(1, 6, 256), past_key_values=cache, use_cache=True)
        m(_ids(1, 1, 256), past_key_values=cache, use_cache=True)
    assert [cache.layer_length(i) for i in range(cache.num_layers)] == [7, 7]


def test_reset_and_reuse_gives_fresh_session() -> None:
    m = _model()
    a, b = _ids(1, 5, 256, seed=1), _ids(1, 5, 256, seed=2)
    with torch.no_grad():
        cache = m.new_cache()
        m(a, past_key_values=cache, use_cache=True)
        cache.reset()
        reused = m(b, past_key_values=cache, use_cache=True).logits
        fresh = m(b).logits
    assert torch.equal(reused, fresh) or torch.allclose(reused, fresh, **F64)


def test_use_cache_false_does_not_touch_external_cache() -> None:
    m = _model()
    cache = m.new_cache()
    with torch.no_grad():
        out = m(_ids(1, 4, 256))
    assert out.past_key_values is None and cache.is_empty
    with pytest.raises(ValueError, match="use_cache=False"):
        m(_ids(1, 4, 256), past_key_values=cache)


def test_use_cache_without_cache_creates_one() -> None:
    m = _model()
    with torch.no_grad():
        out = m(_ids(1, 4, 256), use_cache=True)
    assert isinstance(out.past_key_values, KVCache) and out.past_key_values.seq_len == 4


def test_context_overflow_with_cache() -> None:
    cfg = dataclasses.replace(QWEN3_TINY, max_position_embeddings=8)
    m = _model(cfg)
    cache = m.new_cache()
    with torch.no_grad():
        m(_ids(1, 6, 256), past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="overflow"):
            m(_ids(1, 3, 256), past_key_values=cache, use_cache=True)
    assert cache.seq_len == 6 and cache.layer_length(1) == 6  # untouched on failure


def test_batch_mismatch_with_cache() -> None:
    m = _model()
    cache = m.new_cache()
    with torch.no_grad():
        m(_ids(2, 3, 256), past_key_values=cache, use_cache=True)
        with pytest.raises(ValueError, match="batch"):
            m(_ids(3, 1, 256), past_key_values=cache, use_cache=True)


def test_failed_forward_rolls_back(monkeypatch) -> None:
    m = _model()
    cache = m.new_cache()
    with torch.no_grad():
        m(_ids(1, 3, 256), past_key_values=cache, use_cache=True)

        def boom(*a, **k):
            raise RuntimeError("boom")

        monkeypatch.setattr(m.model.layers[1].mlp, "forward", boom)
        with pytest.raises(RuntimeError, match="boom"):
            m(_ids(1, 2, 256), past_key_values=cache, use_cache=True)
    assert [cache.layer_length(i) for i in range(2)] == [3, 3] and cache.seq_len == 3


def test_padding_mask_with_cache_is_rejected_but_all_ones_ok() -> None:
    m = _model()
    ids = _ids(2, 4, 256)
    with torch.no_grad():
        ok = m(ids, attention_mask=torch.ones(2, 4, dtype=torch.long), use_cache=True)
        assert ok.past_key_values.seq_len == 4
        mask = torch.tensor([[0, 1, 1, 1], [1, 1, 1, 1]])
        with pytest.raises(NotImplementedError, match="padding"):
            m(ids, attention_mask=mask, use_cache=True)


def test_cache_is_gradient_free_under_inference_mode() -> None:
    m = _model(dtype=torch.float32)
    with torch.inference_mode():
        out = m(_ids(1, 3, 256), use_cache=True)
    k, v = out.past_key_values.get(0)
    assert not k.requires_grad and not v.requires_grad


def test_wrong_cache_type_and_layer_count() -> None:
    m = _model()
    with pytest.raises(TypeError):
        m(_ids(1, 2, 256), past_key_values={}, use_cache=True)
    with pytest.raises(ValueError, match="layers"):
        m(_ids(1, 2, 256), past_key_values=KVCache(5), use_cache=True)


def test_logits_to_keep() -> None:
    m = _model()
    ids = _ids(1, 6, 256)
    with torch.no_grad():
        full = m(ids).logits
        last = m(ids, logits_to_keep=1).logits
        two = m(ids, logits_to_keep=2).logits
    assert last.shape == (1, 1, 256) and two.shape == (1, 2, 256)
    assert_close(last, full[:, -1:], **F64)
    with pytest.raises(ValueError):
        m(ids, logits_to_keep=-1)
    with pytest.raises(ValueError, match="labels"):
        m(ids, labels=ids, logits_to_keep=1)


def test_cpp_fixture_decode_rows_match_full_forward(repo_root) -> None:
    """The C++ contract: cached decode logits == matching rows of the uncached forward."""
    import json

    data = json.loads((repo_root / "tests" / "fixtures" / "kv_cache_tiny.json").read_text())
    out = data["outputs"]
    full = torch.tensor(out["full_forward_logits"]["data"])[0]  # [T, V]
    n_prompt = len(data["inputs"]["prompt_ids"][0])
    assert_close(torch.tensor(out["prefill_logits"]["data"])[0], full[:n_prompt], **F32)
    for step in range(len(data["inputs"]["decode_ids"])):
        row = torch.tensor(out[f"decode_logits_step{step}"]["data"])[0, 0]
        assert_close(row, full[n_prompt + step], **F32)
    k = out["final_key_layer0"]
    assert k["shape"] == [1, data["config"]["num_key_value_heads"], n_prompt + 3, 4]
