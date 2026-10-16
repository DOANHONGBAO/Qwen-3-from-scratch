"""M2 pipeline parity with Hugging Face on SMALL random checkpoints (no download).

1. A tiny HF ``Qwen3ForCausalLM`` is saved with ``save_pretrained`` (safetensors +
   config.json written by transformers itself); OUR loader reads that directory.
2. Cached decoding (prefill + token-by-token) is compared with HF's own cache.
3. Greedy token ids are compared with ``transformers``' ``generate`` (reference only).

The real Qwen3-0.6B-Base comparison is ``scripts/verify_pretrained.py`` (weights tier).
"""

from __future__ import annotations

import pytest
import torch
from torch.testing import assert_close

modeling = pytest.importorskip(
    "transformers.models.qwen3.modeling_qwen3", reason="transformers not installed (hf extra)"
)
transformers = pytest.importorskip("transformers")

from qwen3_scratch.config import QWEN3_TINY, TINY_QWEN3, Qwen3Config  # noqa: E402
from qwen3_scratch.generation import generate  # noqa: E402
from qwen3_scratch.pretrained import load_pretrained  # noqa: E402

EXACT = {"atol": 0.0, "rtol": 0.0}
# cached vs. full-sequence matmuls may round differently in float32 (see test_kv_cache.py)
F32 = {"atol": 1e-5, "rtol": 1e-5}


def _hf_config(c: Qwen3Config):
    return transformers.Qwen3Config(
        vocab_size=c.vocab_size,
        hidden_size=c.hidden_size,
        intermediate_size=c.intermediate_size,
        num_hidden_layers=c.num_hidden_layers,
        num_attention_heads=c.num_attention_heads,
        num_key_value_heads=c.num_key_value_heads,
        head_dim=c.head_dim,
        max_position_embeddings=c.max_position_embeddings,
        rope_theta=c.rope_theta,
        rms_norm_eps=c.rms_norm_eps,
        tie_word_embeddings=c.tie_word_embeddings,
        attn_implementation="eager",
    )


def _hf_model(c: Qwen3Config, seed=0):
    torch.manual_seed(seed)
    hf = modeling.Qwen3ForCausalLM(_hf_config(c)).eval()
    with torch.no_grad():
        for name, p in hf.named_parameters():
            if "norm" in name:
                p.add_(0.1 * torch.randn(p.shape))
            else:
                p.mul_(5.0)  # sharper distributions: greedy ties become unlikely
    return hf


@pytest.fixture(scope="module", params=[QWEN3_TINY, TINY_QWEN3], ids=["qwen3_tiny", "tiny_gqa"])
def saved(request, tmp_path_factory):
    c = request.param
    hf = _hf_model(c)
    path = tmp_path_factory.mktemp("hf_ckpt")
    hf.save_pretrained(path, safe_serialization=True)
    return c, hf, path


def _ids(c, b=2, t=9, seed=1):
    return torch.randint(0, c.vocab_size, (b, t), generator=torch.Generator().manual_seed(seed))


def test_loader_reads_transformers_checkpoint(saved) -> None:
    c, hf, path = saved
    assert (path / "model.safetensors").is_file()
    ours, report = load_pretrained(path, dtype=torch.float32)
    assert report.ok and report.tied == c.tie_word_embeddings
    assert ours.config.hidden_size == c.hidden_size and ours.config.rope_theta == c.rope_theta
    for name, p in hf.state_dict().items():
        assert torch.equal(ours.state_dict()[name], p), name
    ids = _ids(c)
    with torch.no_grad():
        assert_close(ours(ids).logits, hf(ids).logits, **EXACT)


def test_reloaded_hf_and_ours_agree(saved) -> None:
    c, _, path = saved
    hf = modeling.Qwen3ForCausalLM.from_pretrained(
        path, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    ours, _ = load_pretrained(path)
    ids = _ids(c, seed=4)
    with torch.no_grad():
        assert_close(ours(ids).logits, hf(ids).logits, **EXACT)


def test_cached_decoding_matches_hf_cache(saved) -> None:
    c, hf, path = saved
    ours, _ = load_pretrained(path)
    ids = _ids(c, t=10)
    with torch.no_grad():
        hf_cache = transformers.DynamicCache(config=hf.config)
        cache = ours.new_cache()
        for start, end in [(0, 6), (6, 7), (7, 8), (8, 10)]:
            chunk = ids[:, start:end]
            h = hf(chunk, past_key_values=hf_cache, use_cache=True).logits
            o = ours(chunk, past_key_values=cache, use_cache=True).logits
            assert_close(o, h, **F32)
            assert cache.seq_len == end
        k_hf = hf_cache.layers[0].keys
        assert_close(cache.get(0)[0], k_hf, **F32)  # same post-RoPE keys, native heads


def test_greedy_tokens_match_hf_generate(saved) -> None:
    c, hf, path = saved
    ours, _ = load_pretrained(path)
    prompt = _ids(c, b=1, t=5, seed=7)
    ref = hf.generate(
        prompt,
        attention_mask=torch.ones_like(prompt),
        max_new_tokens=12,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=None,
    )
    for use_cache in (True, False):
        out = generate(ours, prompt, max_new_tokens=12, use_cache=use_cache)
        assert torch.equal(out.sequences, ref), use_cache
