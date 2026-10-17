#include "qwen3/decoder.hpp"

#include "qwen3/ops.hpp"
#include "qwen3/profiler.hpp"

DecoderLayer::DecoderLayer(qwen3::Config c)
    : input_layernorm(static_cast<int>(c.hidden_size), static_cast<float>(c.rms_norm_eps)),
      self_attn(static_cast<int>(c.hidden_size), static_cast<int>(c.num_attention_heads),
                static_cast<int>(c.num_key_value_heads), static_cast<int>(c.head_dim),
                static_cast<float>(c.rope_theta), static_cast<float>(c.rms_norm_eps)),
      post_attention_layernorm(static_cast<int>(c.hidden_size), static_cast<float>(c.rms_norm_eps)),
      mlp(static_cast<int>(c.hidden_size), static_cast<int>(c.intermediate_size)) {}

Tensor DecoderLayer::forward(Tensor& x, KVCache* cache, int layer) {
    Tensor a, m;
    {
        ScopedTimer timer("layer.norm");
        a = input_layernorm.forward(x);
    }
    {
        ScopedTimer timer("layer.attention");
        a = self_attn.forward(a, cache, layer);
    }
    Tensor h = add(x, a);                       // residual 1
    {
        ScopedTimer timer("layer.norm");
        m = post_attention_layernorm.forward(h);
    }
    {
        ScopedTimer timer("layer.mlp");
        m = mlp.forward(m);
    }
    return add(h, m);                           // residual 2
}
