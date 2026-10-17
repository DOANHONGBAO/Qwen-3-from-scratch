#pragma once

#include "qwen3/attention.hpp"
#include "qwen3/config.hpp"
#include "qwen3/rmsnorm.hpp"
#include "qwen3/swiglu.hpp"

// Mot block Transformer cua Qwen3 (pre-norm, 2 residual):
//   h   = x + self_attn( input_layernorm(x) )
//   out = h + mlp( post_attention_layernorm(h) )
class DecoderLayer {
public:
    DecoderLayer(qwen3::Config config);

    Tensor forward(Tensor& x, KVCache* cache, int layer);

    RMSNorm input_layernorm;
    Attention self_attn;
    RMSNorm post_attention_layernorm;
    SwiGLU mlp;
};
