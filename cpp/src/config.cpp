#include "qwen3/config.hpp"

#include <stdexcept>
#include <string>

namespace qwen3 {

namespace {
void require_positive(std::int64_t value, const char* name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive, got " +
                                std::to_string(value));
  }
}
}  // namespace

void Config::validate() const {
  require_positive(vocab_size, "vocab_size");
  require_positive(hidden_size, "hidden_size");
  require_positive(intermediate_size, "intermediate_size");
  require_positive(num_hidden_layers, "num_hidden_layers");
  require_positive(num_attention_heads, "num_attention_heads");
  require_positive(num_key_value_heads, "num_key_value_heads");
  require_positive(head_dim, "head_dim");
  require_positive(max_position_embeddings, "max_position_embeddings");
  if (num_attention_heads % num_key_value_heads != 0) {
    throw std::invalid_argument("num_attention_heads must be divisible by num_key_value_heads");
  }
  if (head_dim % 2 != 0) {
    throw std::invalid_argument("head_dim must be even for RoPE");
  }
  if (!(rope_theta > 0.0)) {
    throw std::invalid_argument("rope_theta must be > 0");
  }
  if (!(rms_norm_eps > 0.0)) {
    throw std::invalid_argument("rms_norm_eps must be > 0");
  }
}

std::int64_t Config::estimate_num_parameters(bool include_embeddings) const {
  const std::int64_t h = hidden_size;
  std::int64_t attn = h * q_proj_dim() + 2 * h * kv_proj_dim() + q_proj_dim() * h;
  if (attention_bias) {
    attn += q_proj_dim() + 2 * kv_proj_dim() + h;
  }
  const std::int64_t qk = qk_norm ? 2 * head_dim : 0;
  const std::int64_t mlp = 3 * h * intermediate_size;
  const std::int64_t block_norms = 2 * h;
  std::int64_t total = num_hidden_layers * (attn + qk + mlp + block_norms) + h;
  if (include_embeddings) {
    total += vocab_size * h;
    if (!tie_word_embeddings) {
      total += vocab_size * h;
    }
  }
  return total;
}

Config qwen3_0_6b_base() {
  Config c{};
  c.vocab_size = 151936;
  c.hidden_size = 1024;
  c.intermediate_size = 3072;
  c.num_hidden_layers = 28;
  c.num_attention_heads = 16;
  c.num_key_value_heads = 8;
  c.head_dim = 128;
  c.max_position_embeddings = 32768;
  c.rope_theta = 1000000.0;
  c.rms_norm_eps = 1e-6;
  c.tie_word_embeddings = true;
  c.attention_bias = false;
  c.qk_norm = true;
  return c;
}

Config tiny_qwen3() {
  Config c{};
  c.vocab_size = 512;
  c.hidden_size = 64;
  c.intermediate_size = 192;
  c.num_hidden_layers = 2;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 32;
  c.max_position_embeddings = 128;
  c.rope_theta = 10000.0;
  c.rms_norm_eps = 1e-6;
  c.tie_word_embeddings = true;
  c.attention_bias = false;
  c.qk_norm = true;
  return c;
}

}  // namespace qwen3
