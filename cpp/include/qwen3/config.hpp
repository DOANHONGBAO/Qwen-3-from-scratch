// C++ mirror of qwen3_scratch.config.Qwen3Config (Python).
//
// Field names intentionally match the Hugging Face config.json keys so that the
// Python and C++ implementations share one vocabulary. Any change here must be
// mirrored in src/qwen3_scratch/config.py and covered by tests on both sides.
#pragma once

#include <cstdint>
#include <string_view>

namespace qwen3 {

inline constexpr std::string_view kVersion = "0.0.1";

struct Config {
  std::int64_t vocab_size;
  std::int64_t hidden_size;
  std::int64_t intermediate_size;
  std::int64_t num_hidden_layers;
  std::int64_t num_attention_heads;
  std::int64_t num_key_value_heads;
  std::int64_t head_dim;
  std::int64_t max_position_embeddings;
  double rope_theta;
  double rms_norm_eps;
  bool tie_word_embeddings;
  bool attention_bias = false;
  bool qk_norm = true;

  // Throws std::invalid_argument with a descriptive message if inconsistent.
  void validate() const;

  [[nodiscard]] std::int64_t num_queries_per_kv() const {
    return num_attention_heads / num_key_value_heads;
  }
  [[nodiscard]] std::int64_t q_proj_dim() const { return num_attention_heads * head_dim; }
  [[nodiscard]] std::int64_t kv_proj_dim() const { return num_key_value_heads * head_dim; }

  // Must return exactly the same value as Qwen3Config.estimate_num_parameters().
  [[nodiscard]] std::int64_t estimate_num_parameters(bool include_embeddings = true) const;
};

// Official Qwen/Qwen3-0.6B-Base hyperparameters.
Config qwen3_0_6b_base();
// Tiny configuration used by fast unit tests (same values as TINY_QWEN3 in Python).
Config tiny_qwen3();

}  // namespace qwen3
