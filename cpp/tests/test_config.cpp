// Dependency-free CTest smoke test for the C++ config mirror.
// The expected numbers are identical to tests/test_config.py (Python), which is
// the first cross-language parity check of the project.
#include <cstdlib>
#include <iostream>
#include <stdexcept>

#include "qwen3/config.hpp"

namespace {
int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::cerr << "FAILED: " << what << '\n';
    ++failures;
  }
}

template <typename Fn>
void check_throws(Fn&& fn, const char* what) {
  try {
    fn();
  } catch (const std::invalid_argument&) {
    return;
  }
  std::cerr << "FAILED (no exception): " << what << '\n';
  ++failures;
}
}  // namespace

int main() {
  const auto big = qwen3::qwen3_0_6b_base();
  big.validate();
  check(big.estimate_num_parameters() == 596049920, "0.6B total parameter count");
  check(big.estimate_num_parameters(false) == 440467456, "0.6B non-embedding parameter count");
  check(big.q_proj_dim() == 2048, "0.6B q_proj_dim");
  check(big.kv_proj_dim() == 1024, "0.6B kv_proj_dim");
  check(big.num_queries_per_kv() == 2, "0.6B GQA group size");

  const auto tiny = qwen3::tiny_qwen3();
  tiny.validate();
  check(tiny.num_queries_per_kv() == 2, "tiny GQA group size");

  check_throws(
      [] {
        auto c = qwen3::tiny_qwen3();
        c.num_key_value_heads = 3;
        c.validate();
      },
      "heads not divisible by kv heads");
  check_throws(
      [] {
        auto c = qwen3::tiny_qwen3();
        c.head_dim = 31;
        c.validate();
      },
      "odd head_dim");

  if (failures == 0) {
    std::cout << "qwen3 C++ config tests PASSED (version " << qwen3::kVersion << ")\n";
    return EXIT_SUCCESS;
  }
  return EXIT_FAILURE;
}
