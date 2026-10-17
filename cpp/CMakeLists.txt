cmake_minimum_required(VERSION 3.20)

project(qwen3_cpp
  VERSION 0.0.1
  DESCRIPTION "C++20 port of the verified qwen3_scratch PyTorch model (M3+)"
  LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(QWEN3_BUILD_TESTS "Build C++ unit tests" ON)

# Library: header-first design. Model kernels (RMSNorm, RoPE, GQA, ...) are added
# in M3, one module per Python counterpart (see docs/architecture.md).
add_library(qwen3 
src/config.cpp src/tensor.cpp src/ops.cpp src/rmsnorm.cpp src/swiglu.cpp
src/rope.cpp
src/json.cpp src/safetensors.cpp src/kv_cache.cpp src/attention.cpp
src/decoder.cpp src/model.cpp src/generate.cpp src/profiler.cpp
)
target_include_directories(qwen3 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(qwen3 PUBLIC cxx_std_20)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(qwen3 PRIVATE -Wall -Wextra -Wpedantic -Wconversion)
endif()

# Multi-threading (M4): OpenMP splits linear and attention across CPU cores. It ships
# with GCC (libgomp); without it everything still builds and runs on one thread.
option(QWEN3_OPENMP "Use OpenMP threads if available" ON)
if(QWEN3_OPENMP)
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(qwen3 PUBLIC OpenMP::OpenMP_CXX)
  endif()
endif()

# -march=native: let the compiler use every SIMD instruction of THIS CPU (AVX2/FMA on the
# maintainer's Ryzen). Off by default so CI and other machines get portable binaries;
# scripts/benchmark_cpp.py builds a separate native copy.
option(QWEN3_NATIVE "Optimise for the build machine (-march=native)" OFF)
if(QWEN3_NATIVE AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(qwen3 PRIVATE -march=native)
endif()

# Command-line tool: load a Hugging Face checkpoint and run greedy generation.
add_executable(qwen3_run tools/qwen3_run.cpp)
target_link_libraries(qwen3_run PRIVATE qwen3)
add_executable(qwen3_bench tools/qwen3_bench.cpp)
target_link_libraries(qwen3_bench PRIVATE qwen3)

if(QWEN3_BUILD_TESTS)
  enable_testing()
  # Dependency-free test executable registered with CTest. A test framework
  # (e.g. GoogleTest or Catch2) can be introduced in M3 if it earns its keep.
  add_executable(test_config tests/test_config.cpp)
  target_link_libraries(test_config PRIVATE qwen3)
  add_test(NAME config.presets COMMAND test_config)

  add_executable(test_tensor tests/test_tensor.cpp)
  target_link_libraries(test_tensor PRIVATE qwen3)
  add_test(NAME tensor.basic COMMAND test_tensor)

  add_executable(test_ops tests/test_ops.cpp)
  target_link_libraries(test_ops PRIVATE qwen3)
  add_test(NAME ops.reference COMMAND test_ops)

  add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
  target_link_libraries(test_rmsnorm PRIVATE qwen3)
  add_test(NAME rmsnorm.reference COMMAND test_rmsnorm)

  add_executable(test_swiglu tests/test_swiglu.cpp)
  target_link_libraries(test_swiglu PRIVATE qwen3)
  add_test(NAME swiglu.reference COMMAND test_swiglu)

  add_executable(test_rope tests/test_rope.cpp)
  target_link_libraries(test_rope PRIVATE qwen3)
  add_test(NAME rope.reference COMMAND test_rope)

    add_executable(test_kv_cache tests/test_kv_cache.cpp)
  target_link_libraries(test_kv_cache PRIVATE qwen3)
  add_test(NAME kv_cache.rules COMMAND test_kv_cache)

  add_executable(test_attention tests/test_attention.cpp)
  target_link_libraries(test_attention PRIVATE qwen3)
  add_test(NAME attention.reference COMMAND test_attention)

  add_executable(test_safetensors tests/test_safetensors.cpp)
  target_link_libraries(test_safetensors PRIVATE qwen3)
  add_test(NAME safetensors.reader COMMAND test_safetensors)

  add_executable(test_model tests/test_model.cpp)
  target_link_libraries(test_model PRIVATE qwen3)
  # Fixture of M2: names and shapes of every tensor in the real Qwen3-0.6B checkpoint
  target_compile_definitions(test_model PRIVATE
    QWEN3_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/../tests/fixtures")
  add_test(NAME model.reference COMMAND test_model)

  add_executable(test_threads tests/test_threads.cpp)
  target_link_libraries(test_threads PRIVATE qwen3)
  add_test(NAME threads.determinism COMMAND test_threads)

endif()
