#!/usr/bin/env bash
# Configure, build and test the C++20 scaffold with CMake + CTest.
#
#   bash scripts/build_cpp.sh            # Release build in build/cpp
#   BUILD_TYPE=Debug bash scripts/build_cpp.sh
#
# Fedora prerequisites (only when you start M3): sudo dnf install cmake gcc-c++ ninja-build
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build/cpp"
BUILD_TYPE="${BUILD_TYPE:-Release}"

GENERATOR_ARGS=()
if command -v ninja >/dev/null 2>&1; then
  GENERATOR_ARGS=(-G Ninja)
fi

cmake -S "${REPO_ROOT}/cpp" -B "${BUILD_DIR}" "${GENERATOR_ARGS[@]}" -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
cmake --build "${BUILD_DIR}" --parallel
ctest --test-dir "${BUILD_DIR}" --output-on-failure
