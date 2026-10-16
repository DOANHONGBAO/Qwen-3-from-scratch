#!/usr/bin/env bash
# Render a Kaggle kernel folder from a template WITHOUT pushing it.
#
#   KAGGLE_USERNAME=<your-username> bash kaggle/prepare_kernel.sh smoke_test
#
# Output goes to build/kaggle/<name>/ (git-ignored). Your Kaggle username is not a
# secret, but no credential is read or written by this script.
# To actually run it on a GPU (a deliberate, manual step):
#   kaggle kernels push -p build/kaggle/<name> --accelerator NvidiaTeslaT4
#   kaggle kernels status <username>/qwen3-scratch-gpu-smoke-test
#   kaggle kernels output <username>/qwen3-scratch-gpu-smoke-test -p build/kaggle/<name>/output
set -euo pipefail

NAME="${1:-smoke_test}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="${REPO_ROOT}/kaggle/${NAME}"
DST="${REPO_ROOT}/build/kaggle/${NAME}"

if [[ -z "${KAGGLE_USERNAME:-}" ]]; then
  echo "ERROR: set KAGGLE_USERNAME to your Kaggle username (not your key)." >&2
  exit 1
fi
if [[ ! -f "${SRC}/kernel-metadata.template.json" ]]; then
  echo "ERROR: ${SRC}/kernel-metadata.template.json not found" >&2
  exit 1
fi

mkdir -p "${DST}"
find "${SRC}" -maxdepth 1 -type f ! -name 'kernel-metadata.template.json' -exec cp {} "${DST}/" \;
sed "s/KAGGLE_USERNAME/${KAGGLE_USERNAME}/" "${SRC}/kernel-metadata.template.json" \
  > "${DST}/kernel-metadata.json"
# Pin the source version for kernels that clone the repository (qwen3_inference).
COMMIT="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
for f in "${DST}"/*.py; do
  sed -i "s/COMMIT_SHA/${COMMIT}/" "$f"
done
if ! git -C "${REPO_ROOT}" branch -r --contains "${COMMIT}" | grep -q .; then
  echo "WARNING: ${COMMIT} is not on any remote branch; push it before running the kernel." >&2
fi
echo "Prepared ${DST} (not pushed)."
