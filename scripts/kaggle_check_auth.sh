#!/usr/bin/env bash
# Check whether Kaggle CLI authentication is configured WITHOUT printing secrets.
#
#   bash scripts/kaggle_check_auth.sh
#
# It only reports which mechanism is present (env var / token file / legacy json)
# and whether file permissions are private. It never prints credential contents
# and never starts a kernel.
set -euo pipefail

found=0
if [[ -n "${KAGGLE_API_TOKEN:-}" ]]; then
  echo "Found: KAGGLE_API_TOKEN environment variable (value hidden)"
  found=1
fi
if [[ -n "${KAGGLE_USERNAME:-}" && -n "${KAGGLE_KEY:-}" ]]; then
  echo "Found: KAGGLE_USERNAME/KAGGLE_KEY environment variables (values hidden)"
  found=1
fi
for f in "${HOME}/.kaggle/access_token" "${HOME}/.kaggle/kaggle.json"; do
  if [[ -f "$f" ]]; then
    perms="$(stat -c '%a' "$f")"
    echo "Found: $f (permissions ${perms})"
    if [[ "$perms" != "600" ]]; then
      echo "  WARNING: run 'chmod 600 $f' so other users cannot read it"
    fi
    found=1
  fi
done

if ! command -v kaggle >/dev/null 2>&1; then
  echo "Kaggle CLI not installed. Install with: pip install -e \".[kaggle]\""
  exit 2
fi
echo "Kaggle CLI: $(kaggle --version 2>/dev/null || echo 'version unknown')"

if [[ "$found" -eq 0 ]]; then
  echo "No Kaggle credentials found. See docs/kaggle.md (use 'kaggle auth login' or an API token)."
  exit 3
fi

# A read-only API call that lists at most one of YOUR kernels; output is suppressed.
if kaggle kernels list --mine --page-size 1 >/dev/null 2>&1; then
  echo "Authenticated API call: OK"
else
  echo "Authenticated API call: FAILED (credentials present but rejected, or no network)"
  exit 4
fi
