#!/usr/bin/env bash
# Apply the repo's compatibility patches to the vendored AIrsenal submodule.
#
# Why: AIrsenal needs small season-rollover fixes that live in our repo (see
# patches/*.patch) instead of being committed to the submodule. This keeps the
# submodule pinned to a clean, remote-available upstream commit while the app
# still gets the fixes. Idempotent — safe to run on every start.
#
# Usage: bash scripts/apply_airsenal_patches.sh   (from backend/)
set -euo pipefail

SCRIPT_PATH="${BASH_SOURCE[0]:-$0}"
REPO_ROOT="$(cd "$(dirname "$SCRIPT_PATH")/.." && pwd -P)"
AIRSENAL_DIR="$REPO_ROOT/AIrsenal"
PATCH_DIR="$REPO_ROOT/patches"

if [[ ! -d "$AIRSENAL_DIR" ]]; then
  echo "ℹ️  AIrsenal not present at $AIRSENAL_DIR — skipping patches."
  exit 0
fi
if [[ ! -e "$AIRSENAL_DIR/.git" ]]; then
  echo "ℹ️  $AIRSENAL_DIR is not a git checkout — skipping patches."
  exit 0
fi
if ! command -v git >/dev/null 2>&1; then
  echo "⚠️  git not found — cannot apply AIrsenal patches." >&2
  exit 0
fi

shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [[ ${#patches[@]} -eq 0 ]]; then
  exit 0
fi

applied=0
for p in "${patches[@]}"; do
  name="$(basename "$p")"
  # Already applied? (reverse-check succeeds)
  if git -C "$AIRSENAL_DIR" apply --reverse --check "$p" >/dev/null 2>&1; then
    continue
  fi
  if git -C "$AIRSENAL_DIR" apply --check "$p" >/dev/null 2>&1; then
    git -C "$AIRSENAL_DIR" apply "$p"
    echo "✅ Applied AIrsenal patch: $name"
    applied=$((applied + 1))
  else
    echo "⚠️  Could not apply AIrsenal patch '$name'." >&2
    echo "    The submodule may be at an unexpected commit. Run:" >&2
    echo "      git -C \"$AIRSENAL_DIR\" checkout -- . && git submodule update --init" >&2
  fi
done

if [[ "$applied" -eq 0 ]]; then
  echo "✅ AIrsenal patches already applied."
fi
