#!/usr/bin/env bash
# Usage: source scripts/backend.sh
set -euo pipefail


if [[ -f "$PWD/main.py" && -f "$PWD/scripts/backend.sh" ]]; then
  REPO_ROOT="$(cd "$PWD" && pwd -P)"
elif [[ "$(basename "$PWD")" == "scripts" && -f "$PWD/../main.py" ]]; then
  REPO_ROOT="$(cd "$PWD/.." && pwd -P)"
elif [[ -n "${BASH_SOURCE[0]-}" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
else
  echo "❌ Could not determine repo root."
  echo "Run this from the repo root with: source scripts/backend.sh"
  return 1 2>/dev/null || exit 1
fi


ROOT_VENV_ACTIVATE="$REPO_ROOT/.venv/bin/activate"

# --- deactivate current venv if any (only works when sourced) ---
if [[ -n "${VIRTUAL_ENV-}" ]]; then
  if type deactivate >/dev/null 2>&1; then
    deactivate || true
  fi
fi

# --- activate root venv ---
if [[ ! -f "$ROOT_VENV_ACTIVATE" ]]; then
  echo "❌ Could not find root venv activate script at:"
  echo "   $ROOT_VENV_ACTIVATE"
  echo "Fix: create .venv in repo root, or edit ROOT_VENV_ACTIVATE in this script."
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$ROOT_VENV_ACTIVATE"
echo "✅ Activated root venv: $VIRTUAL_ENV"

# --- pick uvicorn app path ---
APP=""

if [[ -f "$REPO_ROOT/src/main.py" ]]; then
  APP="src.main:app"
elif [[ -f "$REPO_ROOT/main.py" ]]; then
  APP="main:app"
else
  echo "❌ Could not find FastAPI entrypoint."
  echo "Expected backend_repo/main.py or main.py"
  echo "Edit this script to point to your module, e.g. mypackage.api:app"
  return 1 2>/dev/null || exit 1
fi

printf "Host (default 127.0.0.1): "
read -r host
host="${host:-127.0.0.1}"
printf "Port (default 8000): "
read -r port
port="${port:-8000}"

cd "$REPO_ROOT"

echo "🚀 Starting FastAPI: $APP on $host:$port"
uvicorn "$APP" --reload --app-dir "$REPO_ROOT" --host "$host" --port "$port"
