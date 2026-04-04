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

# --- load .env.local if present ---
ENV_FILE="$REPO_ROOT/.env.local"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  source "$ENV_FILE"
  set +a
  echo "✅ Loaded environment from: $ENV_FILE"
fi

echo ""
echo "🧠 LLM Provider Setup"
printf "Provider [gemini/openrouter] (default: ${LLM_PROVIDER:-gemini}): "
read -r llm_provider
llm_provider="${llm_provider:-${LLM_PROVIDER:-gemini}}"
llm_provider="$(echo "$llm_provider" | tr '[:upper:]' '[:lower:]')"

if [[ "$llm_provider" != "gemini" && "$llm_provider" != "openrouter" ]]; then
  echo "❌ Invalid provider: $llm_provider"
  echo "Use 'gemini' or 'openrouter'"
  return 1 2>/dev/null || exit 1
fi
export LLM_PROVIDER="$llm_provider"

if [[ "$LLM_PROVIDER" == "openrouter" ]]; then
  if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
    echo "❌ OPENROUTER_API_KEY is not set (in .env.local)."
    echo "Add OPENROUTER_API_KEY=... and retry."
    return 1 2>/dev/null || exit 1
  fi

  chosen_model_default="${OPENROUTER_MODEL:-}"
  echo ""
  echo "🔎 Fetching top 10 free OpenRouter models..."

  models_json="$(curl -sS https://openrouter.ai/api/v1/models || true)"
  if [[ -n "$models_json" ]]; then
    free_models="$(python3 - <<'PY'
import json, sys
raw = sys.stdin.read().strip()
try:
    data = json.loads(raw)
except Exception:
    print("")
    raise SystemExit(0)

models = []
for m in data.get("data", []):
    pricing = m.get("pricing", {})
    if str(pricing.get("prompt", "")) == "0" and str(pricing.get("completion", "")) == "0":
        mid = m.get("id")
        if isinstance(mid, str) and mid:
            models.append(mid)

models = sorted(dict.fromkeys(models))[:10]
print("\n".join(models))
PY
<<< "$models_json")"

    if [[ -n "$free_models" ]]; then
      echo "Available free models (top 10):"
      i=1
      while IFS= read -r model; do
        echo "  $i) $model"
        i=$((i+1))
      done <<< "$free_models"
      echo ""
      echo "Choose one of the models above."
    else
      echo "⚠️ Could not parse free models list."
    fi
  else
    echo "⚠️ Could not fetch model list from OpenRouter."
  fi

  printf "OpenRouter model (optional, default: ${chosen_model_default:-openrouter/free}): "
  read -r chosen_model
  chosen_model="${chosen_model:-${chosen_model_default:-openrouter/free}}"
  export OPENROUTER_MODEL="$chosen_model"
  echo "✅ Using OpenRouter model: $OPENROUTER_MODEL"
elif [[ "$LLM_PROVIDER" == "gemini" ]]; then
  if [[ -z "${GEMINI_API_KEY:-}" ]]; then
    echo "⚠️ GEMINI_API_KEY is empty. Gemini calls will degrade/fail until you set it in .env.local."
  fi
fi

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
