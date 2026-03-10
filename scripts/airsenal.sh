#!/usr/bin/env bash
# Usage: source scripts/airsenal.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AIRSENAL_HOME_DIR="$REPO_ROOT/.airsenal_home"
AIRSENAL_VENV_ACTIVATE="$REPO_ROOT/AIrsenal/.venv/bin/activate"

# --- deactivate current venv if any (only works when sourced) ---
if [[ -n "${VIRTUAL_ENV-}" ]]; then
  if declare -F deactivate >/dev/null 2>&1; then
    deactivate || true
  fi
fi

# --- activate AIrsenal venv ---
if [[ ! -f "$AIRSENAL_VENV_ACTIVATE" ]]; then
  echo "❌ Could not find AIrsenal venv activate script at:"
  echo "   $AIRSENAL_VENV_ACTIVATE"
  echo "Fix: ensure AIrsenal/.venv exists, or edit AIRSENAL_VENV_ACTIVATE in this script."
  return 1 2>/dev/null || exit 1
fi
# shellcheck disable=SC1090
source "$AIRSENAL_VENV_ACTIVATE"

# --- export AIrsenal home + optional secrets/config ---
export AIRSENAL_HOME="$AIRSENAL_HOME_DIR"

# Optional: export values from files if present (won’t print them)
[[ -f "$AIRSENAL_HOME_DIR/AIRSENAL_DB_FILE" ]] && export AIRSENAL_DB_FILE="$(cat "$AIRSENAL_HOME_DIR/AIRSENAL_DB_FILE")"
[[ -f "$AIRSENAL_HOME_DIR/FPL_TEAM_ID" ]]     && export FPL_TEAM_ID="$(cat "$AIRSENAL_HOME_DIR/FPL_TEAM_ID")"
[[ -f "$AIRSENAL_HOME_DIR/FPL_LOGIN" ]]       && export FPL_LOGIN="$(cat "$AIRSENAL_HOME_DIR/FPL_LOGIN")"
[[ -f "$AIRSENAL_HOME_DIR/FPL_PASSWORD" ]]    && export FPL_PASSWORD="$(cat "$AIRSENAL_HOME_DIR/FPL_PASSWORD")"

echo "✅ Activated AIrsenal venv: $VIRTUAL_ENV"
echo "✅ AIRSENAL_HOME=$AIRSENAL_HOME"

TEAM_ID="${FPL_TEAM_ID:-}"
if [[ -z "$TEAM_ID" && -f "$AIRSENAL_HOME_DIR/FPL_TEAM_ID" ]]; then
  TEAM_ID="$(cat "$AIRSENAL_HOME_DIR/FPL_TEAM_ID")"
fi

echo
echo "Choose an action:"
echo "  1) Run optimization (airsenal_run_optimization)"
echo "  2) Export JSON (python adapters/airsenal_adapter.py)"
echo "  3) Run a custom command"
echo "  4) Do nothing (just keep env activated)"
read -r -p "Enter 1-4: " choice

case "$choice" in
  1)
    read -r -p "weeks_ahead (default 3): " weeks
    weeks="${weeks:-3}"
    if [[ -z "$TEAM_ID" ]]; then
      read -r -p "FPL Team ID: " TEAM_ID
    fi
    airsenal_run_optimization --weeks_ahead "$weeks" --fpl_team_id "$TEAM_ID"
    ;;
  2)
    read -r -p "GW (default auto): " gw
    gw="${gw:-auto}"
    DB="${AIRSENAL_DB_FILE:-$REPO_ROOT/data/airsenal/data.db}"
    OUT="$REPO_ROOT/data/api"
    if [[ -z "$TEAM_ID" ]]; then
      read -r -p "FPL Team ID: " TEAM_ID
    fi
    python "$REPO_ROOT/adapters/airsenal_adapter.py" \
      --db "$DB" \
      --out "$OUT" \
      --gw "$gw" \
      --team-id "$TEAM_ID"
    ;;
  3)
    echo "Enter a command to run (example: airsenal_update_db --help)"
    read -r -p "> " line
    # Run exactly what you type (in this venv + env)
    bash -lc "$line"
    ;;
  4)
    echo "👍 Leaving you in the AIrsenal environment."
    ;;
  *)
    echo "Invalid choice."
    ;;
esac
