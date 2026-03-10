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
if [[ -f "$AIRSENAL_HOME_DIR/AIRSENAL_DB_FILE" ]]; then
  AIRSENAL_DB_FILE="$(cat "$AIRSENAL_HOME_DIR/AIRSENAL_DB_FILE")"
  export AIRSENAL_DB_FILE
fi
if [[ -f "$AIRSENAL_HOME_DIR/FPL_TEAM_ID" ]]; then
  FPL_TEAM_ID="$(cat "$AIRSENAL_HOME_DIR/FPL_TEAM_ID")"
  export FPL_TEAM_ID
fi
if [[ -f "$AIRSENAL_HOME_DIR/FPL_LOGIN" ]]; then
  FPL_LOGIN="$(cat "$AIRSENAL_HOME_DIR/FPL_LOGIN")"
  export FPL_LOGIN
fi
if [[ -f "$AIRSENAL_HOME_DIR/FPL_PASSWORD" ]]; then
  FPL_PASSWORD="$(cat "$AIRSENAL_HOME_DIR/FPL_PASSWORD")"
  export FPL_PASSWORD
fi

echo "✅ Activated AIrsenal venv: $VIRTUAL_ENV"
echo "✅ AIRSENAL_HOME=$AIRSENAL_HOME"

TEAM_ID="${FPL_TEAM_ID:-}"
if [[ -z "$TEAM_ID" && -f "$AIRSENAL_HOME_DIR/FPL_TEAM_ID" ]]; then
  TEAM_ID="$(cat "$AIRSENAL_HOME_DIR/FPL_TEAM_ID")"
fi
DB="${AIRSENAL_DB_FILE:-$REPO_ROOT/data/airsenal/data.db}"
OUT="$REPO_ROOT/data/api"

get_file_mtime() {
  local path="$1"
  if stat -f %m "$path" >/dev/null 2>&1; then
    stat -f %m "$path"
  elif stat -c %Y "$path" >/dev/null 2>&1; then
    stat -c %Y "$path"
  else
    return 1
  fi
}

db_age_days() {
  local path="$1"
  local mtime now
  [[ -f "$path" ]] || return 1
  mtime="$(get_file_mtime "$path")" || return 1
  now="$(date +%s)"
  echo $(((now - mtime) / 86400))
}

prompt_yes_no() {
  local prompt="$1"
  local default="${2:-n}"
  local suffix answer

  if [[ "$default" == "y" ]]; then
    suffix="[Y/n]"
  else
    suffix="[y/N]"
  fi

  read -r -p "$prompt $suffix " answer
  answer="${answer:-$default}"
  [[ "$answer" =~ ^[Yy]([Ee][Ss])?$ ]]
}

run_update_db() {
  echo
  echo "🔄 Updating the AIrsenal database..."
  airsenal_update_db
}

run_predictions() {
  local weeks="$1"
  echo
  echo "🔮 Running predictions for the next $weeks gameweek(s)..."
  airsenal_run_prediction --weeks_ahead "$weeks"
}

prompt_weeks_ahead() {
  local weeks
  read -r -p "weeks_ahead (default 3): " weeks
  echo "${weeks:-3}"
}

run_optimization() {
  local weeks="$1"
  ensure_team_id || return 1
  echo
  echo "🧠 Running optimization for the next $weeks gameweek(s)..."
  airsenal_run_optimization --weeks_ahead "$weeks" --fpl_team_id "$TEAM_ID"
}

ensure_team_id() {
  if [[ -z "$TEAM_ID" ]]; then
    read -r -p "FPL Team ID: " TEAM_ID
  fi

  if [[ -z "$TEAM_ID" ]]; then
    echo "❌ FPL Team ID is required for optimization."
    return 1
  fi
}

maybe_prompt_update_before_optimization() {
  local default_update="n"
  local age_days

  echo
  if age_days="$(db_age_days "$DB" 2>/dev/null)"; then
    if (( age_days >= 7 )); then
      echo "⚠️  Database looks $age_days day(s) old: $DB"
      echo "   Updating before optimization is recommended."
      default_update="y"
    else
      echo "✅ Database was updated about $age_days day(s) ago: $DB"
    fi
  elif [[ -f "$DB" ]]; then
    echo "ℹ️  Found database at $DB but couldn't determine its age."
    echo "   Updating before optimization is recommended."
    default_update="y"
  else
    echo "⚠️  Database file not found at: $DB"
    echo "   Updating before optimization is recommended."
    default_update="y"
  fi

  if prompt_yes_no "Update the database before optimization?" "$default_update"; then
    run_update_db
  fi
}

echo
echo "Choose an action:"
echo "  1) Run optimization (airsenal_run_optimization)"
echo "  2) Update database (airsenal_update_db)"
echo "  3) Run predictions (airsenal_run_prediction --weeks_ahead 3)"
echo "  4) Update DB + Run Predictions"
echo "  5) Full pipeline: Update → Predict → Optimize"
echo "  6) Export JSON (python adapters/airsenal_adapter.py)"
echo "  7) Run a custom command"
echo "  8) Do nothing (just keep env activated)"
read -r -p "Enter 1-8: " choice

case "$choice" in
  1)
    maybe_prompt_update_before_optimization
    weeks="$(prompt_weeks_ahead)"
    run_optimization "$weeks" || return 1 2>/dev/null || exit 1
    ;;
  2)
    run_update_db
    ;;
  3)
    weeks="$(prompt_weeks_ahead)"
    run_predictions "$weeks"
    ;;
  4)
    run_update_db
    weeks="$(prompt_weeks_ahead)"
    run_predictions "$weeks"
    ;;
  5)
    run_update_db
    weeks="$(prompt_weeks_ahead)"
    run_predictions "$weeks"
    run_optimization "$weeks" || return 1 2>/dev/null || exit 1
    ;;
  6)
    read -r -p "GW (default auto): " gw
    gw="${gw:-auto}"
    if [[ -z "$TEAM_ID" ]]; then
      read -r -p "FPL Team ID (optional, improves transfer pairing): " TEAM_ID
    fi
    adapter_cmd=(python "$REPO_ROOT/adapters/airsenal_adapter.py" --db "$DB" --out "$OUT" --gw "$gw")
    if [[ -n "$TEAM_ID" ]]; then
      adapter_cmd+=(--team-id "$TEAM_ID")
    fi
    "${adapter_cmd[@]}"
    ;;
  7)
    echo "Enter a command to run (example: airsenal_update_db --help)"
    read -r -p "> " line
    # Run exactly what you type (in this venv + env)
    bash -lc "$line"
    ;;
  8)
    echo "👍 Leaving you in the AIrsenal environment."
    ;;
  *)
    echo "Invalid choice."
    ;;
esac
