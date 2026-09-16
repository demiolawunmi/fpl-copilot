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

# Interpreter for this venv (some systems have no bare `python` on PATH; use the venv binary)
AIRSENAL_PYTHON=""
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  if [[ -x "$VIRTUAL_ENV/bin/python" ]]; then
    AIRSENAL_PYTHON="$VIRTUAL_ENV/bin/python"
  elif [[ -x "$VIRTUAL_ENV/bin/python3" ]]; then
    AIRSENAL_PYTHON="$VIRTUAL_ENV/bin/python3"
  fi
fi
if [[ -z "$AIRSENAL_PYTHON" ]]; then
  if [[ -x "$REPO_ROOT/AIrsenal/.venv/bin/python" ]]; then
    AIRSENAL_PYTHON="$REPO_ROOT/AIrsenal/.venv/bin/python"
  elif [[ -x "$REPO_ROOT/AIrsenal/.venv/bin/python3" ]]; then
    AIRSENAL_PYTHON="$REPO_ROOT/AIrsenal/.venv/bin/python3"
  fi
fi
if [[ -z "$AIRSENAL_PYTHON" ]]; then
  echo "❌ Could not find python in AIrsenal venv (expected .venv/bin/python or python3)."
  return 1 2>/dev/null || exit 1
fi

# --- export AIrsenal home + optional secrets/config ---
mkdir -p "$AIRSENAL_HOME_DIR"
export AIRSENAL_HOME="$AIRSENAL_HOME_DIR"

# Optional: export values from files if present (won’t print them)
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

# Always use the repo-local AIrsenal SQLite DB after the fullstack refactor.
# Do not rely on AIRSENAL_HOME/data.db or a saved AIRSENAL_DB_FILE path, which may
# still point at an old workspace location.
REPO_AIRSENAL_DB="$REPO_ROOT/data/airsenal/data.db"
mkdir -p "$(dirname "$REPO_AIRSENAL_DB")"
export AIRSENAL_DB_FILE="$REPO_AIRSENAL_DB"

echo "✅ Activated AIrsenal venv: $VIRTUAL_ENV"
echo "✅ AIRSENAL_HOME=$AIRSENAL_HOME"
if [[ -n "${AIRSENAL_DB_FILE:-}" ]]; then
  echo "✅ AIRSENAL_DB_FILE=$AIRSENAL_DB_FILE"
fi

# pyproject [project.scripts] installs airsenal_* onto PATH; if the venv was
# created without `pip install -e .` in AIrsenal/, call the same entrypoints via Python.
AIRSENAL_SRC_ROOT="$REPO_ROOT/AIrsenal"
airsenal_cli() {
  local entry="$1"
  shift
  if command -v "$entry" >/dev/null 2>&1; then
    "$entry" "$@"
    return $?
  fi
  export PYTHONPATH="$AIRSENAL_SRC_ROOT${PYTHONPATH:+:$PYTHONPATH}"
  case "$entry" in
    airsenal_update_db)
      "$AIRSENAL_PYTHON" -c 'import sys; from airsenal.scripts.update_db import main; sys.argv = ["airsenal_update_db"] + sys.argv[1:]; main()' "$@"
      ;;
    airsenal_run_prediction)
      "$AIRSENAL_PYTHON" -c 'import sys; from airsenal.scripts.fill_predictedscore_table import main; sys.argv = ["airsenal_run_prediction"] + sys.argv[1:]; main()' "$@"
      ;;
    airsenal_run_optimization)
      "$AIRSENAL_PYTHON" -c 'import sys; from airsenal.scripts.fill_transfersuggestion_table import main; sys.argv = ["airsenal_run_optimization"] + sys.argv[1:]; main()' "$@"
      ;;
    *)
      echo "❌ Command not found: $entry" >&2
      echo "   Install AIrsenal into this venv (adds scripts to PATH):" >&2
      echo "   cd \"$AIRSENAL_SRC_ROOT\" && pip install -e ." >&2
      return 127
      ;;
  esac
}

TEAM_ID="${FPL_TEAM_ID:-}"
if [[ -z "$TEAM_ID" && -f "$AIRSENAL_HOME_DIR/FPL_TEAM_ID" ]]; then
  TEAM_ID="$(cat "$AIRSENAL_HOME_DIR/FPL_TEAM_ID")"
fi
DB="$REPO_AIRSENAL_DB"
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
  airsenal_cli airsenal_update_db
}

run_predictions() {
  local weeks="$1"
  echo
  echo "🔮 Running predictions for the next $weeks gameweek(s)..."
  airsenal_cli airsenal_run_prediction --weeks_ahead "$weeks"
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
  airsenal_cli airsenal_run_optimization --weeks_ahead "$weeks" --fpl_team_id "$TEAM_ID"
}

# Export API JSON via adapters/airsenal_adapter.py (predictions, transfers,
# gw_<GW>_optimization_<FPL_TEAM_ID>.json when --team-id is set, etc.)
run_adapter_export() {
  local gw="${1:-auto}"
  echo
  echo "📦 Exporting JSON to $OUT"
  echo "   (gw_<GW>_optimization_<TEAM_ID>.json + gw_<GW>_lineup_<TEAM_ID>.json when team id set)"
  local adapter_cmd=("$AIRSENAL_PYTHON" "$REPO_ROOT/adapters/airsenal_adapter.py" --db "$DB" --out "$OUT" --gw "$gw")
  if [[ -n "${TEAM_ID:-}" ]]; then
    adapter_cmd+=(--team-id "$TEAM_ID")
  fi
  "${adapter_cmd[@]}"
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
echo "  5) Full pipeline: Update → Predict → Optimize → Export JSON (incl. optimization)"
echo "  6) Export JSON only (airsenal_adapter: predictions, transfers, gw_*_optimization_* …)"
echo "  7) Run a custom command"
echo "  8) Do nothing (just keep env activated)"
echo "  9) Test adapter export (--gw auto; needs team id → gw_*_optimization_* + gw_*_lineup_*)"
read -r -p "Enter 1-9: " choice

case "$choice" in
  1)
    weeks="$(prompt_weeks_ahead)"
    maybe_prompt_update_before_optimization
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
    weeks="$(prompt_weeks_ahead)"
    run_update_db
    run_predictions "$weeks"
    ;;
  5)
    weeks="$(prompt_weeks_ahead)"
    ensure_team_id || return 1 2>/dev/null || exit 1
    read -r -p "GW for JSON export / optimization file (default auto): " export_gw
    export_gw="${export_gw:-auto}"
    run_update_db
    run_predictions "$weeks"
    run_optimization "$weeks" || return 1 2>/dev/null || exit 1
    run_adapter_export "$export_gw"
    ;;
  6)
    read -r -p "GW (default auto): " gw
    gw="${gw:-auto}"
    if [[ -z "$TEAM_ID" ]]; then
      read -r -p "FPL Team ID (optional; required for gw_*_optimization_*): " TEAM_ID
    fi
    run_adapter_export "$gw"
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
  9)
    echo
    echo "🧪 Test: run airsenal_adapter with --gw auto (writes JSON under $OUT)."
    echo "   Team id is required for gw_<GW>_optimization_<TEAM> and gw_<GW>_lineup_<TEAM>."
    ensure_team_id || return 1 2>/dev/null || exit 1
    run_adapter_export "auto"
    ;;
  *)
    echo "Invalid choice."
    ;;
esac
