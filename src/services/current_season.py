"""
Season detection for FPL Copilot.

Derives the current FPL season string (e.g. ``"2627"`` for 2026/27) from the
live official bootstrap feed instead of trusting local snapshots, and exposes
a freshness check comparing it against the locally generated data
(``data/api/teams.json`` / AIrsenal DB), which lags until the refresh pipeline
(``airsenal_update_db`` + adapter exports) has run after a season rollover.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import ssl
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

FPL_BOOTSTRAP_URL = "https://fantasy.premierleague.com/api/bootstrap-static/"

CACHE_TTL_SECONDS = 3600

_REPO_ROOT = Path(__file__).resolve().parents[2]
TEAMS_PATH = _REPO_ROOT / "data" / "api" / "teams.json"
AIRSENAL_DB_PATH = _REPO_ROOT / "data" / "airsenal" / "data.db"

_fetch_ts: float = 0.0
_cached_events: List[Dict[str, Any]] = []
_stale_events: List[Dict[str, Any]] = []


def _ssl_context() -> ssl.SSLContext:
    """SSL context with certifi CAs (Framework Pythons on macOS lack system certs)."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def fetch_fpl_events() -> List[Dict[str, Any]]:
    """Fetch gameweek events from the live bootstrap feed (1h TTL, non-raising)."""
    global _fetch_ts, _cached_events, _stale_events

    now = time.time()
    if _cached_events and (now - _fetch_ts) < CACHE_TTL_SECONDS:
        return _cached_events

    try:
        req = urllib.request.Request(
            FPL_BOOTSTRAP_URL,
            headers={"User-Agent": "FPLCopilot/1.0 (season detection)"},
        )
        with urllib.request.urlopen(req, timeout=30, context=_ssl_context()) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        events = data.get("events") if isinstance(data, dict) else None
        if not isinstance(events, list) or not events:
            return _stale_events or []

        _cached_events = events
        _stale_events = events
        _fetch_ts = now
        return _cached_events
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return _stale_events or []


def derive_season_from_deadline(deadline_iso: str) -> Optional[str]:
    """Convert an event deadline like ``2026-08-21T17:30:00Z`` into ``"2627"``."""
    try:
        dt = datetime.fromisoformat(deadline_iso.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    # Seasons run Aug–May: from June onwards the season starts this year.
    start_year = dt.year if dt.month > 5 else dt.year - 1
    return f"{str(start_year)[2:]}{str(start_year + 1)[2:]}"


def get_fpl_season() -> Optional[str]:
    """Current season per the live FPL feed, e.g. ``"2627"``; None if unavailable."""
    events = fetch_fpl_events()
    deadlines = sorted(
        str(e.get("deadline_time"))
        for e in events
        if e.get("deadline_time")
    )
    if not deadlines:
        return None
    return derive_season_from_deadline(deadlines[0])


def get_teams_json_season() -> Optional[str]:
    """Season of the locally exported teams.json (MAX over rows)."""
    try:
        with TEAMS_PATH.open("r", encoding="utf-8") as f:
            teams = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    seasons = [str(t.get("season", "")) for t in teams if t.get("season")]
    return max(seasons) if seasons else None


def get_airsenal_db_season() -> Optional[str]:
    """Season of the local AIrsenal DB fixture table (MAX)."""
    try:
        con = sqlite3.connect(f"file:{AIRSENAL_DB_PATH}?mode=ro", uri=True)
        try:
            row = con.execute("SELECT MAX(season) FROM fixture").fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return str(row[0]) if row and row[0] else None


def get_season_status() -> Dict[str, Any]:
    """Compare live FPL season against local data snapshots."""
    fpl_season = get_fpl_season()
    data_candidates = [
        s for s in (get_teams_json_season(), get_airsenal_db_season()) if s
    ]
    data_season = max(data_candidates) if data_candidates else None
    return {
        "fpl_season": fpl_season,
        "data_season": data_season,
        "is_current": bool(fpl_season and data_season == fpl_season),
        "checked_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }
