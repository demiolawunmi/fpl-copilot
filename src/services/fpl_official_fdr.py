"""
Official FPL fixture difficulty from the FPL fixtures feed.

Source: ``GET https://fantasy.premierleague.com/api/fixtures/``

Pipeline placement (see ``main._build_fixture_fdr_response``):
1. ``resolve_fixture_context`` — existing fixture resolution from ``fixtures.json`` / team mode.
2. **This module** — attach ``official_fpl_*`` fields from the FPL API (cached, non-fatal).
3. Injury player loading and ``compute_fixture_fdr`` — custom Elo / injury / squad-change FDR.

The official integers (``team_h_difficulty`` / ``team_a_difficulty``) are never mixed into
the custom Elo-based metrics; they are exposed separately on the saturated payload.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import urllib.error
import urllib.request

FPL_FIXTURES_URL = "https://fantasy.premierleague.com/api/fixtures/"

CACHE_TTL_SECONDS = 3600

_fetch_ts: float = 0.0
_cached_raw: List[Dict[str, Any]] = []
_stale_raw: List[Dict[str, Any]] = []


def fetch_fpl_fixtures() -> List[Dict[str, Any]]:
    """Fetch the official FPL fixtures JSON array.

    Uses a 1-hour in-memory TTL. On HTTP failure, returns the last successful
    payload (stale) if any; otherwise an empty list. Network errors do not raise.
    """
    global _fetch_ts, _cached_raw, _stale_raw

    now = time.time()
    if _cached_raw and (now - _fetch_ts) < CACHE_TTL_SECONDS:
        return _cached_raw

    try:
        req = urllib.request.Request(
            FPL_FIXTURES_URL,
            headers={"User-Agent": "FPLCopilot/1.0 (fixture difficulty)"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        if not isinstance(data, list):
            return _stale_raw or []

        _cached_raw = data
        _stale_raw = data
        _fetch_ts = now
        return _cached_raw
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError):
        return _stale_raw or []


def normalize_fpl_fixture(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Keep fields used in-app plus FPL difficulty; ignore the rest."""
    return {
        "id": raw.get("id"),
        "event": raw.get("event"),
        "kickoff_time": raw.get("kickoff_time"),
        "team_h": raw.get("team_h"),
        "team_a": raw.get("team_a"),
        "team_h_difficulty": raw.get("team_h_difficulty"),
        "team_a_difficulty": raw.get("team_a_difficulty"),
        "finished": raw.get("finished"),
        "started": raw.get("started"),
        "team_h_score": raw.get("team_h_score"),
        "team_a_score": raw.get("team_a_score"),
    }


def build_fixture_difficulty_lookup(
    fixtures: List[Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[Tuple[int, int, int], Dict[str, Any]]]:
    """Index FPL fixtures by ``id`` and by ``(team_h, team_a, event)``."""
    by_id: Dict[int, Dict[str, Any]] = {}
    by_h_a_event: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for f in fixtures:
        fid = f.get("id")
        if fid is not None:
            try:
                by_id[int(fid)] = f
            except (TypeError, ValueError):
                pass
        th, ta, ev = f.get("team_h"), f.get("team_a"), f.get("event")
        if th is not None and ta is not None and ev is not None:
            try:
                by_h_a_event[(int(th), int(ta), int(ev))] = f
            except (TypeError, ValueError):
                pass
    return by_id, by_h_a_event


def get_official_fpl_difficulty_for_team(
    fixture: Dict[str, Any],
    team_id: int,
    is_home: bool,
) -> Optional[int]:
    """Return the official difficulty for *team_id*'s side (home vs away).

    - Home team: ``team_h_difficulty``
    - Away team: ``team_a_difficulty``
    """
    th = fixture.get("team_h")
    ta = fixture.get("team_a")
    try:
        tid = int(team_id)
        th_i = int(th) if th is not None else None
        ta_i = int(ta) if ta is not None else None
    except (TypeError, ValueError):
        return None

    if tid == th_i and is_home:
        return _as_int_or_none(fixture.get("team_h_difficulty"))
    if tid == ta_i and not is_home:
        return _as_int_or_none(fixture.get("team_a_difficulty"))
    # Fallback: if is_home disagrees with FPL ids, still map by id
    if tid == th_i:
        return _as_int_or_none(fixture.get("team_h_difficulty"))
    if tid == ta_i:
        return _as_int_or_none(fixture.get("team_a_difficulty"))
    return None


def _as_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def resolve_fpl_fixture_for_context(
    fixtures: List[Dict[str, Any]],
    by_id: Dict[int, Dict[str, Any]],
    by_h_a_event: Dict[Tuple[int, int, int], Dict[str, Any]],
    context: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Match an FPL fixture row to our saturated ``context`` (prefer fixture id, then team ids + GW)."""
    fid = context.get("fixture_id")
    if fid is not None:
        try:
            k = int(fid)
            if k in by_id:
                return by_id[k]
        except (TypeError, ValueError):
            pass

    ht = context.get("home_team_id")
    at = context.get("away_team_id")
    if ht is None or at is None:
        return None
    try:
        ht_i, at_i = int(ht), int(at)
    except (TypeError, ValueError):
        return None

    gw = context.get("gameweek")
    if gw is not None:
        try:
            gwi = int(gw)
            key = (ht_i, at_i, gwi)
            if key in by_h_a_event:
                return by_h_a_event[key]
        except (TypeError, ValueError):
            pass

    candidates = [
        f
        for f in fixtures
        if f.get("team_h") == ht_i and f.get("team_a") == at_i
    ]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1 and gw is not None:
        try:
            gwi = int(gw)
            for c in candidates:
                if c.get("event") == gwi:
                    return c
        except (TypeError, ValueError):
            pass
    return None


def official_fpl_source_flag(fixtures_loaded: bool) -> str:
    """``fpl_api`` when we have any usable fixture list; ``unavailable`` when empty."""
    return "fpl_api" if fixtures_loaded else "unavailable"


def build_official_fpl_fields(context: Dict[str, Any]) -> Dict[str, Any]:
    """Produce ``official_fpl_*`` keys to merge into the saturated FDR response.

    Never raises. On total failure, returns nulls and ``official_fpl_source=unavailable``.
    """
    raw = fetch_fpl_fixtures()
    if not raw:
        return {
            "official_fpl_fdr": None,
            "official_fpl_home_difficulty": None,
            "official_fpl_away_difficulty": None,
            "official_fpl_event": None,
            "official_fpl_kickoff_time": None,
            "official_fpl_source": "unavailable",
        }

    fixtures = [normalize_fpl_fixture(x) for x in raw if isinstance(x, dict)]
    by_id, by_h_a_event = build_fixture_difficulty_lookup(fixtures)
    fpl_fx = resolve_fpl_fixture_for_context(fixtures, by_id, by_h_a_event, context)

    if not fpl_fx:
        return {
            "official_fpl_fdr": None,
            "official_fpl_home_difficulty": None,
            "official_fpl_away_difficulty": None,
            "official_fpl_event": None,
            "official_fpl_kickoff_time": None,
            "official_fpl_source": official_fpl_source_flag(True),
        }

    team_id = context.get("team_id")
    is_home = bool(context.get("is_home"))
    side: Optional[int] = None
    if team_id is not None:
        try:
            side = get_official_fpl_difficulty_for_team(fpl_fx, int(team_id), is_home)
        except (TypeError, ValueError):
            side = None

    return {
        "official_fpl_fdr": side,
        "official_fpl_home_difficulty": _as_int_or_none(fpl_fx.get("team_h_difficulty")),
        "official_fpl_away_difficulty": _as_int_or_none(fpl_fx.get("team_a_difficulty")),
        "official_fpl_event": _as_int_or_none(fpl_fx.get("event")),
        "official_fpl_kickoff_time": fpl_fx.get("kickoff_time") if isinstance(fpl_fx.get("kickoff_time"), str) else None,
        "official_fpl_source": official_fpl_source_flag(True),
    }
