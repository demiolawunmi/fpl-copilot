"""
ClubElo ingestion service.

Fetches the ClubElo CSV snapshot for a given date and returns a
club-name → Elo dict.  Results are cached in memory for one hour so
that repeated calls within the same backend process don't hit the
remote API repeatedly.

ClubElo API (http://api.clubelo.com):
  Date snapshot: GET http://api.clubelo.com/YYYY-MM-DD
  Response: CSV with columns Rank,Club,Country,Level,Elo,From,To
"""

from __future__ import annotations

import csv
import io
import json
import logging
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

CLUBELO_BASE_URL = "http://api.clubelo.com"
CACHE_TTL_SECONDS = 3600  # 1 hour

# In-memory cache: { date_str -> (fetched_at_ts, {club: elo}) }
_elo_cache: Dict[str, tuple[float, Dict[str, float]]] = {}

_TEAM_NAME_ALIASES: Dict[str, tuple[str, ...]] = {
    "mancity": ("ManCity", "Manchester City"),
    "manutd": ("ManUnited", "Manchester United"),
    "spurs": ("Tottenham", "Tottenham Hotspur"),
    "nottmforest": ("Nottingham Forest", "Forest"),
    "wolves": ("Wolverhampton", "Wolverhampton Wanderers"),
}


def _find_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        if (candidate / "data" / "api" / "teams.json").is_file():
            return candidate
    raise RuntimeError("Could not locate repo root containing data/api/teams.json")


TEAMS_PATH = _find_repo_root() / "data" / "api" / "teams.json"


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _normalize_team_name(name: str) -> str:
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _candidate_team_names(team_name: str) -> list[str]:
    normalized = _normalize_team_name(team_name)
    candidates = [team_name]
    candidates.extend(_TEAM_NAME_ALIASES.get(normalized, ()))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


@lru_cache(maxsize=1)
def get_current_premier_league_teams() -> list[dict[str, Any]]:
    with TEAMS_PATH.open("r", encoding="utf-8") as f:
        teams = json.load(f)

    seasons = [str(team.get("season", "")) for team in teams if team.get("season")]
    current_season = max(seasons) if seasons else None

    current_teams = []
    for team in teams:
        if current_season is not None and str(team.get("season")) != current_season:
            continue
        full_name = team.get("full_name")
        team_id = team.get("team_id")
        if full_name and team_id is not None:
            current_teams.append({"team_id": int(team_id), "full_name": full_name})
    return current_teams


@lru_cache(maxsize=1)
def get_current_premier_league_team_names() -> list[str]:
    return [team["full_name"] for team in get_current_premier_league_teams()]


def resolve_team_elo(ratings: Dict[str, float], team_name: str) -> Optional[float]:
    for candidate in _candidate_team_names(team_name):
        if candidate in ratings:
            return ratings[candidate]

    candidate_keys = {_normalize_team_name(candidate) for candidate in _candidate_team_names(team_name)}
    for club, elo in ratings.items():
        if _normalize_team_name(club) in candidate_keys:
            return elo
    return None


def build_premier_league_elo_snapshot(
    ratings: Dict[str, float],
    snapshot_date: Optional[str] = None,
) -> Dict[str, object]:
    date_str = snapshot_date or _today_utc()
    premier_league_ratings = []

    for team in get_current_premier_league_teams():
        team_name = team["full_name"]
        elo = resolve_team_elo(ratings, team_name)
        if elo is None:
            logger.warning("No ClubElo rating matched current FPL team '%s'", team_name)
            continue
        premier_league_ratings.append(
            {
                "team_id": team["team_id"],
                "team": team_name,
                "elo": round(float(elo), 2),
            }
        )

    return {"snapshot_date": date_str, "ratings": premier_league_ratings}


def _parse_clubelo_csv(text: str) -> Dict[str, float]:
    """Parse ClubElo CSV text into ``{club: elo}``."""
    ratings: Dict[str, float] = {}
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        club = (row.get("Club") or "").strip()
        elo_str = (row.get("Elo") or "").strip()
        if club and elo_str:
            try:
                ratings[club] = float(elo_str)
            except ValueError:
                pass
    return ratings


def fetch_elo_ratings(
    snapshot_date: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, float]:
    """Return a mapping of ClubElo club name → Elo rating.

    Parameters
    ----------
    snapshot_date:
        ISO date string ``"YYYY-MM-DD"``.  Defaults to today (UTC).
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    dict
        ``{club_name: elo_float}`` for every club in the response,
        e.g. ``{"ManCity": 2060.3, "Arsenal": 2047.1, ...}``.
        Returns an empty dict if the request fails.
    """
    date_str = snapshot_date or _today_utc()

    # Serve from cache if fresh
    cached = _elo_cache.get(date_str)
    if cached is not None:
        fetched_at, ratings = cached
        if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
            return ratings

    url = f"{CLUBELO_BASE_URL}/{date_str}"
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("ClubElo request failed for %s: %s", date_str, exc)
        # Return stale cached data if available, otherwise empty dict
        if cached is not None:
            return cached[1]
        return {}

    ratings = _parse_clubelo_csv(resp.text)
    _elo_cache[date_str] = (time.monotonic(), ratings)
    return ratings


def fetch_premier_league_elo_snapshot(
    snapshot_date: Optional[str] = None,
    timeout: float = 10.0,
) -> Dict[str, object]:
    ratings = fetch_elo_ratings(snapshot_date=snapshot_date, timeout=timeout)
    return build_premier_league_elo_snapshot(ratings, snapshot_date=snapshot_date)


def get_team_elo(
    team_name: str,
    snapshot_date: Optional[str] = None,
    fallback: float = 1500.0,
) -> float:
    """Return the Elo rating for *team_name*, or *fallback* if not found.

    Tries exact and alias matches first, then a normalized name search.
    """
    ratings = fetch_elo_ratings(snapshot_date)
    elo = resolve_team_elo(ratings, team_name)
    return elo if elo is not None else fallback
