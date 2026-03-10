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
import logging
import time
from datetime import date, datetime, timezone
from typing import Dict, Optional

import requests

logger = logging.getLogger(__name__)

CLUBELO_BASE_URL = "http://api.clubelo.com"
CACHE_TTL_SECONDS = 3600  # 1 hour

# In-memory cache: { date_str -> (fetched_at_ts, {club: elo}) }
_elo_cache: Dict[str, tuple[float, Dict[str, float]]] = {}


def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


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


def get_team_elo(
    team_name: str,
    snapshot_date: Optional[str] = None,
    fallback: float = 1500.0,
) -> float:
    """Return the Elo rating for *team_name*, or *fallback* if not found.

    Tries the exact name first, then a case-insensitive search.
    """
    ratings = fetch_elo_ratings(snapshot_date)
    if team_name in ratings:
        return ratings[team_name]
    lower = team_name.lower()
    for club, elo in ratings.items():
        if club.lower() == lower:
            return elo
    return fallback
