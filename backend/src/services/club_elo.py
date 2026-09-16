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
from pathlib import Path
from typing import Any, Dict, Optional

import requests

logger = logging.getLogger(__name__)

CLUBELO_BASE_URL = "http://api.clubelo.com"
CACHE_TTL_SECONDS = 3600  # 1 hour
FAILURE_COOLDOWN_SECONDS = 300  # skip a failing API for 5 minutes

# In-memory cache: { date_str -> (fetched_at_ts, {club: elo}) }
_elo_cache: Dict[str, tuple[float, Dict[str, float]]] = {}
# Cooldown bookkeeping so an outaged upstream isn't re-timeout'd on every request.
_last_failure_ts: Dict[str, float] = {}

_TEAM_NAME_ALIASES: Dict[str, tuple[str, ...]] = {
    "mancity": ("ManCity", "Manchester City"),
    "manutd": ("ManUnited", "Manchester United"),
    "spurs": ("Tottenham", "Tottenham Hotspur"),
    "nottmforest": ("Nottingham Forest", "Forest"),
    "wolves": ("Wolverhampton", "Wolverhampton Wanderers"),
    # Promoted-club short forms used by ClubElo slugs / CSV.
    "hullcity": ("Hull",),
    "coventrycity": ("Coventry",),
    "ipswichtown": ("Ipswich",),
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


# Generic name suffixes that differ between FPL and ClubElo naming
# ("Hull City" vs "Hull", "Ipswich Town" vs "Ipswich").
_GENERIC_SUFFIXES = ("city", "town")


def _normalized_variants(key: str) -> set[str]:
    """The normalized key plus suffix-stripped variants (e.g. hullcity → {hullcity, hull})."""
    variants = {key}
    for suffix in _GENERIC_SUFFIXES:
        if key.endswith(suffix) and len(key) > len(suffix) + 2:
            variants.add(key[: -len(suffix)])
    return variants


def _candidate_team_names(team_name: str) -> list[str]:
    normalized = _normalize_team_name(team_name)
    candidates = [team_name]
    candidates.extend(_TEAM_NAME_ALIASES.get(normalized, ()))
    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


# Cache keyed by teams.json mtime so a regenerated export (season rollover,
# promoted clubs) is picked up without restarting the server.
_teams_mtime_cache: Optional[tuple[float, list[dict[str, Any]]]] = None


def get_current_premier_league_teams() -> list[dict[str, Any]]:
    global _teams_mtime_cache

    try:
        mtime = TEAMS_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _teams_mtime_cache is not None and _teams_mtime_cache[0] == mtime:
        return _teams_mtime_cache[1]

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
    _teams_mtime_cache = (mtime, current_teams)
    return current_teams


def get_current_premier_league_team_names() -> list[str]:
    return [team["full_name"] for team in get_current_premier_league_teams()]


# FPL short code (e.g. "COV") -> full club name, keyed by teams.json mtime so a
# regenerated export (season rollover / promoted clubs) is picked up live.
_team_code_cache: Optional[tuple[float, Dict[str, str]]] = None


def get_team_code_to_full_name() -> Dict[str, str]:
    """Map FPL short code (``COV``) → full club name for the current season.

    Sourced from ``data/api/teams.json`` — the same file that drives
    ``/api/fdr/elo`` — so short-code resolution can never drift from the
    fixtures/FDR resolver when clubs are promoted or relegated.
    """
    global _team_code_cache

    try:
        mtime = TEAMS_PATH.stat().st_mtime
    except OSError:
        mtime = 0.0
    if _team_code_cache is not None and _team_code_cache[0] == mtime:
        return _team_code_cache[1]

    with TEAMS_PATH.open("r", encoding="utf-8") as f:
        teams = json.load(f)

    seasons = [str(team.get("season", "")) for team in teams if team.get("season")]
    current_season = max(seasons) if seasons else None

    mapping: Dict[str, str] = {}
    for team in teams:
        if current_season is not None and str(team.get("season")) != current_season:
            continue
        code = team.get("name")
        full_name = team.get("full_name")
        if code and full_name:
            mapping[str(code).upper()] = str(full_name)
    _team_code_cache = (mtime, mapping)
    return mapping


def resolve_team_elo(ratings: Dict[str, float], team_name: str) -> Optional[float]:
    for candidate in _candidate_team_names(team_name):
        if candidate in ratings:
            return ratings[candidate]

    candidate_keys = {_normalize_team_name(candidate) for candidate in _candidate_team_names(team_name)}
    for club, elo in ratings.items():
        if _normalize_team_name(club) in candidate_keys:
            return elo

    # Pass 3 (dynamic, no per-club config): match after stripping generic
    # suffixes so FPL "Hull City" matches ClubElo "Hull", "Ipswich Town" →
    # "Ipswich", etc., for any promoted/relegated club automatically.
    candidate_variants = set()
    for key in candidate_keys:
        candidate_variants.update(_normalized_variants(key))
    for club, elo in ratings.items():
        if _normalized_variants(_normalize_team_name(club)) & candidate_variants:
            return elo
    return None


def resolve_team_elo_by_code(
    ratings: Dict[str, float], team_code: str
) -> Optional[float]:
    """Resolve Elo for an FPL short code via the shared teams.json mapping.

    Single entry point for consumers that only have a short code (e.g. the
    copilot blend scorer reading ``player_attributes.team``), so they don't
    re-implement (and drift from) the fixtures/FDR name resolution.
    """
    full_name = get_team_code_to_full_name().get(team_code.upper())
    return resolve_team_elo(ratings, full_name or team_code)


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


def _most_recent_cached_ratings(date_str: str) -> Optional[Dict[str, float]]:
    """Ratings from the most recent cached snapshot on/before *date_str*.

    ClubElo ratings move slowly, so serving a few-days-old snapshot beats
    failing outright when the upstream API has a transient outage.
    """
    candidates = [d for d in _elo_cache if d <= date_str]
    if not candidates:
        return None
    best = max(candidates)
    fetched_at, ratings = _elo_cache[best]
    if not ratings:
        return None
    logger.info(
        "ClubElo: serving stale cache for %s (fetched for %s)",
        date_str,
        best,
    )
    return ratings


def fetch_elo_ratings(
    snapshot_date: Optional[str] = None,
    timeout: float = 15.0,
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
        On request failure: fresh cache → stale cache (nearest earlier date)
        → empty dict.
    """
    date_str = snapshot_date or _today_utc()

    # Serve from cache if fresh
    cached = _elo_cache.get(date_str)
    if cached is not None:
        fetched_at, ratings = cached
        if time.monotonic() - fetched_at < CACHE_TTL_SECONDS:
            return ratings

    url = f"{CLUBELO_BASE_URL}/{date_str}"

    # Within the failure cooldown, don't re-pay timeout costs — go straight
    # to the scraper / cache fallbacks.
    last_fail = _last_failure_ts.get(date_str)
    if last_fail is not None and time.monotonic() - last_fail < FAILURE_COOLDOWN_SECONDS:
        return _fallback_ratings(date_str, cached)

    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            last_exc = None
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))

    if last_exc is not None:
        logger.warning("ClubElo request failed for %s: %s", date_str, last_exc)
        _last_failure_ts[date_str] = time.monotonic()
        return _fallback_ratings(date_str, cached)

    _last_failure_ts.pop(date_str, None)
    ratings = _parse_clubelo_csv(resp.text)
    _elo_cache[date_str] = (time.monotonic(), ratings)
    return ratings


def _fallback_ratings(
    date_str: str, cached: Optional[tuple[float, Dict[str, float]]]
) -> Dict[str, float]:
    """Scraper fallback, then fresh-then-stale cache."""
    # Fallback 1: scrape the rankings page (independent of the CSV API).
    try:
        from src.services.club_elo_scraper import scrape_elo_ratings

        scraped = scrape_elo_ratings()
        if scraped:
            _elo_cache[date_str] = (time.monotonic(), scraped)
            return scraped
    except Exception as scrap_exc:
        logger.warning("ClubElo scraper fallback failed: %s", scrap_exc)

    # Fallback 2: today's cached snapshot, else nearest earlier date.
    if cached is not None and cached[1]:
        return cached[1]
    stale = _most_recent_cached_ratings(date_str)
    return stale if stale is not None else {}


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
