from __future__ import annotations

import json
import sqlite3
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _find_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        if (candidate / "data" / "api" / "injury_news.json").is_file() and (candidate / "data" / "api" / "teams.json").is_file():
            return candidate
    raise RuntimeError("Could not locate repo root containing data/api injury files")


REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "data" / "api"
INJURY_NEWS_PATH = DATA_DIR / "injury_news.json"
TEAMS_PATH = DATA_DIR / "teams.json"
DB_PATH = REPO_ROOT / "data" / "airsenal" / "data.db"
FIXTURES_PATH = DATA_DIR / "fixtures.json"

_TEAM_CACHE: list[dict[str, Any]] | None = None
_INJURY_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_DB_PLAYER_CACHE: tuple[float, list[dict[str, Any]]] | None = None
_FIXTURE_CACHE: tuple[float, list[dict[str, Any]]] | None = None

_POSITION_GROUPS = {
    "GK": "goalkeeper",
    "CB": "defence",
    "FB": "defence",
    "WB": "defence",
    "DM": "midfield",
    "CM": "midfield",
    "AM": "attack",
    "W": "attack",
    "ST": "attack",
}

_POSITION_CANONICAL = {
    "GK": "GK",
    "GKP": "GK",
    "DEF": "CB",
    "CB": "CB",
    "FB": "FB",
    "WB": "WB",
    "MID": "CM",
    "DM": "DM",
    "CM": "CM",
    "AM": "AM",
    "W": "W",
    "FWD": "ST",
    "FW": "ST",
    "ST": "ST",
}

_PLACEHOLDER_STRINGS = {"", "string", "none", "null"}

_TEAM_ALIASES_BY_CODE: dict[str, tuple[str, ...]] = {
    "BHA": ("Brighton & Hove Albion", "Brighton and Hove Albion"),
    "LEE": ("Leeds United",),
    "MCI": ("Manchester City", "ManCity"),
    "MUN": ("Manchester United", "Man United", "ManUnited"),
    "NEW": ("Newcastle United",),
    "NFO": ("Nottingham Forest", "Nottm Forest", "Forest"),
    "SUN": ("Sunderland AFC",),
    "TOT": ("Tottenham", "Tottenham Hotspur"),
    "WHU": ("West Ham United",),
    "WOL": ("Wolverhampton", "Wolverhampton Wanderers"),
}

_LOAN_OUT_PHRASES = (
    " on loan ",
    " loan for the rest of the season",
    " loan spell",
)

_TRANSFER_OUT_PHRASES = (
    " permanently",
    " has joined ",
    " joined ",
    " has signed for ",
    " transferred ",
)

_INJURY_HINTS = (
    "injury",
    "knock",
    "illness",
    "expected back",
    "unknown return date",
    "hamstring",
    "knee",
    "ankle",
    "foot",
    "thigh",
    "calf",
    "back ",
    "back injury",
    "muscle",
    "groin",
    "leg ",
    "shoulder",
    "concussion",
    "fracture",
    "acl",
)


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(ch for ch in ascii_text.lower() if ch.isalnum())


def _file_last_updated(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _is_placeholder_text(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in _PLACEHOLDER_STRINGS


def sanitize_nullable_string(value: Any) -> Any:
    return None if _is_placeholder_text(value) else value


def sanitize_snapshot_date(value: Optional[str]) -> Optional[str]:
    return sanitize_nullable_string(value)


def classify_absence_type(
    source_news: Optional[str],
    status: Optional[str],
    prob_available: Optional[float] = None,
) -> str:
    news = f" {(source_news or '').lower()} "
    normalized_status = (status or "").lower()

    if normalized_status == "suspended" or " suspend" in news:
        return "suspension"
    if any(phrase in news for phrase in _LOAN_OUT_PHRASES):
        return "loan_out"
    if any(phrase in news for phrase in _TRANSFER_OUT_PHRASES):
        return "transfer_out"
    if normalized_status == "doubtful":
        return "doubtful"
    if normalized_status == "major_doubt":
        return "major_doubt"
    if normalized_status == "questionable":
        return "questionable"

    if prob_available is not None:
        if 0.5 < prob_available < 1.0:
            return "doubtful"
        if 0.0 < prob_available <= 0.5:
            return "questionable"

    if normalized_status == "out" or any(hint in news for hint in _INJURY_HINTS):
        return "injury"
    return "other"


def canonicalize_position(position: Optional[str]) -> Optional[str]:
    normalized = _normalize_text(position or "")
    if not normalized:
        return None
    return _POSITION_CANONICAL.get(normalized.upper(), _POSITION_CANONICAL.get((position or "").upper(), (position or "").upper()))


def _load_current_team_rows() -> list[dict[str, Any]]:
    global _TEAM_CACHE
    if _TEAM_CACHE is not None:
        return _TEAM_CACHE

    with TEAMS_PATH.open("r", encoding="utf-8") as f:
        teams = json.load(f)

    seasons = [str(team.get("season", "")) for team in teams if team.get("season")]
    current_season = max(seasons) if seasons else None

    current_teams = []
    for team in teams:
        if current_season is not None and str(team.get("season")) != current_season:
            continue
        if team.get("team_id") is None or not team.get("full_name") or not team.get("name"):
            continue
        current_teams.append(
            {
                "team_id": int(team["team_id"]),
                "code": team["name"],
                "full_name": team["full_name"],
                "season": str(team.get("season")) if team.get("season") else None,
            }
        )

    _TEAM_CACHE = current_teams
    return current_teams


def _team_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for team in _load_current_team_rows():
        alias_names = _TEAM_ALIASES_BY_CODE.get(team["code"], ())
        for key in (team["code"], team["full_name"], *alias_names):
            lookup[_normalize_text(key)] = team
    return lookup


def _team_lookup_by_id() -> dict[int, dict[str, Any]]:
    return {team["team_id"]: team for team in _load_current_team_rows()}


def resolve_team(team_ref: Any) -> Optional[dict[str, Any]]:
    if team_ref in (None, ""):
        return None

    if isinstance(team_ref, str):
        stripped = team_ref.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return _team_lookup_by_id().get(int(stripped))
        return _team_lookup().get(_normalize_text(stripped))

    if isinstance(team_ref, int):
        return _team_lookup_by_id().get(team_ref)

    return None


def team_id_for_name(team_name: Optional[str]) -> Optional[int]:
    team_meta = resolve_team(team_name)
    return int(team_meta["team_id"]) if team_meta else None


def _parse_fixture_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _find_fixture_by_teams(team_name: str, opponent_name: str, is_home: bool) -> Optional[dict[str, Any]]:
    expected_home = team_name if is_home else opponent_name
    expected_away = opponent_name if is_home else team_name

    matches = [
        fixture
        for fixture in load_current_fixtures()
        if fixture.get("home_team") == expected_home and fixture.get("away_team") == expected_away
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def derive_position_group(position: Optional[str]) -> Optional[str]:
    canonical = canonicalize_position(position)
    if not canonical:
        return None
    return _POSITION_GROUPS.get(canonical)


def _connect_db() -> sqlite3.Connection:
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


def _current_db_season(con: sqlite3.Connection) -> str:
    row = con.execute("SELECT MAX(season) AS season FROM team").fetchone()
    return str(row["season"])


def _load_current_db_players() -> list[dict[str, Any]]:
    global _DB_PLAYER_CACHE

    mtime = DB_PATH.stat().st_mtime
    if _DB_PLAYER_CACHE is not None and _DB_PLAYER_CACHE[0] == mtime:
        return _DB_PLAYER_CACHE[1]

    with _connect_db() as con:
        season = _current_db_season(con)
        rows = con.execute(
            """
            WITH latest_attr AS (
                SELECT pa.player_id, pa.team, pa.position
                FROM player_attributes pa
                JOIN (
                    SELECT player_id, MAX(gameweek) AS max_gameweek
                    FROM player_attributes
                    WHERE season = ?
                    GROUP BY player_id
                ) latest
                  ON latest.player_id = pa.player_id
                 AND latest.max_gameweek = pa.gameweek
                WHERE pa.season = ?
            )
            SELECT p.player_id, p.name, p.display_name, latest_attr.team, latest_attr.position
            FROM player p
            LEFT JOIN latest_attr ON latest_attr.player_id = p.player_id
            """,
            (season, season),
        ).fetchall()

    teams = _team_lookup()
    players = []
    for row in rows:
        team_code = row["team"]
        team_meta = teams.get(_normalize_text(team_code or ""), {})
        players.append(
            {
                "player_id": row["player_id"],
                "player_name": row["display_name"] or row["name"],
                "full_name": row["name"],
                "team": team_meta.get("full_name", team_code),
                "team_code": team_code,
                "position": canonicalize_position(row["position"]),
            }
        )

    _DB_PLAYER_CACHE = (mtime, players)
    return players


def _match_db_player(player_id: Any = None, player_name: Optional[str] = None, team: Optional[str] = None) -> Optional[dict[str, Any]]:
    players = _load_current_db_players()
    if player_id not in (None, 0, "0"):
        for player in players:
            if player["player_id"] == player_id:
                return player

    normalized_name = _normalize_text(player_name or "")
    if not normalized_name:
        return None

    normalized_team = _normalize_text(team or "")
    team_meta = _team_lookup().get(normalized_team) if normalized_team else None
    allowed_teams = {
        normalized_team,
        _normalize_text(team_meta["full_name"]) if team_meta else "",
        _normalize_text(team_meta["code"]) if team_meta else "",
    }

    matches = []
    for player in players:
        player_names = {
            _normalize_text(player.get("player_name") or ""),
            _normalize_text(player.get("full_name") or ""),
        }
        if normalized_name not in player_names and not any(
            normalized_name in name or name in normalized_name for name in player_names if name
        ):
            continue
        if normalized_team:
            player_teams = {
                _normalize_text(player.get("team") or ""),
                _normalize_text(player.get("team_code") or ""),
            }
            if not (player_teams & allowed_teams):
                continue
        matches.append(player)

    return matches[0] if len(matches) == 1 else None


def _recent_stats_from_db(player_id: int) -> dict[str, float]:
    return _player_usage_stats_from_db(player_id)


def _player_score_rows_from_db(player_id: int) -> list[sqlite3.Row]:
    with _connect_db() as con:
        season = _current_db_season(con)
        rows = con.execute(
            """
            SELECT ps.minutes, ps.goals, ps.assists, f.date, f.gameweek
            FROM player_score ps
            JOIN fixture f ON f.fixture_id = ps.fixture_id
            WHERE ps.player_id = ?
              AND f.season = ?
            ORDER BY f.gameweek DESC, COALESCE(f.date, '') DESC, ps.id DESC
            """,
            (player_id, season),
        ).fetchall()

    return list(rows)


def _count_team_matches_since(team_code: Optional[str], since_date: Optional[str]) -> Optional[float]:
    if not team_code or not since_date:
        return None

    since_dt = _parse_fixture_datetime(since_date)
    if since_dt is None:
        return None

    now = datetime.now(timezone.utc)
    matches = 0
    for fixture in load_current_fixtures():
        fixture_dt = _parse_fixture_datetime(fixture.get("date"))
        if fixture_dt is None or fixture_dt <= since_dt or fixture_dt > now:
            continue
        if team_code in {fixture.get("home_team_code"), fixture.get("away_team_code")}:
            matches += 1

    return float(matches)


def _player_usage_stats_from_db(player_id: int, team_code: Optional[str] = None) -> dict[str, float | str | None]:
    rows = _player_score_rows_from_db(player_id)

    minutes_last6 = float(sum(row["minutes"] for row in rows))
    minutes_season = float(sum(row["minutes"] for row in rows))

    last6_rows = rows[:6]
    last10_rows = rows[:10]
    minutes_last6 = float(sum(row["minutes"] for row in last6_rows))
    minutes_last10_before_absence = float(sum(row["minutes"] for row in last10_rows))

    goals_last6 = float(sum(row["goals"] for row in last6_rows))
    assists_last6 = float(sum(row["assists"] for row in last6_rows))
    goals_last10 = float(sum(row["goals"] for row in last10_rows))
    assists_last10 = float(sum(row["assists"] for row in last10_rows))
    total_goals = float(sum(row["goals"] for row in rows))
    total_assists = float(sum(row["assists"] for row in rows))

    stat_minutes = minutes_last6 or minutes_last10_before_absence or minutes_season
    stat_goals = goals_last6 if minutes_last6 else goals_last10 if minutes_last10_before_absence else total_goals
    stat_assists = assists_last6 if minutes_last6 else assists_last10 if minutes_last10_before_absence else total_assists

    sample_rows = last10_rows or rows[:10]
    starts = sum(1 for row in sample_rows if float(row["minutes"] or 0.0) >= 60.0)
    starter_probability = (starts / len(sample_rows)) if sample_rows else None

    latest_row = rows[0] if rows else None
    last_appearance_date = latest_row["date"] if latest_row else None
    matches_since_departure = _count_team_matches_since(team_code, last_appearance_date)

    if stat_minutes <= 0.0:
        return {
            "minutes_last6": minutes_last6,
            "minutes_season": minutes_season,
            "minutes_last10_before_absence": minutes_last10_before_absence,
            "goals90": 0.0,
            "assists90": 0.0,
            "starter_probability": starter_probability,
            "matches_since_departure": matches_since_departure,
            "last_appearance_date": last_appearance_date,
        }

    return {
        "minutes_last6": minutes_last6,
        "minutes_season": minutes_season,
        "minutes_last10_before_absence": minutes_last10_before_absence,
        "goals90": 90.0 * stat_goals / stat_minutes,
        "assists90": 90.0 * stat_assists / stat_minutes,
        "starter_probability": starter_probability,
        "matches_since_departure": matches_since_departure,
        "last_appearance_date": last_appearance_date,
    }


def _should_fill_numeric_stat(player: dict[str, Any], key: str, force_fill: bool = False) -> bool:
    value = player.get(key)
    if value is None:
        return True
    if force_fill and value == 0:
        return True
    if value == 0:
        identity_fields = (player.get("player_name"), player.get("team"), player.get("position"))
        if any(_is_placeholder_text(field) for field in identity_fields):
            return True
        if sum(1 for field in identity_fields if field) <= 1:
            return True
    return False


def sanitize_player_record(player: dict[str, Any]) -> dict[str, Any]:
    sanitized = dict(player)
    for key in (
        "player_name",
        "team",
        "position",
        "position_group",
        "absence_type",
        "status",
        "source_news",
        "last_updated",
        "last_appearance_date",
        "departure_date",
    ):
        sanitized[key] = sanitize_nullable_string(sanitized.get(key))

    if sanitized.get("player_id") in (0, "0", ""):
        sanitized["player_id"] = None

    sanitized["position"] = canonicalize_position(sanitized.get("position"))
    if not sanitized.get("position_group"):
        sanitized["position_group"] = derive_position_group(sanitized.get("position"))
    if not sanitized.get("absence_type"):
        sanitized["absence_type"] = classify_absence_type(
            sanitized.get("source_news"),
            sanitized.get("status"),
            sanitized.get("prob_available"),
        )
    return sanitized


def autofill_player_record_from_db(player: dict[str, Any]) -> dict[str, Any]:
    enriched = sanitize_player_record(player)
    force_fill_stats = (
        any(_is_placeholder_text(player.get(field)) for field in ("player_name", "team", "position"))
        or sum(1 for field in (enriched.get("player_name"), enriched.get("team"), enriched.get("position")) if field) <= 1
    )
    db_player = _match_db_player(
        player_id=enriched.get("player_id"),
        player_name=enriched.get("player_name"),
        team=enriched.get("team"),
    )
    if db_player is None:
        return enriched

    for key, value in (
        ("player_id", db_player.get("player_id")),
        ("player_name", db_player.get("player_name")),
        ("team", db_player.get("team")),
        ("position", db_player.get("position")),
    ):
        if not enriched.get(key) and value is not None:
            enriched[key] = value

    if not enriched.get("position_group"):
        enriched["position_group"] = derive_position_group(enriched.get("position"))

    stats = _player_usage_stats_from_db(db_player["player_id"], db_player.get("team_code"))
    for key in ("last_appearance_date",):
        value = stats.get(key)
        if not enriched.get(key) and value is not None:
            enriched[key] = value

    for key in (
        "minutes_last6",
        "minutes_season",
        "minutes_last10_before_absence",
        "goals90",
        "assists90",
        "starter_probability",
        "matches_since_departure",
    ):
        value = stats.get(key)
        if _should_fill_numeric_stat(enriched, key, force_fill=force_fill_stats):
            enriched[key] = value

    if not enriched.get("departure_date") and enriched.get("last_appearance_date"):
        enriched["departure_date"] = enriched["last_appearance_date"]

    return enriched


def autofill_player_records_from_db(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [autofill_player_record_from_db(player) for player in players]


def status_from_injury_news(chance_next_round: Any, news: Optional[str]) -> Optional[str]:
    news_lower = (news or "").lower()
    if "suspend" in news_lower:
        return "suspended"

    if chance_next_round is None:
        return None

    try:
        chance = int(chance_next_round)
    except (TypeError, ValueError):
        return None

    if chance >= 100:
        return "available"
    if chance >= 75:
        return "doubtful"
    if chance >= 50:
        return "questionable"
    if chance >= 25:
        return "major_doubt"
    return "out"


def load_current_injury_news() -> list[dict[str, Any]]:
    global _INJURY_CACHE

    mtime = INJURY_NEWS_PATH.stat().st_mtime
    db_mtime = DB_PATH.stat().st_mtime
    cache_key = max(mtime, db_mtime)
    if _INJURY_CACHE is not None and _INJURY_CACHE[0] == cache_key:
        return _INJURY_CACHE[1]

    with INJURY_NEWS_PATH.open("r", encoding="utf-8") as f:
        raw_rows = json.load(f)

    teams = _team_lookup()
    current_team_codes = {team["code"] for team in _load_current_team_rows()}
    last_updated = _file_last_updated(INJURY_NEWS_PATH)

    rows: list[dict[str, Any]] = []
    for row in raw_rows:
        team_code = row.get("team")
        if team_code not in current_team_codes:
            continue
        team_meta = teams.get(_normalize_text(team_code), {})
        chance = row.get("chance_next_round")
        db_player = _match_db_player(player_id=row.get("player_id"), player_name=row.get("name"), team=team_code)
        position = db_player.get("position") if db_player else None
        rows.append(
            {
                "player_id": row.get("player_id"),
                "player_name": row.get("name"),
                "team": team_meta.get("full_name", team_code),
                "team_code": team_code,
                "position": position,
                "position_group": derive_position_group(position),
                "prob_available": None if chance is None else max(0.0, min(1.0, float(chance) / 100.0)),
                "status": status_from_injury_news(chance, row.get("news")),
                "absence_type": classify_absence_type(
                    row.get("news"),
                    status_from_injury_news(chance, row.get("news")),
                    None if chance is None else max(0.0, min(1.0, float(chance) / 100.0)),
                ),
                "source_news": row.get("news"),
                "last_updated": last_updated,
            }
        )

    _INJURY_CACHE = (cache_key, rows)
    return rows


def _match_injury_row(
    player_id: Any = None,
    player_name: Optional[str] = None,
    team: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    rows = load_current_injury_news()

    if player_id is not None:
        for row in rows:
            if row.get("player_id") == player_id:
                return row

    if not player_name:
        return None

    normalized_name = _normalize_text(player_name)
    normalized_team = _normalize_text(team or "")
    team_meta = _team_lookup().get(normalized_team) if normalized_team else None
    allowed_teams = {
        normalized_team,
        _normalize_text(team_meta["full_name"]) if team_meta else "",
        _normalize_text(team_meta["code"]) if team_meta else "",
    }

    for row in rows:
        if _normalize_text(row.get("player_name") or "") != normalized_name:
            continue
        if not normalized_team:
            return row
        row_team_keys = {
            _normalize_text(row.get("team") or ""),
            _normalize_text(row.get("team_code") or ""),
        }
        if row_team_keys & allowed_teams:
            return row
    return None


def canonical_team_name(team_name: Optional[str]) -> Optional[str]:
    team_meta = resolve_team(team_name)
    return team_meta["full_name"] if team_meta else team_name


def team_code_for_name(team_name: Optional[str]) -> Optional[str]:
    team_meta = resolve_team(team_name)
    return team_meta["code"] if team_meta else None


def load_current_fixtures() -> list[dict[str, Any]]:
    global _FIXTURE_CACHE

    mtime = FIXTURES_PATH.stat().st_mtime
    if _FIXTURE_CACHE is not None and _FIXTURE_CACHE[0] == mtime:
        return _FIXTURE_CACHE[1]

    with FIXTURES_PATH.open("r", encoding="utf-8") as f:
        fixtures = json.load(f)

    current_season = max((team["season"] for team in _load_current_team_rows() if team.get("season")), default=None)
    teams = _team_lookup()
    rows = []
    for fixture in fixtures:
        if current_season is not None and str(fixture.get("season")) != current_season:
            continue
        home_meta = teams.get(_normalize_text(fixture.get("home_team") or ""), {})
        away_meta = teams.get(_normalize_text(fixture.get("away_team") or ""), {})
        rows.append(
            {
                "fixture_id": fixture.get("fixture_id"),
                "date": fixture.get("date"),
                "gameweek": fixture.get("gameweek"),
                "season": fixture.get("season"),
                "home_team": home_meta.get("full_name", fixture.get("home_team")),
                "home_team_id": home_meta.get("team_id"),
                "away_team": away_meta.get("full_name", fixture.get("away_team")),
                "away_team_id": away_meta.get("team_id"),
                "home_team_code": fixture.get("home_team"),
                "away_team_code": fixture.get("away_team"),
            }
        )

    _FIXTURE_CACHE = (mtime, rows)
    return rows


def resolve_fixture_context(
    fixture_id: Optional[int] = None,
    team_name: Optional[str] = None,
    opponent_name: Optional[str] = None,
    is_home: Optional[bool] = None,
) -> dict[str, Any]:
    team_name = canonical_team_name(team_name)
    opponent_name = canonical_team_name(opponent_name)

    if fixture_id is None:
        if not team_name or not opponent_name or is_home is None:
            raise ValueError("Provide fixture_id or the full team/opponent/is_home context")
        fixture = _find_fixture_by_teams(team_name, opponent_name, is_home)
        return {
            "fixture_id": fixture.get("fixture_id") if fixture else None,
            "date": fixture.get("date") if fixture else None,
            "gameweek": fixture.get("gameweek") if fixture else None,
            "season": fixture.get("season") if fixture else None,
            "team": team_name,
            "team_id": team_id_for_name(team_name),
            "opponent": opponent_name,
            "opponent_id": team_id_for_name(opponent_name),
            "is_home": is_home,
            "home_team": team_name if is_home else opponent_name,
            "home_team_id": team_id_for_name(team_name if is_home else opponent_name),
            "away_team": opponent_name if is_home else team_name,
            "away_team_id": team_id_for_name(opponent_name if is_home else team_name),
        }

    fixture = next((row for row in load_current_fixtures() if row.get("fixture_id") == fixture_id), None)
    if fixture is None:
        raise ValueError(f"Unknown fixture_id: {fixture_id}")

    home_team = fixture["home_team"]
    away_team = fixture["away_team"]

    if team_name:
        if team_name == home_team:
            resolved_team, resolved_opponent, resolved_is_home = home_team, away_team, True
        elif team_name == away_team:
            resolved_team, resolved_opponent, resolved_is_home = away_team, home_team, False
        else:
            raise ValueError(f"Team '{team_name}' does not belong to fixture_id {fixture_id}")
    elif opponent_name:
        if opponent_name == away_team:
            resolved_team, resolved_opponent, resolved_is_home = home_team, away_team, True
        elif opponent_name == home_team:
            resolved_team, resolved_opponent, resolved_is_home = away_team, home_team, False
        else:
            raise ValueError(f"Opponent '{opponent_name}' does not belong to fixture_id {fixture_id}")
    else:
        resolved_team, resolved_opponent, resolved_is_home = home_team, away_team, True

    if opponent_name and opponent_name != resolved_opponent:
        raise ValueError(f"Opponent '{opponent_name}' conflicts with fixture_id {fixture_id}")
    if is_home is not None and is_home != resolved_is_home:
        raise ValueError(f"is_home={is_home} conflicts with fixture_id {fixture_id}")

    return {
        "fixture_id": fixture_id,
        "date": fixture.get("date"),
        "gameweek": fixture.get("gameweek"),
        "season": fixture.get("season"),
        "team": resolved_team,
        "team_id": fixture.get("home_team_id") if resolved_is_home else fixture.get("away_team_id"),
        "opponent": resolved_opponent,
        "opponent_id": fixture.get("away_team_id") if resolved_is_home else fixture.get("home_team_id"),
        "is_home": resolved_is_home,
        "home_team": home_team,
        "home_team_id": fixture.get("home_team_id"),
        "away_team": away_team,
        "away_team_id": fixture.get("away_team_id"),
    }


def get_next_team_fixtures(
    team_ref: Any,
    limit: int = 3,
    reference_time: Optional[datetime] = None,
) -> list[dict[str, Any]]:
    team_meta = resolve_team(team_ref)
    if team_meta is None:
        raise ValueError(f"Unknown team: {team_ref}")

    ref = reference_time or datetime.now(timezone.utc)
    team_id = int(team_meta["team_id"])

    future_dated: list[dict[str, Any]] = []
    undated: list[dict[str, Any]] = []
    for fixture in load_current_fixtures():
        participates = team_id in {fixture.get("home_team_id"), fixture.get("away_team_id")}
        if not participates:
            continue

        fixture_dt = _parse_fixture_datetime(fixture.get("date"))
        if fixture_dt is None:
            undated.append(fixture)
        elif fixture_dt >= ref:
            future_dated.append(fixture)

    future_dated.sort(key=lambda fixture: (_parse_fixture_datetime(fixture.get("date")), fixture.get("fixture_id") or 0))
    undated.sort(key=lambda fixture: (fixture.get("gameweek") is None, fixture.get("gameweek") or 10**9, fixture.get("fixture_id") or 0))
    return (future_dated + undated)[:limit]


def get_team_injury_players(team_name: str) -> list[dict[str, Any]]:
    canonical_team = canonical_team_name(team_name)
    players = [row for row in load_current_injury_news() if row.get("team") == canonical_team]
    return enrich_player_records(players)


def _player_merge_key(player: dict[str, Any]) -> tuple[Any, str, str]:
    if player.get("player_id") is not None:
        return (player.get("player_id"), "", "")
    return (
        None,
        _normalize_text(player.get("player_name") or ""),
        _normalize_text(player.get("team") or ""),
    )


def merge_player_records(primary_players: list[dict[str, Any]], supplemental_players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[Any, str, str], dict[str, Any]] = {}

    for player in supplemental_players:
        merged[_player_merge_key(player)] = dict(player)

    for player in primary_players:
        key = _player_merge_key(player)
        existing = merged.get(key, {})
        combined = dict(existing)
        for field, value in player.items():
            if value not in (None, "", []):
                combined[field] = value
        merged[key] = combined

    return list(merged.values())


def saturate_team_players(team_name: str, provided_players: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    provided = provided_players or []
    canonical_provided = [
        {**player, "team": canonical_team_name(player.get("team") or team_name)}
        for player in provided
    ]
    auto_players = get_team_injury_players(team_name)
    return enrich_player_records(merge_player_records(canonical_provided, auto_players))


def enrich_player_record(player: dict[str, Any]) -> dict[str, Any]:
    enriched = autofill_player_record_from_db(player)
    injury_row = _match_injury_row(
        player_id=enriched.get("player_id"),
        player_name=enriched.get("player_name"),
        team=enriched.get("team"),
    )

    if injury_row is not None:
        for key in ("player_name", "source_news", "last_updated", "prob_available", "status", "position", "absence_type"):
            if not enriched.get(key) and injury_row.get(key) is not None:
                enriched[key] = injury_row[key]

        if not enriched.get("team") and injury_row.get("team"):
            enriched["team"] = injury_row["team"]

    team_meta = _team_lookup().get(_normalize_text(enriched.get("team") or "")) if enriched.get("team") else None
    if team_meta:
        enriched["team"] = team_meta["full_name"]

    enriched["position"] = canonicalize_position(enriched.get("position"))
    if not enriched.get("position_group"):
        enriched["position_group"] = derive_position_group(enriched.get("position"))
    enriched["absence_type"] = classify_absence_type(
        enriched.get("source_news"),
        enriched.get("status"),
        enriched.get("prob_available"),
    )
    if enriched["absence_type"] in {"loan_out", "transfer_out"} and not enriched.get("departure_date"):
        enriched["departure_date"] = enriched.get("last_appearance_date") or enriched.get("last_updated")

    return enriched


def enrich_player_records(players: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [enrich_player_record(player) for player in players]
