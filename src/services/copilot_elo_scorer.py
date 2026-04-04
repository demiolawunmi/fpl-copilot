"""Copilot ELO scorer — derives player scores from team ELO × position factors."""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_POSITION_FACTORS: Dict[str, float] = {
    "FWD": 1.2,
    "MID": 1.1,
    "DEF": 0.9,
    "GK": 0.8,
}

_AIRSENAL_TO_FULL_NAME: Dict[str, str] = {
    "ARS": "Arsenal",
    "AVL": "Aston Villa",
    "BHA": "Brighton & Hove Albion",
    "BOU": "Bournemouth",
    "BRE": "Brentford",
    "BUR": "Burnley",
    "CHE": "Chelsea",
    "CRY": "Crystal Palace",
    "EVE": "Everton",
    "FUL": "Fulham",
    "LEE": "Leeds United",
    "LIV": "Liverpool",
    "MCI": "Manchester City",
    "MUN": "Manchester United",
    "NEW": "Newcastle United",
    "NFO": "Nottingham Forest",
    "SUN": "Sunderland",
    "TOT": "Tottenham Hotspur",
    "WHU": "West Ham United",
    "WOL": "Wolverhampton Wanderers",
}


class CopilotEloScorer:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = str(db_path) if db_path else _default_db_path()
        self._elo_ratings: Optional[Dict[str, float]] = None

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _fetch_elo_ratings(self) -> Dict[str, float]:
        if self._elo_ratings is not None:
            return self._elo_ratings
        from src.services.club_elo import fetch_elo_ratings
        self._elo_ratings = fetch_elo_ratings()
        return self._elo_ratings

    def _resolve_team_elo(self, ratings: Dict[str, float], team_code: str) -> Optional[float]:
        from src.services.club_elo import resolve_team_elo
        full_name = _AIRSENAL_TO_FULL_NAME.get(team_code, team_code)
        return resolve_team_elo(ratings, full_name)

    def get_player_elo_scores(self, gameweek: Optional[int] = None) -> List[Dict[str, Any]]:
        ratings = self._fetch_elo_ratings()
        if not ratings:
            return []

        with self._connect() as con:
            if gameweek is not None:
                rows = con.execute(
                    """
                    SELECT p.player_id, p.fpl_api_id, p.display_name, pa.team, pa.position
                    FROM player_attributes pa
                    JOIN player p ON pa.player_id = p.player_id
                    WHERE pa.gameweek = ?
                    ORDER BY p.player_id;
                    """,
                    (gameweek,),
                ).fetchall()
            else:
                max_gw_row = con.execute(
                    "SELECT MAX(gameweek) AS max_gw FROM player_attributes;"
                ).fetchone()
                max_gw = max_gw_row["max_gw"] if max_gw_row else None
                if max_gw is None:
                    return []
                rows = con.execute(
                    """
                    SELECT p.player_id, p.fpl_api_id, p.display_name, pa.team, pa.position
                    FROM player_attributes pa
                    JOIN player p ON pa.player_id = p.player_id
                    WHERE pa.gameweek = ?
                    ORDER BY p.player_id;
                    """,
                    (max_gw,),
                ).fetchall()

        results: List[Dict[str, Any]] = []
        for row in rows:
            position = row["position"]
            factor = _POSITION_FACTORS.get(position)
            if factor is None:
                continue

            team_elo = self._resolve_team_elo(ratings, row["team"])
            if team_elo is None:
                continue

            results.append({
                "player_id": row["player_id"],
                "fpl_api_id": row["fpl_api_id"],
                "player_name": row["display_name"] or "",
                "team": row["team"],
                "position": position,
                "elo_score": round(team_elo * factor, 2),
            })

        return results

    def get_player_prices_for_ids(
        self,
        player_ids: List[int],
        gameweek: Optional[int] = None,
    ) -> Dict[int, float]:
        """Return player_id -> current FPL price in £m for the given (or latest) gameweek.

        AIrsenal stores ``player_attributes.price`` in tenths of a million (e.g. 105 → £10.5m).
        """
        if not player_ids:
            return {}

        try:
            with self._connect() as con:
                if gameweek is not None:
                    gw = gameweek
                else:
                    max_gw_row = con.execute(
                        "SELECT MAX(gameweek) AS max_gw FROM player_attributes;"
                    ).fetchone()
                    gw = max_gw_row["max_gw"] if max_gw_row else None
                if gw is None:
                    return {}

                placeholders = ",".join("?" * len(player_ids))
                rows = con.execute(
                    f"""
                    SELECT player_id, MAX(price) AS price
                    FROM player_attributes
                    WHERE gameweek = ? AND player_id IN ({placeholders})
                    GROUP BY player_id;
                    """,
                    (gw, *player_ids),
                ).fetchall()
        except sqlite3.OperationalError:
            return {}

        return {
            int(row["player_id"]): round(float(row["price"]) / 10.0, 1)
            for row in rows
            if row["price"] is not None
        }


def _default_db_path() -> str:
    current = Path(__file__).resolve().parent
    for candidate in (current, *current.parents):
        db = candidate / "data" / "airsenal" / "data.db"
        if db.is_file():
            return str(db)
    raise FileNotFoundError("Could not locate AIrsenal data.db")
