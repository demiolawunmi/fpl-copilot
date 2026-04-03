from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


class CopilotAirsenalExtractor:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def _detect_latest_tag(self) -> str | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT MAX(tag) AS tag FROM player_prediction;"
            ).fetchone()
            if row is None:
                return None
            return row["tag"]

    def get_player_predictions(self, gameweek: int | None = None) -> list[dict[str, Any]]:
        tag = self._detect_latest_tag()
        if tag is None:
            return []

        with self._connect() as con:
            rows = con.execute(
                """
                SELECT
                    pp.player_id,
                    COALESCE(p.display_name, p.name) AS player_name,
                    SUM(pp.predicted_points) AS predicted_points
                FROM player_prediction pp
                JOIN player p ON pp.player_id = p.player_id
                WHERE pp.tag = ?
                GROUP BY pp.player_id
                ORDER BY predicted_points DESC;
                """,
                (tag,),
            ).fetchall()

        return [
            {
                "player_id": row["player_id"],
                "player_name": row["player_name"],
                "predicted_points": float(row["predicted_points"]),
            }
            for row in rows
        ]
