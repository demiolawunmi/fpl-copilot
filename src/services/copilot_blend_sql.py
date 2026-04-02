from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Mapping


_EXPECTED_SOURCES = ("fplcopilot", "airsenal")


def _validate_source_weights(source_weights: Mapping[str, float]) -> dict[str, float]:
    keys = set(source_weights.keys())
    expected = set(_EXPECTED_SOURCES)
    if keys != expected:
        raise ValueError("source_weights must include exactly: fplcopilot and airsenal")

    normalized: dict[str, float] = {}
    for source in _EXPECTED_SOURCES:
        raw = source_weights[source]
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"source_weights.{source} must be a number") from exc
        if value < 0.0 or value > 1.0:
            raise ValueError(f"source_weights.{source} must be between 0.0 and 1.0")
        normalized[source] = value

    total = normalized["fplcopilot"] + normalized["airsenal"]
    if abs(total - 1.0) > 1e-9:
        raise ValueError("source_weights must sum to 1.0")

    return normalized


class CopilotSqlBlendAssembler:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.db_path)
        con.row_factory = sqlite3.Row
        return con

    def assemble_model_context(
        self,
        *,
        source_weights: Mapping[str, float],
        player_name_contains: str | None = None,
        limit: int = 25,
    ) -> dict[str, Any]:
        if limit <= 0:
            raise ValueError("limit must be greater than 0")

        weights = _validate_source_weights(source_weights)
        normalized_filter = player_name_contains if player_name_contains else None

        with self._connect() as con:
            rows = con.execute(
                """
                WITH source_rows AS (
                    SELECT
                        source,
                        player_id,
                        player_name,
                        projected_points
                    FROM copilot_source_player_scores
                    WHERE source IN (?, ?)
                      AND (? IS NULL OR instr(lower(player_name), lower(?)) > 0)
                )
                SELECT
                    player_id,
                    player_name,
                    ROUND(
                        SUM(
                            CASE source
                                WHEN ? THEN projected_points * ?
                                WHEN ? THEN projected_points * ?
                                ELSE 0.0
                            END
                        ),
                        6
                    ) AS blended_projected_points,
                    ROUND(SUM(CASE source WHEN ? THEN projected_points ELSE 0.0 END), 6) AS fplcopilot_projected_points,
                    ROUND(SUM(CASE source WHEN ? THEN projected_points ELSE 0.0 END), 6) AS airsenal_projected_points
                FROM source_rows
                GROUP BY player_id, player_name
                ORDER BY blended_projected_points DESC, player_id ASC, player_name ASC
                LIMIT ?;
                """,
                (
                    "fplcopilot",
                    "airsenal",
                    normalized_filter,
                    normalized_filter,
                    "fplcopilot",
                    weights["fplcopilot"],
                    "airsenal",
                    weights["airsenal"],
                    "fplcopilot",
                    "airsenal",
                    limit,
                ),
            ).fetchall()

        blended_players = [
            {
                "player_id": int(row["player_id"]),
                "player_name": str(row["player_name"]),
                "blended_projected_points": float(row["blended_projected_points"]),
                "source_points": {
                    "fplcopilot": float(row["fplcopilot_projected_points"]),
                    "airsenal": float(row["airsenal_projected_points"]),
                },
            }
            for row in rows
        ]

        return {
            "schema_version": "1.0",
            "weights": {
                "fplcopilot": weights["fplcopilot"],
                "airsenal": weights["airsenal"],
            },
            "sources": ["fplcopilot", "airsenal"],
            "player_name_contains": normalized_filter,
            "blended_players": blended_players,
        }
