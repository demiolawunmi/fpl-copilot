from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


_EXPECTED_SOURCES = ("elo", "airsenal")


def _validate_source_weights(source_weights: Mapping[str, float]) -> dict[str, float]:
    keys = set(source_weights.keys())
    expected = set(_EXPECTED_SOURCES)
    if keys != expected:
        raise ValueError(f"source_weights must include exactly: {', '.join(sorted(expected))}")

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

    total = sum(normalized.values())
    if abs(total - 1.0) > 1e-9:
        raise ValueError("source_weights must sum to 1.0")

    return normalized


class CopilotEloLlmAssembler:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        self._elo_scorer = None
        self._airsenal_extractor = None

    @property
    def elo_scorer(self):
        if self._elo_scorer is None:
            from src.services.copilot_elo_scorer import CopilotEloScorer
            self._elo_scorer = CopilotEloScorer(self.db_path)
        return self._elo_scorer

    @property
    def airsenal_extractor(self):
        if self._airsenal_extractor is None:
            from src.services.copilot_airsenal_extractor import CopilotAirsenalExtractor
            self._airsenal_extractor = CopilotAirsenalExtractor(self.db_path)
        return self._airsenal_extractor

    def assemble_model_context(
        self,
        *,
        source_weights: Mapping[str, float],
        player_name_contains: str | None = None,
        gameweek: int | None = None,
    ) -> dict[str, Any]:
        weights = _validate_source_weights(source_weights)

        elo_scores = self.elo_scorer.get_player_elo_scores(gameweek=gameweek)
        airsenal_predictions = self.airsenal_extractor.get_player_predictions(gameweek=gameweek)

        elo_by_id: dict[int, dict[str, Any]] = {
            p["player_id"]: p for p in elo_scores
        }
        airsenal_by_id: dict[int, dict[str, Any]] = {
            p["player_id"]: p for p in airsenal_predictions
        }

        all_player_ids = set(elo_by_id.keys()) | set(airsenal_by_id.keys())

        blended_players: list[dict[str, Any]] = []
        for pid in sorted(all_player_ids):
            elo_entry = elo_by_id.get(pid, {})
            airsenal_entry = airsenal_by_id.get(pid, {})

            player_name = elo_entry.get("player_name") or airsenal_entry.get("player_name", "")
            team = elo_entry.get("team", "")
            position = elo_entry.get("position", "")
            elo_score = elo_entry.get("elo_score", 0)
            airsenal_predicted_points = airsenal_entry.get("predicted_points", 0)

            blended_players.append({
                "player_id": pid,
                "player_name": player_name,
                "team": team,
                "position": position,
                "elo_score": elo_score,
                "airsenal_predicted_points": airsenal_predicted_points,
            })

        if player_name_contains:
            filter_lower = player_name_contains.lower()
            blended_players = [
                p for p in blended_players
                if filter_lower in p["player_name"].lower()
            ]

        return {
            "schema_version": "1.0",
            "weights": {
                "elo": weights["elo"],
                "airsenal": weights["airsenal"],
            },
            "sources": list(_EXPECTED_SOURCES),
            "player_name_contains": player_name_contains,
            "blended_players": blended_players,
        }
