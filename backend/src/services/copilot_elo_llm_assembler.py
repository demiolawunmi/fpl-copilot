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
        bank: float | None = None,
        free_transfers: int | None = None,
        current_squad: list[dict[str, Any]] | None = None,
        fpl_team_id: int | None = None,
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

        price_by_id = self.elo_scorer.get_player_prices_for_ids(
            sorted(all_player_ids),
            gameweek=gameweek,
        )

        blended_players: list[dict[str, Any]] = []
        for pid in sorted(all_player_ids):
            elo_entry = elo_by_id.get(pid, {})
            airsenal_entry = airsenal_by_id.get(pid, {})

            player_name = elo_entry.get("player_name") or airsenal_entry.get("player_name", "")
            fpl_api_id = elo_entry.get("fpl_api_id")
            if fpl_api_id is None:
                fpl_api_id = airsenal_entry.get("fpl_api_id")
            team = elo_entry.get("team", "")
            position = elo_entry.get("position", "")
            elo_score = elo_entry.get("elo_score", 0)
            airsenal_predicted_points = airsenal_entry.get("predicted_points", 0)
            price_m = float(price_by_id.get(pid, 0.0))

            blended_players.append({
                "player_id": pid,
                "fpl_api_id": fpl_api_id,
                "player_name": player_name,
                "team": team,
                "position": position,
                "price": price_m,
                "elo_score": elo_score,
                "airsenal_predicted_points": airsenal_predicted_points,
            })

        if player_name_contains:
            filter_lower = player_name_contains.lower()
            blended_players = [
                p for p in blended_players
                if filter_lower in p["player_name"].lower()
            ]

        blended_by_fpl_id = {
            int(p["fpl_api_id"]): p
            for p in blended_players
            if p.get("fpl_api_id") is not None
        }
        normalized_current_squad: list[dict[str, Any]] = []
        for squad_player in current_squad or []:
            fpl_api_id = squad_player.get("fpl_api_id")
            if fpl_api_id is None:
                continue
            try:
                normalized_fpl_api_id = int(fpl_api_id)
            except (TypeError, ValueError):
                continue

            matched_blended = blended_by_fpl_id.get(normalized_fpl_api_id, {})
            squad_price = float(squad_player.get("price", 0.0) or 0.0)
            blended_price = float(matched_blended.get("price", 0.0) or 0.0)
            normalized_current_squad.append({
                "player_id": matched_blended.get("player_id"),
                "fpl_api_id": normalized_fpl_api_id,
                "player_name": squad_player.get("player_name") or matched_blended.get("player_name", ""),
                "team": squad_player.get("team") or matched_blended.get("team", ""),
                "position": squad_player.get("position") or matched_blended.get("position", ""),
                "price": squad_price if squad_price > 0 else blended_price,
                "x_pts": float(squad_player.get("x_pts", 0.0) or 0.0),
                "elo_score": float(matched_blended.get("elo_score", 0.0) or 0.0),
                "airsenal_predicted_points": float(matched_blended.get("airsenal_predicted_points", 0.0) or 0.0),
            })

        airsenal_optimization: dict[str, Any] = {"available": False, "reason": "fpl_team_id_not_provided"}
        if fpl_team_id is not None:
            from src.services.copilot_optimization_context import load_optimization_for_team

            airsenal_optimization = load_optimization_for_team(
                self.db_path,
                fpl_team_id=int(fpl_team_id),
                target_gameweek=gameweek,
            )

        return {
            "schema_version": "1.0",
            "weights": {
                "elo": weights["elo"],
                "airsenal": weights["airsenal"],
            },
            "gameweek": gameweek,
            "bank": None if bank is None else float(bank),
            "free_transfers": None if free_transfers is None else int(free_transfers),
            "fpl_team_id": fpl_team_id,
            "sources": list(_EXPECTED_SOURCES),
            "player_name_contains": player_name_contains,
            "current_squad": normalized_current_squad,
            "blended_players": blended_players,
            "airsenal_optimization": airsenal_optimization,
        }
