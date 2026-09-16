from __future__ import annotations

from typing import Any, Mapping


class CopilotBlendFallback:
    def __init__(self, *, top_n: int = 10) -> None:
        self.top_n = top_n

    def build_fallback_payload(
        self,
        *,
        schema_version: str,
        correlation_id: str,
        model_context: dict[str, Any],
    ) -> dict[str, Any]:
        weights = model_context.get("weights", {})
        elo_weight = float(weights.get("elo", 0.5))
        airsenal_weight = float(weights.get("airsenal", 0.5))

        blended_players = model_context.get("blended_players", [])

        scored = []
        for p in blended_players:
            elo_score = float(p.get("elo_score", 0.0))
            airsenal_pts = float(p.get("airsenal_predicted_points", 0.0))
            composite = elo_weight * elo_score + airsenal_weight * airsenal_pts
            scored.append({
                "player_id": p["player_id"],
                "player_name": p["player_name"],
                "team": p.get("team", ""),
                "position": p.get("position", ""),
                "elo_score": elo_score,
                "airsenal_predicted_points": airsenal_pts,
                "composite_score": composite,
            })

        scored.sort(key=lambda x: (-x["composite_score"], x["player_id"]))
        top_players = scored[: self.top_n]

        transfer_recommendations = []
        for idx, player in enumerate(top_players[:3]):
            transfer_recommendations.append({
                "transfer_id": f"fb-transfer-{idx + 1}",
                "out": {"player_id": 0, "player_name": "TBD"},
                "in": {
                    "player_id": player["player_id"],
                    "player_name": player["player_name"],
                },
                "reason": (
                    f"Composite score {player['composite_score']:.1f} "
                    f"(ELO={player['elo_score']:.1f}, AIrsenal={player['airsenal_predicted_points']:.1f})"
                ),
                "projected_points_delta": player["airsenal_predicted_points"],
            })

        captain = top_players[0] if top_players else None
        captain_text = (
            f"Captain {captain['player_name']} (composite {captain['composite_score']:.1f})"
            if captain
            else "No captain recommendation available"
        )

        rationale_parts = [
            f"Weighted blend: ELO {elo_weight:.0%}, AIrsenal {airsenal_weight:.0%}",
            f"Top pick: {top_players[0]['player_name']} (composite {top_players[0]['composite_score']:.1f})"
            if top_players
            else "No player data available",
        ]

        return {
            "schema_version": schema_version,
            "correlation_id": correlation_id,
            "core": {
                "summary": f"Serving fallback recommendations ({len(top_players)} players ranked by weighted blend).",
                "confidence": 0.4,
            },
            "recommended_transfers": transfer_recommendations,
            "ask_copilot": {
                "answer": captain_text,
                "rationale": rationale_parts,
                "confidence": 0.4,
            },
            "degraded_mode": {
                "is_degraded": True,
                "code": "FALLBACK",
                "message": "LLM unavailable — serving weighted-average fallback.",
                "fallback_used": True,
            },
        }
