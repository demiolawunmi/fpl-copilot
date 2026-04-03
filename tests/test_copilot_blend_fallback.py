from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_blend_fallback import CopilotBlendFallback


_MODEL_CONTEXT = {
    "schema_version": "1.0",
    "weights": {"elo": 0.7, "airsenal": 0.3},
    "sources": ["elo", "airsenal"],
    "blended_players": [
        {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
        {"player_id": 202, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5, "airsenal_predicted_points": 11.0},
        {"player_id": 303, "player_name": "Palmer", "team": "CHE", "position": "MID", "elo_score": 1580.0, "airsenal_predicted_points": 7.5},
        {"player_id": 404, "player_name": "Salah", "team": "LIV", "position": "MID", "elo_score": 1700.0, "airsenal_predicted_points": 9.5},
    ],
}


def test_happy_path_returns_valid_fallback_payload():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-fallback",
        model_context=_MODEL_CONTEXT,
    )

    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-fallback"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "FALLBACK"
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["core"]["confidence"] == 0.4
    assert result["ask_copilot"]["confidence"] == 0.4


def test_top_players_ranked_by_composite_score():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-rank",
        model_context=_MODEL_CONTEXT,
    )

    transfers = result["recommended_transfers"]
    assert len(transfers) == 3

    first_in = transfers[0]["in"]
    assert first_in["player_name"] == "Haaland"


def test_composite_score_calculation():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-composite",
        model_context=_MODEL_CONTEXT,
    )

    transfers = result["recommended_transfers"]
    haaland_transfer = next(t for t in transfers if t["in"]["player_name"] == "Haaland")
    expected_composite = 0.7 * 1850.5 + 0.3 * 11.0
    assert f"{expected_composite:.1f}" in haaland_transfer["reason"]


def test_respects_top_n_limit():
    fallback = CopilotBlendFallback(top_n=2)

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-topn",
        model_context=_MODEL_CONTEXT,
    )

    assert len(result["recommended_transfers"]) == 2


def test_empty_players_returns_safe_payload():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-empty",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 0.5, "airsenal": 0.5},
            "sources": ["elo", "airsenal"],
            "blended_players": [],
        },
    )

    assert result["recommended_transfers"] == []
    assert "No captain recommendation available" in result["ask_copilot"]["answer"]


def test_elo_only_weighting():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-elo-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 1.0, "airsenal": 0.0},
            "sources": ["elo"],
            "blended_players": [
                {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 0.0},
            ],
        },
    )

    assert result["degraded_mode"]["is_degraded"] is True
    assert len(result["recommended_transfers"]) == 1
    assert result["recommended_transfers"][0]["in"]["player_name"] == "Saka"


def test_airsenal_only_weighting():
    fallback = CopilotBlendFallback()

    result = fallback.build_fallback_payload(
        schema_version="1.0",
        correlation_id="corr-airsenal-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 0.0, "airsenal": 1.0},
            "sources": ["airsenal"],
            "blended_players": [
                {"player_id": 404, "player_name": "Salah", "team": "LIV", "position": "MID", "elo_score": 0.0, "airsenal_predicted_points": 9.5},
            ],
        },
    )

    assert len(result["recommended_transfers"]) == 1
    assert result["recommended_transfers"][0]["in"]["player_name"] == "Salah"
