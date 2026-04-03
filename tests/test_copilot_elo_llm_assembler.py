from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_elo_llm_assembler import CopilotEloLlmAssembler


def _make_elo_scores():
    return [
        {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0},
        {"player_id": 202, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5},
        {"player_id": 303, "player_name": "Palmer", "team": "CHE", "position": "MID", "elo_score": 1580.0},
    ]


def _make_airsenal_predictions():
    return [
        {"player_id": 101, "player_name": "Saka", "predicted_points": 8.0},
        {"player_id": 202, "player_name": "Haaland", "predicted_points": 11.0},
        {"player_id": 404, "player_name": "Salah", "predicted_points": 9.5},
    ]


def test_happy_path_blend_elo_07_airsenal_03():
    """Both sources present, weights 0.7/0.3, players merged by player_id."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=_make_elo_scores()),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=_make_airsenal_predictions()),
    ):
        context = assembler.assemble_model_context(
            source_weights={"elo": 0.7, "airsenal": 0.3},
        )

    assert context["schema_version"] == "1.0"
    assert context["weights"] == {"elo": 0.7, "airsenal": 0.3}
    assert context["sources"] == ["elo", "airsenal"]
    assert context["player_name_contains"] is None

    players = context["blended_players"]
    # All 4 unique players from both sources
    assert len(players) == 4

    # Haaland: elo=1850.5, airsenal=11.0
    haaland = next(p for p in players if p["player_id"] == 202)
    assert haaland["player_name"] == "Haaland"
    assert haaland["team"] == "MCI"
    assert haaland["position"] == "FWD"
    assert haaland["elo_score"] == 1850.5
    assert haaland["airsenal_predicted_points"] == 11.0

    # Salah: only in airsenal, elo_score should be 0
    salah = next(p for p in players if p["player_id"] == 404)
    assert salah["player_name"] == "Salah"
    assert salah["elo_score"] == 0
    assert salah["airsenal_predicted_points"] == 9.5

    # Saka: elo=1650.0, airsenal=8.0
    saka = next(p for p in players if p["player_id"] == 101)
    assert saka["elo_score"] == 1650.0
    assert saka["airsenal_predicted_points"] == 8.0


def test_missing_player_in_one_source_score_zero():
    """Player exists only in ELO source — airsenal_predicted_points = 0."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    elo_only = [
        {"player_id": 999, "player_name": "Unknown", "team": "LIV", "position": "FWD", "elo_score": 1700.0},
    ]

    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=elo_only),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=[]),
    ):
        context = assembler.assemble_model_context(
            source_weights={"elo": 0.5, "airsenal": 0.5},
        )

    assert len(context["blended_players"]) == 1
    player = context["blended_players"][0]
    assert player["player_id"] == 999
    assert player["elo_score"] == 1700.0
    assert player["airsenal_predicted_points"] == 0


def test_missing_player_in_elo_source_score_zero():
    """Player exists only in AIrsenal source — elo_score = 0."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    airsenal_only = [
        {"player_id": 888, "player_name": "Ghost", "predicted_points": 7.5},
    ]

    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=[]),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=airsenal_only),
    ):
        context = assembler.assemble_model_context(
            source_weights={"elo": 0.5, "airsenal": 0.5},
        )

    assert len(context["blended_players"]) == 1
    player = context["blended_players"][0]
    assert player["player_id"] == 888
    assert player["elo_score"] == 0
    assert player["airsenal_predicted_points"] == 7.5


def test_invalid_weight_rejected_missing_source():
    """Weight validation rejects missing required source keys."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with pytest.raises(ValueError, match="must include"):
        assembler.assemble_model_context(
            source_weights={"elo": 0.5},
        )


def test_invalid_weight_rejected_sum_not_one():
    """Weight validation rejects weights that don't sum to 1.0."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with pytest.raises(ValueError, match="must sum to 1.0"):
        assembler.assemble_model_context(
            source_weights={"elo": 0.8, "airsenal": 0.4},
        )


def test_invalid_weight_rejected_out_of_range():
    """Weight validation rejects weights outside 0.0–1.0."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        assembler.assemble_model_context(
            source_weights={"elo": 1.5, "airsenal": -0.5},
        )


def test_player_name_contains_filter():
    """player_name_contains filters blended_players by substring (case-insensitive)."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=_make_elo_scores()),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=_make_airsenal_predictions()),
    ):
        context = assembler.assemble_model_context(
            source_weights={"elo": 0.7, "airsenal": 0.3},
            player_name_contains="a",
        )

    assert context["player_name_contains"] == "a"
    # Saka, Haaland, Palmer, Salah all contain 'a'
    assert len(context["blended_players"]) == 4

    # Filter for 'ha' should match Haaland only
    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=_make_elo_scores()),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=_make_airsenal_predictions()),
    ):
        context2 = assembler.assemble_model_context(
            source_weights={"elo": 0.7, "airsenal": 0.3},
            player_name_contains="ha",
        )

    assert len(context2["blended_players"]) == 1
    assert context2["blended_players"][0]["player_name"] == "Haaland"


def test_schema_structure_matches_contract():
    """Return schema exactly matches the contract expected by Gemini adapter."""
    assembler = CopilotEloLlmAssembler(db_path=":memory:")

    with (
        patch.object(assembler.elo_scorer, "get_player_elo_scores", return_value=_make_elo_scores()),
        patch.object(assembler.airsenal_extractor, "get_player_predictions", return_value=_make_airsenal_predictions()),
    ):
        context = assembler.assemble_model_context(
            source_weights={"elo": 0.7, "airsenal": 0.3},
        )

    # Top-level keys
    assert set(context.keys()) == {
        "schema_version", "weights", "sources", "player_name_contains", "blended_players"
    }

    # Player-level keys
    player = context["blended_players"][0]
    assert set(player.keys()) == {
        "player_id", "player_name", "team", "position",
        "elo_score", "airsenal_predicted_points",
    }
