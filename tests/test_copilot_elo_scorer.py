"""Tests for CopilotEloScorer — team ELO → player scores via position factors."""

from pathlib import Path
import sqlite3
import sys
from unittest.mock import patch

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_elo_scorer import CopilotEloScorer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _seed_airsenal_db(db_path: Path) -> None:
    """Create a minimal AIrsenal-style DB with player + player_attributes."""
    con = sqlite3.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE player (
            player_id INTEGER NOT NULL,
            fpl_api_id INTEGER,
            name VARCHAR(100) NOT NULL,
            display_name VARCHAR(100),
            opta_code VARCHAR,
            PRIMARY KEY (player_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE player_attributes (
            id INTEGER NOT NULL,
            player_id INTEGER,
            season VARCHAR(100) NOT NULL,
            gameweek INTEGER NOT NULL,
            price INTEGER NOT NULL,
            team VARCHAR(100) NOT NULL,
            position VARCHAR(100) NOT NULL,
            chance_of_playing_next_round INTEGER,
            news VARCHAR(100),
            return_gameweek INTEGER,
            transfers_balance INTEGER,
            selected INTEGER,
            transfers_in INTEGER,
            transfers_out INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(player_id) REFERENCES player (player_id)
        );
        """
    )
    con.executemany(
        "INSERT INTO player (player_id, fpl_api_id, name, display_name, opta_code) VALUES (?, ?, ?, ?, ?);",
        [
            (1, 101, "Erling Haaland", "Haaland", "p123"),
            (2, 202, "Bukayo Saka", "Saka", "p456"),
            (3, 303, "Virgil van Dijk", "van Dijk", "p789"),
            (4, 404, "Alisson Becker", "Alisson", "p012"),
            (5, 505, "Cole Palmer", "Palmer", "p345"),
        ],
    )
    con.executemany(
        """
        INSERT INTO player_attributes
            (id, player_id, season, gameweek, price, team, position)
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        [
            (1, 1, "2526", 1, 150, "MCI", "FWD"),
            (2, 2, "2526", 1, 100, "ARS", "MID"),
            (3, 3, "2526", 1, 65,  "LIV", "DEF"),
            (4, 4, "2526", 1, 55,  "LIV", "GK"),
            (5, 5, "2526", 1, 105, "CHE", "MID"),
            # Second gameweek entry for Palmer (to test gw filtering)
            (6, 5, "2526", 2, 105, "CHE", "MID"),
        ],
    )
    con.commit()
    con.close()


def _make_scorer(tmp_path: Path, elo_ratings: dict[str, float]) -> CopilotEloScorer:
    """Build a scorer wired to a temp DB and mocked ClubElo."""
    db_path = tmp_path / "airsenal.db"
    _seed_airsenal_db(db_path)
    scorer = CopilotEloScorer(db_path=str(db_path))
    # Patch the ClubElo fetch to return our controlled ratings
    with patch.object(scorer, "_fetch_elo_ratings", return_value=elo_ratings):
        # Pre-warm the internal cache so subsequent calls use it
        scorer._elo_ratings = elo_ratings
    return scorer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_correct_structure(tmp_path: Path) -> None:
    """Each player dict must contain the five required keys."""
    ratings = {
        "Manchester City": 1900.0,
        "Arsenal": 1880.0,
        "Liverpool": 1870.0,
        "Chelsea": 1850.0,
    }
    scorer = _make_scorer(tmp_path, ratings)

    results = scorer.get_player_elo_scores(gameweek=1)

    assert isinstance(results, list)
    assert len(results) == 5

    for player in results:
        assert set(player.keys()) == {"player_id", "player_name", "team", "position", "elo_score"}
        assert isinstance(player["player_id"], int)
        assert isinstance(player["player_name"], str)
        assert isinstance(player["team"], str)
        assert isinstance(player["position"], str)
        assert isinstance(player["elo_score"], float)


def test_position_factors_applied_correctly(tmp_path: Path) -> None:
    """FWD=1.2, MID=1.1, DEF=0.9, GK=0.8 must be applied to team ELO."""
    ratings = {
        "Manchester City": 2000.0,
        "Arsenal": 1800.0,
        "Liverpool": 1700.0,
    }
    scorer = _make_scorer(tmp_path, ratings)

    results = scorer.get_player_elo_scores(gameweek=1)
    by_name = {p["player_name"]: p for p in results}

    # Haaland (FWD, MCI 2000) → 2000 * 1.2 = 2400
    assert by_name["Haaland"]["elo_score"] == pytest.approx(2400.0)
    # Saka (MID, ARS 1800) → 1800 * 1.1 = 1980
    assert by_name["Saka"]["elo_score"] == pytest.approx(1980.0)
    # van Dijk (DEF, LIV 1700) → 1700 * 0.9 = 1530
    assert by_name["van Dijk"]["elo_score"] == pytest.approx(1530.0)
    # Alisson (GK, LIV 1700) → 1700 * 0.8 = 1360
    assert by_name["Alisson"]["elo_score"] == pytest.approx(1360.0)


def test_gameweek_filtering(tmp_path: Path) -> None:
    """Only players from the requested gameweek should be returned."""
    ratings = {"Chelsea": 1850.0}
    scorer = _make_scorer(tmp_path, ratings)

    # GW 1 → no Chelsea players in GW 1 seed data (Palmer is GW 1? Actually Palmer is GW 1 too)
    # Let's check GW 2 — only Palmer has GW 2 entry
    results_gw2 = scorer.get_player_elo_scores(gameweek=2)
    assert len(results_gw2) == 1
    assert results_gw2[0]["player_name"] == "Palmer"

    # GW 1 → Palmer is also in GW 1
    results_gw1 = scorer.get_player_elo_scores(gameweek=1)
    palmer_gw1 = [p for p in results_gw1 if p["player_name"] == "Palmer"]
    assert len(palmer_gw1) == 1


def test_empty_ratings_returns_empty_list(tmp_path: Path) -> None:
    """When ClubElo returns no ratings, the scorer returns an empty list."""
    scorer = _make_scorer(tmp_path, {})
    results = scorer.get_player_elo_scores(gameweek=1)
    assert results == []


def test_unknown_team_skipped_gracefully(tmp_path: Path) -> None:
    """Players whose team has no ClubElo rating are excluded, not crashed."""
    # Only provide rating for ManCity — everyone else should be skipped
    ratings = {"Manchester City": 1900.0}
    scorer = _make_scorer(tmp_path, ratings)

    results = scorer.get_player_elo_scores(gameweek=1)
    assert len(results) == 1
    assert results[0]["player_name"] == "Haaland"
    assert results[0]["elo_score"] == pytest.approx(2280.0)  # 1900 * 1.2


def test_default_gameweek_uses_latest(tmp_path: Path) -> None:
    """When gameweek is None, the scorer picks the latest available gameweek."""
    ratings = {
        "Manchester City": 1900.0,
        "Arsenal": 1880.0,
        "Liverpool": 1870.0,
        "Chelsea": 1850.0,
    }
    scorer = _make_scorer(tmp_path, ratings)

    # With no gw specified, should use latest (GW 2 in our seed)
    results = scorer.get_player_elo_scores()
    # Only Palmer has GW 2 data
    assert len(results) == 1
    assert results[0]["player_name"] == "Palmer"
