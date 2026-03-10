"""
Unit tests for the FDR services:
  - src/services/injury_impact
  - src/services/fixture_fdr (pure maths, no network calls)
  - src/services/club_elo (CSV parsing)
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# injury_impact tests
# ---------------------------------------------------------------------------

from src.services.injury_impact import (
    availability_loss,
    team_injury_losses,
    ATT_ROLE,
    DEF_ROLE,
)


class TestAvailabilityLoss:
    def test_fully_fit_by_prob(self):
        assert availability_loss(prob_available=1.0) == pytest.approx(0.0)

    def test_out_by_prob(self):
        assert availability_loss(prob_available=0.0) == pytest.approx(1.0)

    def test_prob_clamps_below_zero(self):
        assert availability_loss(prob_available=-0.5) == pytest.approx(1.0)

    def test_prob_clamps_above_one(self):
        assert availability_loss(prob_available=1.5) == pytest.approx(0.0)

    def test_status_available(self):
        assert availability_loss(status="available") == pytest.approx(0.0)

    def test_status_out(self):
        assert availability_loss(status="out") == pytest.approx(1.0)

    def test_status_suspended(self):
        assert availability_loss(status="suspended") == pytest.approx(1.0)

    def test_status_doubtful(self):
        assert availability_loss(status="doubtful") == pytest.approx(0.25)

    def test_status_questionable(self):
        assert availability_loss(status="questionable") == pytest.approx(0.50)

    def test_status_major_doubt(self):
        assert availability_loss(status="major_doubt") == pytest.approx(0.75)

    def test_status_case_insensitive(self):
        assert availability_loss(status="OUT") == pytest.approx(1.0)

    def test_unknown_status_returns_zero(self):
        assert availability_loss(status="fit_enough") == pytest.approx(0.0)

    def test_no_args_returns_zero(self):
        assert availability_loss() == pytest.approx(0.0)

    def test_prob_takes_precedence_over_status(self):
        # prob_available=0.75 => loss=0.25, not the 1.0 from status="out"
        assert availability_loss(prob_available=0.75, status="out") == pytest.approx(0.25)


class TestTeamInjuryLosses:
    def test_empty_players_returns_zeros(self):
        att, def_ = team_injury_losses([])
        assert att == pytest.approx(0.0)
        assert def_ == pytest.approx(0.0)

    def test_all_fit_returns_zeros(self):
        players = [
            {"position": "ST", "minutes_last6": 540, "goals90": 0.5, "prob_available": 1.0},
            {"position": "CB", "minutes_last6": 540, "prob_available": 1.0},
        ]
        att, def_ = team_injury_losses(players)
        assert att == pytest.approx(0.0)
        assert def_ == pytest.approx(0.0)

    def test_key_striker_out_increases_att_loss(self):
        players = [
            {
                "position": "ST",
                "minutes_last6": 540,
                "goals90": 0.8,
                "assists90": 0.3,
                "prob_available": 0.0,  # ruled out
            }
        ]
        att, def_ = team_injury_losses(players)
        # Striker has high att_role (0.95) so att_loss should be substantial
        assert att > 0.1
        # Striker has low def_role (0.10) so def_loss should be small
        assert def_ < att

    def test_key_cb_out_increases_def_loss(self):
        players = [
            {
                "position": "CB",
                "minutes_last6": 540,
                "goals90": 0.0,
                "assists90": 0.0,
                "prob_available": 0.0,  # ruled out
            }
        ]
        att, def_ = team_injury_losses(players)
        # CB has high def_role (0.95), low att_role (0.10)
        assert def_ > att

    def test_losses_are_in_zero_one(self):
        players = [
            {"position": p, "minutes_last6": 540, "goals90": 0.5, "status": "out"}
            for p in ATT_ROLE
        ]
        att, def_ = team_injury_losses(players)
        assert 0.0 <= att < 1.0
        assert 0.0 <= def_ < 1.0

    def test_saturation_many_bench_injuries(self):
        """Losing 20 squad players should not send loss to 1.0 (saturation)."""
        players = [
            {"position": "CM", "minutes_last6": 10, "status": "out"}
            for _ in range(20)
        ]
        att, def_ = team_injury_losses(players)
        assert att < 1.0
        assert def_ < 1.0

    def test_fallback_for_unknown_position(self):
        players = [
            {"position": "UNKNOWN", "minutes_last6": 540, "prob_available": 0.0}
        ]
        # Should not raise
        att, def_ = team_injury_losses(players)
        assert att >= 0.0
        assert def_ >= 0.0


# ---------------------------------------------------------------------------
# fixture_fdr pure-maths tests
# ---------------------------------------------------------------------------

from src.services.fixture_fdr import (
    sigmoid,
    clamp,
    elo_base,
    raw_fdrs,
    to_fdr,
    HOME_ELO_BONUS,
)


class TestSigmoid:
    def test_at_zero(self):
        assert sigmoid(0) == pytest.approx(0.5)

    def test_large_positive(self):
        assert sigmoid(100) == pytest.approx(1.0, abs=1e-6)

    def test_large_negative(self):
        assert sigmoid(-100) == pytest.approx(0.0, abs=1e-6)


class TestClamp:
    def test_below_min(self):
        assert clamp(-5, 1, 5) == 1

    def test_above_max(self):
        assert clamp(10, 1, 5) == 5

    def test_in_range(self):
        assert clamp(3, 1, 5) == 3


class TestEloBase:
    def test_equal_teams_home_advantage(self):
        # Same Elo, home side should have slight advantage → negative base (easier)
        base = elo_base(1500, 1500, is_home=True)
        assert base < 0

    def test_equal_teams_away_disadvantage(self):
        base = elo_base(1500, 1500, is_home=False)
        assert base > 0

    def test_much_stronger_opponent_hard(self):
        # Team is very weak vs very strong opponent
        base = elo_base(1200, 1900, is_home=False)
        assert base > 0.35  # clearly harder than average

    def test_much_weaker_opponent_easy(self):
        # Team is very strong vs very weak opponent at home
        base = elo_base(1900, 1200, is_home=True)
        assert base < -0.35


class TestRawFdrs:
    def test_no_injuries_symmetric(self):
        # Same Elo, home, no injuries
        ra, rd, ro = raw_fdrs(1600, 1600, is_home=True)
        # Both should equal elo_base
        base = elo_base(1600, 1600, is_home=True)
        assert ra == pytest.approx(base)
        assert rd == pytest.approx(base)

    def test_opp_def_loss_lowers_attack_difficulty(self):
        # If opponent loses key defenders their defence is weaker →
        # attack FDR for our team should be lower (easier to score)
        opp_def_players = [
            {"position": "CB", "minutes_last6": 540, "status": "out"},
            {"position": "GK", "minutes_last6": 540, "status": "out"},
        ]
        ra_with, _, _ = raw_fdrs(1600, 1600, is_home=True, opp_players=opp_def_players)
        ra_without, _, _ = raw_fdrs(1600, 1600, is_home=True)
        assert ra_with < ra_without

    def test_opp_att_loss_lowers_defence_difficulty(self):
        # If opponent loses key attackers their attack is weaker →
        # defence FDR for our team should be lower (easier to keep clean sheet)
        opp_att_players = [
            {"position": "ST", "minutes_last6": 540, "goals90": 0.8, "status": "out"},
        ]
        _, rd_with, _ = raw_fdrs(1600, 1600, is_home=True, opp_players=opp_att_players)
        _, rd_without, _ = raw_fdrs(1600, 1600, is_home=True)
        assert rd_with < rd_without

    def test_own_att_loss_increases_attack_difficulty(self):
        own_att_players = [
            {"position": "ST", "minutes_last6": 540, "goals90": 0.8, "status": "out"},
        ]
        ra_with, _, _ = raw_fdrs(1600, 1600, is_home=True, team_players=own_att_players)
        ra_without, _, _ = raw_fdrs(1600, 1600, is_home=True)
        assert ra_with > ra_without

    def test_own_def_loss_increases_defence_difficulty(self):
        own_def_players = [
            {"position": "CB", "minutes_last6": 540, "status": "out"},
        ]
        _, rd_with, _ = raw_fdrs(1600, 1600, is_home=True, team_players=own_def_players)
        _, rd_without, _ = raw_fdrs(1600, 1600, is_home=True)
        assert rd_with > rd_without


class TestToFdr:
    def test_output_in_range(self):
        for raw in [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]:
            fdr = to_fdr(raw)
            assert 1.0 <= fdr <= 5.0

    def test_neutral_raw_near_three(self):
        # sigmoid(0) = 0.5, so 1 + 4*0.5 = 3.0
        assert to_fdr(0.0) == pytest.approx(3.0, abs=0.01)

    def test_harder_raw_gives_higher_fdr(self):
        assert to_fdr(1.0) > to_fdr(0.0)

    def test_easier_raw_gives_lower_fdr(self):
        assert to_fdr(-1.0) < to_fdr(0.0)


# ---------------------------------------------------------------------------
# ClubElo CSV parsing tests (no network)
# ---------------------------------------------------------------------------

from src.services.club_elo import _parse_clubelo_csv, get_team_elo


SAMPLE_CSV = """Rank,Club,Country,Level,Elo,From,To
1,ManCity,ENG,1,2060.5,2024-01-01,2026-01-01
2,Arsenal,ENG,1,2047.1,2024-01-01,2026-01-01
3,Chelsea,ENG,1,1922.3,2024-01-01,2026-01-01
"""


class TestParseClubEloCsv:
    def test_basic_parse(self):
        ratings = _parse_clubelo_csv(SAMPLE_CSV)
        assert ratings["ManCity"] == pytest.approx(2060.5)
        assert ratings["Arsenal"] == pytest.approx(2047.1)
        assert ratings["Chelsea"] == pytest.approx(1922.3)

    def test_empty_csv(self):
        ratings = _parse_clubelo_csv("")
        assert ratings == {}

    def test_missing_elo_column(self):
        csv_text = "Rank,Club,Country\n1,ManCity,ENG\n"
        ratings = _parse_clubelo_csv(csv_text)
        assert ratings == {}

    def test_malformed_elo_skipped(self):
        csv_text = "Rank,Club,Country,Level,Elo,From,To\n1,TeamA,ENG,1,not_a_float,2024-01-01,2026-01-01\n"
        ratings = _parse_clubelo_csv(csv_text)
        assert "TeamA" not in ratings


class TestGetTeamElo:
    def test_exact_match(self):
        with patch("src.services.club_elo.fetch_elo_ratings") as mock_fetch:
            mock_fetch.return_value = {"Arsenal": 2047.1, "Chelsea": 1922.3}
            assert get_team_elo("Arsenal") == pytest.approx(2047.1)

    def test_case_insensitive_match(self):
        with patch("src.services.club_elo.fetch_elo_ratings") as mock_fetch:
            mock_fetch.return_value = {"Arsenal": 2047.1}
            assert get_team_elo("arsenal") == pytest.approx(2047.1)

    def test_not_found_returns_fallback(self):
        with patch("src.services.club_elo.fetch_elo_ratings") as mock_fetch:
            mock_fetch.return_value = {}
            assert get_team_elo("UnknownFC", fallback=1500.0) == pytest.approx(1500.0)


# ---------------------------------------------------------------------------
# compute_fixture_fdr integration tests (no network)
# ---------------------------------------------------------------------------

from src.services.fixture_fdr import compute_fixture_fdr


class TestComputeFixtureFdr:
    def _make_request(self, **kwargs):
        defaults = dict(
            team_name="Arsenal",
            opponent_name="Chelsea",
            is_home=True,
            elo_team=2047.0,
            elo_opp=1922.0,
        )
        defaults.update(kwargs)
        return defaults

    def test_response_shape(self):
        result = compute_fixture_fdr(**self._make_request())
        expected_keys = {
            "team", "opponent", "is_home",
            "elo_team", "elo_opponent", "base_raw",
            "team_attack_loss", "team_defence_loss",
            "opp_attack_loss", "opp_defence_loss",
            "raw_attack", "raw_defence", "raw_overall",
            "attack_fdr", "defence_fdr", "overall_fdr",
            "attack_fdr_int", "defence_fdr_int", "overall_fdr_int",
        }
        assert set(result.keys()) == expected_keys

    def test_fdr_values_in_range(self):
        result = compute_fixture_fdr(**self._make_request())
        for key in ("attack_fdr", "defence_fdr", "overall_fdr"):
            assert 1.0 <= result[key] <= 5.0, f"{key} out of range: {result[key]}"

    def test_int_fdr_is_rounded(self):
        result = compute_fixture_fdr(**self._make_request())
        assert result["attack_fdr_int"] == round(result["attack_fdr"])

    def test_stronger_opponent_increases_fdr(self):
        easy = compute_fixture_fdr(
            team_name="T", opponent_name="O", is_home=True,
            elo_team=2000, elo_opp=1500,
        )
        hard = compute_fixture_fdr(
            team_name="T", opponent_name="O", is_home=True,
            elo_team=1500, elo_opp=2000,
        )
        assert hard["overall_fdr"] > easy["overall_fdr"]

    def test_no_injury_data_returns_clean_result(self):
        result = compute_fixture_fdr(**self._make_request())
        assert result["team_attack_loss"] == pytest.approx(0.0)
        assert result["opp_defence_loss"] == pytest.approx(0.0)

    def test_with_injury_data(self):
        opp_players = [
            {"position": "CB", "minutes_last6": 540, "status": "out"},
            {"position": "GK", "minutes_last6": 540, "status": "out"},
        ]
        result = compute_fixture_fdr(
            **self._make_request(opp_players=opp_players)
        )
        assert result["opp_defence_loss"] > 0.0
        # With opponent's defenders out, attack FDR should be lower (easier)
        result_no_injury = compute_fixture_fdr(**self._make_request())
        assert result["attack_fdr"] < result_no_injury["attack_fdr"]
