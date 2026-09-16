"""
Unit tests for the FDR services:
  - src/services/injury_impact
  - src/services/fixture_fdr (pure maths, no network calls)
  - src/services/club_elo (CSV parsing)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# injury_impact tests
# ---------------------------------------------------------------------------

from src.services.injury_impact import (
    availability_loss,
    summarize_injury_impact,
    team_injury_losses,
    ATT_ROLE,
)
from src.services.squad_change import summarize_squad_changes
from src.services.injury_news import get_team_injury_players


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
            {"position": "ST", "minutes_last6": 540, "goals90": 0.5, "prob_available": 1.0, "absence_type": "injury"},
            {"position": "CB", "minutes_last6": 540, "prob_available": 1.0, "absence_type": "injury"},
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
                "absence_type": "injury",
            }
        ]
        att, def_ = team_injury_losses(players)
        # Striker has high att_role (0.95) so att_loss should be substantial
        assert att > 0.1
        # Striker has low def_role (0.10) so def_loss should be small
        assert def_ < att

    def test_long_term_injured_starter_with_zero_recent_minutes_still_has_impact(self):
        players = [
            {
                "player_name": "Long-term Starter",
                "team": "Arsenal",
                "position": "CB",
                "minutes_last6": 0.0,
                "minutes_season": 1980.0,
                "minutes_last10_before_absence": 900.0,
                "starter_probability": 0.95,
                "status": "out",
                "absence_type": "injury",
            }
        ]
        att, def_ = team_injury_losses(players)
        assert att > 0.0
        assert def_ > 0.0

    def test_key_cb_out_increases_def_loss(self):
        players = [
            {
                "position": "CB",
                "minutes_last6": 540,
                "goals90": 0.0,
                "assists90": 0.0,
                "prob_available": 0.0,  # ruled out
                "absence_type": "injury",
            }
        ]
        att, def_ = team_injury_losses(players)
        # CB has high def_role (0.95), low att_role (0.10)
        assert def_ > att

    def test_losses_are_in_zero_one(self):
        players = [
            {"position": p, "minutes_last6": 540, "goals90": 0.5, "status": "out", "absence_type": "injury"}
            for p in ATT_ROLE
        ]
        att, def_ = team_injury_losses(players)
        assert 0.0 <= att < 1.0
        assert 0.0 <= def_ < 1.0

    def test_saturation_many_bench_injuries(self):
        """Losing 20 squad players should not send loss to 1.0 (saturation)."""
        players = [
            {"position": "CM", "minutes_last6": 10, "status": "out", "absence_type": "injury"}
            for _ in range(20)
        ]
        att, def_ = team_injury_losses(players)
        assert att < 1.0
        assert def_ < 1.0

    def test_fallback_for_unknown_position(self):
        players = [
            {"position": "UNKNOWN", "minutes_last6": 540, "prob_available": 0.0, "absence_type": "injury"}
        ]
        # Should not raise
        att, def_ = team_injury_losses(players)
        assert att >= 0.0
        assert def_ >= 0.0

    def test_loaned_player_excluded_from_injury_loss(self):
        players = [
            {
                "player_name": "Loaned Fullback",
                "position": "CB",
                "minutes_last6": 540,
                "prob_available": 0.0,
                "status": "out",
                "absence_type": "loan_out",
                "source_news": "has joined Cardiff City on loan for the rest of the season.",
            }
        ]
        att, def_ = team_injury_losses(players)
        summary = summarize_injury_impact(players)
        assert att == pytest.approx(0.0)
        assert def_ == pytest.approx(0.0)
        assert summary["counted_absences"] == []
        assert summary["ignored_absences"][0]["absence_type"] == "loan_out"

    def test_transferred_player_excluded_from_injury_loss(self):
        players = [
            {
                "player_name": "Transferred Striker",
                "position": "ST",
                "minutes_last6": 540,
                "goals90": 0.7,
                "prob_available": 0.0,
                "status": "out",
                "absence_type": "transfer_out",
                "source_news": "Has joined Marseille permanently.",
            }
        ]
        att, def_ = team_injury_losses(players)
        assert att == pytest.approx(0.0)
        assert def_ == pytest.approx(0.0)

    def test_distinct_real_team_absences_produce_distinct_losses(self):
        everton_losses = team_injury_losses(get_team_injury_players("Everton"))
        bournemouth_losses = team_injury_losses(get_team_injury_players("Bournemouth"))
        assert everton_losses != bournemouth_losses


class TestSquadChangeLayer:
    def test_recent_major_departure_affects_squad_change_layer(self):
        players = [
            {
                "player_name": "Star Winger",
                "position": "W",
                "minutes_last6": 540,
                "goals90": 0.6,
                "assists90": 0.3,
                "absence_type": "transfer_out",
                "source_news": "Has joined Bayern permanently.",
                "last_updated": "2026-03-09T00:00:00+00:00",
            }
        ]
        summary = summarize_squad_changes(players, snapshot_date="2026-03-10")
        assert summary["attack_loss"] > 0.0
        assert summary["defence_loss"] >= 0.0
        assert summary["counted_absences"][0]["absence_type"] == "transfer_out"

    def test_squad_change_decay_reduces_old_departure_effect(self):
        players = [
            {
                "player_name": "Departed Midfielder",
                "position": "CM",
                "minutes_last6": 540,
                "minutes_season": 1800,
                "minutes_last10_before_absence": 900,
                "goals90": 0.2,
                "assists90": 0.2,
                "absence_type": "loan_out",
                "source_news": "has joined Stoke City on loan for the rest of the season.",
                "matches_since_departure": 0,
            }
        ]
        recent = summarize_squad_changes(players, snapshot_date="2026-03-10")
        old_players = [{**players[0], "matches_since_departure": 6}]
        old = summarize_squad_changes(old_players, snapshot_date="2026-03-10")
        assert recent["attack_loss"] > old["attack_loss"]
        assert recent["defence_loss"] > old["defence_loss"]


# ---------------------------------------------------------------------------
# fixture_fdr pure-maths tests
# ---------------------------------------------------------------------------

from src.services.fixture_fdr import (
    sigmoid,
    clamp,
    elo_base,
    raw_fdrs,
    to_fdr,
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
            {"position": "CB", "minutes_last6": 540, "status": "out", "absence_type": "injury"},
            {"position": "GK", "minutes_last6": 540, "status": "out", "absence_type": "injury"},
        ]
        ra_with, _, _ = raw_fdrs(1600, 1600, is_home=True, opp_players=opp_def_players)
        ra_without, _, _ = raw_fdrs(1600, 1600, is_home=True)
        assert ra_with < ra_without

    def test_opp_att_loss_lowers_defence_difficulty(self):
        # If opponent loses key attackers their attack is weaker →
        # defence FDR for our team should be lower (easier to keep clean sheet)
        opp_att_players = [
            {"position": "ST", "minutes_last6": 540, "goals90": 0.8, "status": "out", "absence_type": "injury"},
        ]
        _, rd_with, _ = raw_fdrs(1600, 1600, is_home=True, opp_players=opp_att_players)
        _, rd_without, _ = raw_fdrs(1600, 1600, is_home=True)
        assert rd_with < rd_without

    def test_own_att_loss_increases_attack_difficulty(self):
        own_att_players = [
            {"position": "ST", "minutes_last6": 540, "goals90": 0.8, "status": "out", "absence_type": "injury"},
        ]
        ra_with, _, _ = raw_fdrs(1600, 1600, is_home=True, team_players=own_att_players)
        ra_without, _, _ = raw_fdrs(1600, 1600, is_home=True)
        assert ra_with > ra_without

    def test_own_def_loss_increases_defence_difficulty(self):
        own_def_players = [
            {"position": "CB", "minutes_last6": 540, "status": "out", "absence_type": "injury"},
        ]
        _, rd_with, _ = raw_fdrs(1600, 1600, is_home=True, team_players=own_def_players)
        _, rd_without, _ = raw_fdrs(1600, 1600, is_home=True)
        assert rd_with > rd_without

    def test_recent_departure_adjusts_raw_fdr(self):
        team_departure = [
            {
                "position": "ST",
                "minutes_last6": 540,
                "goals90": 0.7,
                "assists90": 0.2,
                "absence_type": "transfer_out",
                "source_news": "Has joined Al-Hilal permanently.",
                "last_updated": "2026-03-09T00:00:00+00:00",
            }
        ]
        ra_with, rd_with, _ = raw_fdrs(1600, 1600, is_home=True, team_players=team_departure, snapshot_date="2026-03-10")
        ra_without, rd_without, _ = raw_fdrs(1600, 1600, is_home=True, snapshot_date="2026-03-10")
        assert ra_with > ra_without
        assert rd_with >= rd_without


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

from src.services.club_elo import _parse_clubelo_csv, build_premier_league_elo_snapshot, get_team_elo


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

    @pytest.mark.parametrize(
        ("team_name", "clubelo_name", "elo"),
        [
            ("Man City", "ManCity", 2060.5),
            ("Man Utd", "Manchester United", 1910.4),
            ("Spurs", "Tottenham", 1855.2),
            ("Nott'm Forest", "Nottingham Forest", 1760.8),
            ("Wolves", "Wolverhampton", 1742.6),
        ],
    )
    def test_alias_match(self, team_name, clubelo_name, elo):
        with patch("src.services.club_elo.fetch_elo_ratings") as mock_fetch:
            mock_fetch.return_value = {clubelo_name: elo}
            assert get_team_elo(team_name) == pytest.approx(elo)

    def test_not_found_returns_fallback(self):
        with patch("src.services.club_elo.fetch_elo_ratings") as mock_fetch:
            mock_fetch.return_value = {}
            assert get_team_elo("UnknownFC", fallback=1500.0) == pytest.approx(1500.0)


class TestBuildPremierLeagueEloSnapshot:
    def test_filters_to_current_fpl_teams_and_maps_names(self):
        ratings = {
            "Arsenal": 2047.1,
            "ManCity": 2060.5,
            "Tottenham": 1855.2,
            "Nottingham Forest": 1760.8,
            "Wolverhampton": 1742.6,
            "Bayern Munich": 2101.0,
        }
        with patch("src.services.club_elo.get_current_premier_league_teams") as mock_teams:
            mock_teams.return_value = [
                {"team_id": 1, "full_name": "Arsenal"},
                {"team_id": 13, "full_name": "Man City"},
                {"team_id": 18, "full_name": "Spurs"},
                {"team_id": 16, "full_name": "Nott'm Forest"},
                {"team_id": 20, "full_name": "Wolves"},
            ]
            snapshot = build_premier_league_elo_snapshot(ratings, snapshot_date="2026-03-10")

        assert snapshot == {
            "snapshot_date": "2026-03-10",
            "ratings": [
                {"team_id": 1, "team": "Arsenal", "elo": 2047.1},
                {"team_id": 13, "team": "Man City", "elo": 2060.5},
                {"team_id": 18, "team": "Spurs", "elo": 1855.2},
                {"team_id": 16, "team": "Nott'm Forest", "elo": 1760.8},
                {"team_id": 20, "team": "Wolves", "elo": 1742.6},
            ],
        }

    def test_skips_unmatched_teams(self):
        with patch("src.services.club_elo.get_current_premier_league_teams") as mock_teams:
            mock_teams.return_value = [
                {"team_id": 1, "full_name": "Arsenal"},
                {"team_id": 7, "full_name": "Chelsea"},
            ]
            snapshot = build_premier_league_elo_snapshot({"Arsenal": 2047.1}, snapshot_date="2026-03-10")

        assert snapshot["ratings"] == [{"team_id": 1, "team": "Arsenal", "elo": 2047.1}]


# ---------------------------------------------------------------------------
# compute_fixture_fdr integration tests (no network)
# ---------------------------------------------------------------------------

from src.services.fixture_fdr import compute_fixture_fdr
from src.main import app


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
            "team_id", "opponent_id",
            "elo_team", "elo_opponent", "base_raw",
            "team_attack_loss", "team_defence_loss",
            "team_squad_change_attack_loss", "team_squad_change_defence_loss",
            "opp_attack_loss", "opp_defence_loss",
            "opp_squad_change_attack_loss", "opp_squad_change_defence_loss",
            "team_counted_absences", "team_ignored_absences",
            "opp_counted_absences", "opp_ignored_absences",
            "key_absences_counted", "key_absences_ignored",
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
            {"position": "CB", "minutes_last6": 540, "status": "out", "absence_type": "injury"},
            {"position": "GK", "minutes_last6": 540, "status": "out", "absence_type": "injury"},
        ]
        result = compute_fixture_fdr(
            **self._make_request(opp_players=opp_players)
        )
        assert result["opp_defence_loss"] > 0.0
        assert result["opp_counted_absences"]
        # With opponent's defenders out, attack FDR should be lower (easier)
        result_no_injury = compute_fixture_fdr(**self._make_request())
        assert result["attack_fdr"] < result_no_injury["attack_fdr"]


class TestEloEndpoint:
    def test_openapi_uses_typed_response_model(self):
        client = TestClient(app)
        response = client.get("/openapi.json")
        assert response.status_code == 200

        schema = response.json()
        elo_schema = schema["paths"]["/api/fdr/elo"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert elo_schema["$ref"].endswith("/EloSnapshotResponse")

        component = schema["components"]["schemas"]["EloSnapshotResponse"]
        assert component["type"] == "object"
        assert set(component["properties"]) == {"snapshot_date", "ratings"}

        rating_component = schema["components"]["schemas"]["EloRatingRecord"]
        assert rating_component["type"] == "object"
        assert set(rating_component["properties"]) == {"team_id", "team", "elo"}

    def test_endpoint_returns_premier_league_snapshot_shape(self):
        payload = {
            "snapshot_date": "2026-03-10",
            "ratings": [
                {"team_id": 1, "team": "Arsenal", "elo": 2047.0},
                {"team_id": 7, "team": "Chelsea", "elo": 1922.0},
            ],
        }
        with patch("src.services.club_elo.fetch_premier_league_elo_snapshot") as mock_fetch:
            mock_fetch.return_value = payload
            client = TestClient(app)
            response = client.get("/api/fdr/elo?snapshot_date=2026-03-10")

        assert response.status_code == 200
        assert response.json() == payload


class TestTeamFixturesEndpoint:
    def test_openapi_uses_explicit_team_fixtures_response_model(self):
        client = TestClient(app)
        schema = client.get("/openapi.json").json()

        team_fixture_schema = schema["paths"]["/api/fdr/team/{team_name}"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]
        assert team_fixture_schema["$ref"].endswith("/TeamFixturesFDRResponse")

