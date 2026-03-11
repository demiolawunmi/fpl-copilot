from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from src.main import app
from src.services.injury_news import (
    autofill_player_record_from_db,
    canonical_team_name,
    classify_absence_type,
    derive_position_group,
    enrich_player_record,
    get_team_injury_players,
    get_next_team_fixtures,
    load_current_injury_news,
    resolve_fixture_context,
    resolve_team,
    sanitize_snapshot_date,
    saturate_team_players,
    status_from_injury_news,
)


class TestInjuryNewsHelpers:
    def test_classify_absence_type(self):
        assert classify_absence_type("Hamstring injury - Expected back 11 Apr", "out", 0.0) == "injury"
        assert classify_absence_type("Suspended until 22 Mar", "suspended", 0.0) == "suspension"
        assert classify_absence_type("Knock - 75% chance of playing", "doubtful", 0.75) == "doubtful"
        assert classify_absence_type("Hamstring injury - 50% chance of playing", "questionable", 0.5) == "questionable"
        assert classify_absence_type("has joined Stoke City on loan for the rest of the season.", "out", 0.0) == "loan_out"
        assert classify_absence_type("Has joined Marseille permanently.", "out", 0.0) == "transfer_out"

    def test_canonical_team_name_supports_common_aliases(self):
        assert canonical_team_name("Tottenham") == "Spurs"
        assert canonical_team_name("Tottenham Hotspur") == "Spurs"
        assert canonical_team_name("Manchester City") == "Man City"

    def test_derive_position_group(self):
        assert derive_position_group("GK") == "goalkeeper"
        assert derive_position_group("DEF") == "defence"
        assert derive_position_group("CB") == "defence"
        assert derive_position_group("CM") == "midfield"
        assert derive_position_group("ST") == "attack"
        assert derive_position_group(None) is None

    def test_status_from_injury_news(self):
        assert status_from_injury_news(75, "Knock - 75% chance of playing") == "doubtful"
        assert status_from_injury_news(50, "Hamstring injury - 50% chance of playing") == "questionable"
        assert status_from_injury_news(25, "Back injury - 25% chance of playing") == "major_doubt"
        assert status_from_injury_news(0, "Suspended until 22 Mar") == "suspended"
        assert status_from_injury_news(0, "Knee injury - Unknown return date") == "out"

    def test_sanitize_snapshot_date_placeholder(self):
        assert sanitize_snapshot_date("string") is None
        assert sanitize_snapshot_date("2026-03-10") == "2026-03-10"

    def test_load_current_injury_news_uses_readable_team_names(self):
        injuries = load_current_injury_news()
        assert injuries, "Expected current injury rows from data/api/injury_news.json"
        sample = next(row for row in injuries if row["player_name"] == "Calafiori")
        assert sample["team"] == "Arsenal"
        assert sample["team_code"] == "ARS"
        assert sample["position"] == "CB"
        assert sample["position_group"] == "defence"
        assert sample["absence_type"] == "doubtful"
        assert sample["status"] == "doubtful"
        assert sample["prob_available"] == 0.75
        assert sample["source_news"] == "Knock - 75% chance of playing"
        assert sample["last_updated"]

    def test_team_injury_lookup_returns_enriched_players(self):
        players = get_team_injury_players("Spurs")
        van_de_ven = next(player for player in players if player["player_id"] == 701)
        assert van_de_ven["player_name"] == "Van de Ven"
        assert van_de_ven["team"] == "Spurs"
        assert van_de_ven["position"] == "CB"
        assert van_de_ven["minutes_last6"] > 0
        assert van_de_ven["absence_type"] == "suspension"
        assert van_de_ven["status"] == "suspended"

    def test_saturate_team_players_merges_manual_and_injury_news(self):
        players = saturate_team_players(
            "Arsenal",
            [{"player_id": 7, "minutes_last6": 540, "goals90": 0.2}],
        )
        calafiori = next(player for player in players if player["player_id"] == 7)
        assert calafiori["team"] == "Arsenal"
        assert calafiori["minutes_last6"] == 540
        assert calafiori["goals90"] == 0.2
        assert calafiori["absence_type"] == "doubtful"
        assert calafiori["status"] == "doubtful"

    def test_resolve_fixture_context_from_fixture_id_defaults_to_home_team(self):
        context = resolve_fixture_context(fixture_id=13)
        assert context["fixture_id"] == 13
        assert context["team"] == "Man City"
        assert context["team_id"] == 13
        assert context["opponent"] == "Spurs"
        assert context["opponent_id"] == 18
        assert context["is_home"] is True
        assert context["home_team"] == "Man City"
        assert context["home_team_id"] == 13
        assert context["away_team"] == "Spurs"
        assert context["away_team_id"] == 18

    def test_resolve_team_supports_team_id_string(self):
        team = resolve_team("12")
        assert team is not None
        assert team["team_id"] == 12
        assert team["full_name"] == "Liverpool"

    def test_get_next_team_fixtures_supports_alias(self):
        fixtures = get_next_team_fixtures("Tottenham", limit=2)
        assert len(fixtures) == 2
        assert all(18 in {fixture["home_team_id"], fixture["away_team_id"]} for fixture in fixtures)
        assert fixtures == sorted(fixtures, key=lambda fixture: (fixture["date"] or "", fixture["fixture_id"]))

    def test_resolve_fixture_context_validates_conflicts(self):
        try:
            resolve_fixture_context(fixture_id=13, team_name="Arsenal")
            assert False, "Expected fixture/team mismatch to raise"
        except ValueError as exc:
            assert "does not belong" in str(exc)

    def test_db_autofill_by_player_id(self):
        player = autofill_player_record_from_db({"player_id": 701})
        assert player["player_id"] == 701
        assert player["player_name"] == "Van de Ven"
        assert player["team"] == "Spurs"
        assert player["position"] == "CB"
        assert player["position_group"] == "defence"
        assert player["minutes_last6"] > 0
        assert player["goals90"] >= 0
        assert player["assists90"] >= 0

    def test_db_autofill_by_name_and_team(self):
        player = autofill_player_record_from_db({"player_name": "Calafiori", "team": "Arsenal"})
        assert player["player_id"] == 7
        assert player["player_name"] == "Calafiori"
        assert player["team"] == "Arsenal"
        assert player["position"] == "CB"
        assert player["minutes_last6"] > 0

    def test_enrich_player_record_backfills_fields(self):
        player = {
            "player_id": 7,
            "player_name": "Calafiori",
            "team": "ARS",
            "position": "CB",
            "minutes_last6": 540,
        }
        enriched = enrich_player_record(player)
        assert enriched["team"] == "Arsenal"
        assert enriched["position"] == "CB"
        assert enriched["position_group"] == "defence"
        assert enriched["absence_type"] == "doubtful"
        assert enriched["prob_available"] == 0.75
        assert enriched["status"] == "doubtful"
        assert enriched["source_news"] == "Knock - 75% chance of playing"
        assert enriched["last_updated"]


class TestInjuryNewsApi:
    def _fake_fdr_response(
        self,
        team: str = "Arsenal",
        opponent: str = "Chelsea",
        is_home: bool = True,
        team_id: int | None = None,
        opponent_id: int | None = None,
    ):
        return {
            "team": team,
            "team_id": team_id,
            "opponent": opponent,
            "opponent_id": opponent_id,
            "is_home": is_home,
            "elo_team": 2000.0,
            "elo_opponent": 1900.0,
            "base_raw": -0.1,
            "team_attack_loss": 0.0,
            "team_defence_loss": 0.0,
            "team_squad_change_attack_loss": 0.0,
            "team_squad_change_defence_loss": 0.0,
            "opp_attack_loss": 0.0,
            "opp_defence_loss": 0.0,
            "opp_squad_change_attack_loss": 0.0,
            "opp_squad_change_defence_loss": 0.0,
            "team_counted_absences": [],
            "team_ignored_absences": [],
            "opp_counted_absences": [],
            "opp_ignored_absences": [],
            "key_absences_counted": [],
            "key_absences_ignored": [],
            "raw_attack": -0.1,
            "raw_defence": -0.1,
            "raw_overall": -0.1,
            "attack_fdr": 2.5,
            "defence_fdr": 2.5,
            "overall_fdr": 2.5,
            "attack_fdr_int": 2,
            "defence_fdr_int": 2,
            "overall_fdr_int": 2,
        }

    def test_openapi_player_record_includes_explanation_fields(self):
        client = TestClient(app)
        schema = client.get("/openapi.json").json()

        player_record = schema["components"]["schemas"]["PlayerRecord"]
        assert set(player_record["properties"]).issuperset(
            {
                "player_id",
                "player_name",
                "team",
                "position",
                "position_group",
                "absence_type",
                "minutes_season",
                "minutes_last10_before_absence",
                "starter_probability",
                "matches_since_departure",
                "source_news",
                "last_updated",
            }
        )
        assert "position" not in player_record.get("required", [])

    def test_injuries_endpoint_returns_typed_payload(self):
        payload = [
            {
                "player_id": 7,
                "player_name": "Calafiori",
                "team": "Arsenal",
                "team_code": "ARS",
                "position": "CB",
                "position_group": "defence",
                "absence_type": "doubtful",
                "prob_available": 0.75,
                "status": "doubtful",
                "source_news": "Knock - 75% chance of playing",
                "last_updated": "2026-03-10T00:00:00+00:00",
            }
        ]
        with patch("src.services.injury_news.load_current_injury_news") as mock_load:
            mock_load.return_value = payload
            client = TestClient(app)
            response = client.get("/api/fdr/injuries")

        assert response.status_code == 200
        assert response.json() == {"injuries": payload}

    def test_fixture_route_enriches_player_records_before_compute(self):
        fake_injury_rows = [
            {
                "player_id": 7,
                "player_name": "Calafiori",
                "team": "Arsenal",
                "team_code": "ARS",
                "position": "CB",
                "position_group": "defence",
                "absence_type": "doubtful",
                "prob_available": 0.75,
                "status": "doubtful",
                "source_news": "Knock - 75% chance of playing",
                "last_updated": "2026-03-10T00:00:00+00:00",
            }
        ]
        request_payload = {
            "team": "Arsenal",
            "opponent": "Chelsea",
            "is_home": True,
            "team_players": [
                {
                    "player_id": 7,
                    "player_name": "Calafiori",
                    "team": "ARS",
                    "position": "CB",
                    "minutes_last6": 540,
                }
            ],
        }
        expected_fdr = self._fake_fdr_response(team_id=1, opponent_id=7)

        with patch("src.services.injury_news.load_current_injury_news") as mock_load, patch(
            "src.services.fixture_fdr.compute_fixture_fdr"
        ) as mock_compute:
            mock_load.return_value = fake_injury_rows
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.post("/api/fdr/fixture", json=request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fdr"] == expected_fdr
        assert body["saturated"]["team"] == "Arsenal"
        assert body["saturated"]["team_id"] == 1
        assert body["saturated"]["opponent"] == "Chelsea"
        assert body["saturated"]["opponent_id"] == 7
        assert body["saturated"]["is_home"] is True

        kwargs = mock_compute.call_args.kwargs
        enriched_player = kwargs["team_players"][0]
        assert enriched_player["team"] == "Arsenal"
        assert enriched_player["position"] == "CB"
        assert enriched_player["position_group"] == "defence"
        assert enriched_player["absence_type"] == "doubtful"
        assert enriched_player["prob_available"] == 0.75
        assert enriched_player["status"] == "doubtful"
        assert enriched_player["source_news"] == "Knock - 75% chance of playing"
        assert enriched_player["last_updated"] == "2026-03-10T00:00:00+00:00"

    def test_fixture_route_hydrates_swagger_placeholder_player_from_db_and_injury_news(self):
        expected_fdr = self._fake_fdr_response(team="Liverpool", opponent="Spurs", team_id=12, opponent_id=18)
        request_payload = {
            "team": "Liverpool",
            "opponent": "Tottenham",
            "is_home": True,
            "snapshot_date": "string",
            "opp_players": [
                {
                    "player_id": 701,
                    "player_name": "string",
                    "team": "string",
                    "position": "string",
                    "position_group": "string",
                    "minutes_last6": 0,
                    "goals90": 0,
                    "assists90": 0,
                    "prob_available": 0,
                    "status": "string",
                    "source_news": "string",
                    "last_updated": "string",
                }
            ],
        }

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.post("/api/fdr/fixture", json=request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fdr"] == expected_fdr
        assert body["saturated"]["snapshot_date"] is None
        assert body["saturated"]["team_id"] == 12
        assert body["saturated"]["opponent_id"] == 18
        assert "team_ignored_absences" in body["fdr"]
        assert "opp_ignored_absences" in body["fdr"]

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["snapshot_date"] is None
        enriched_player = next(player for player in kwargs["opp_players"] if player["player_id"] == 701)
        assert enriched_player["player_id"] == 701
        assert enriched_player["player_name"] == "Van de Ven"
        assert enriched_player["team"] == "Spurs"
        assert enriched_player["position"] == "CB"
        assert enriched_player["position_group"] == "defence"
        assert enriched_player["absence_type"] == "suspension"
        assert enriched_player["minutes_last6"] > 0
        assert enriched_player["prob_available"] == 0.0
        assert enriched_player["status"] == "suspended"
        assert enriched_player["source_news"]
        assert enriched_player["last_updated"]

    def test_fixture_route_auto_looks_up_team_players_when_omitted(self):
        expected_fdr = self._fake_fdr_response(team="Liverpool", opponent="Spurs", team_id=12, opponent_id=18)
        request_payload = {
            "team": "Liverpool",
            "opponent": "Spurs",
            "is_home": True,
        }

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.post("/api/fdr/fixture", json=request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fdr"] == expected_fdr
        assert body["saturated"]["team"] == "Liverpool"
        assert body["saturated"]["team_id"] == 12
        assert body["saturated"]["opponent"] == "Spurs"
        assert body["saturated"]["opponent_id"] == 18
        assert "team_counted_absences" in body["fdr"]
        assert any(player["player_id"] == 701 for player in body["saturated"]["opp_players"])

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["team_name"] == "Liverpool"
        assert kwargs["opponent_name"] == "Spurs"
        assert kwargs["is_home"] is True
        assert any(player["player_id"] == 701 for player in kwargs["opp_players"])

    def test_fixture_route_team_mode_resolves_fixture_metadata_and_aliases(self):
        expected_fdr = self._fake_fdr_response(team="Liverpool", opponent="Spurs", team_id=12, opponent_id=18)
        request_payload = {
            "team": "Liverpool",
            "opponent": "Tottenham",
            "is_home": True,
        }

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.post("/api/fdr/fixture", json=request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fdr"] == expected_fdr
        assert body["saturated"]["fixture_id"] is not None
        assert body["saturated"]["home_team"] == "Liverpool"
        assert body["saturated"]["home_team_id"] == 12
        assert body["saturated"]["away_team"] == "Spurs"
        assert body["saturated"]["away_team_id"] == 18
        assert body["saturated"]["opponent"] == "Spurs"
        assert any(player["player_id"] == 701 for player in body["saturated"]["opp_players"])

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["team_name"] == "Liverpool"
        assert kwargs["opponent_name"] == "Spurs"
        assert any(player["player_id"] == 701 for player in kwargs["opp_players"])

    def test_fixture_route_resolves_fixture_id_and_saturates_both_teams(self):
        expected_fdr = self._fake_fdr_response(team="Man City", opponent="Spurs", team_id=13, opponent_id=18)
        request_payload = {"fixture_id": 13}

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.post("/api/fdr/fixture", json=request_payload)

        assert response.status_code == 200
        body = response.json()
        assert body["fdr"] == expected_fdr
        assert body["saturated"]["fixture_id"] == 13
        assert body["saturated"]["home_team"] == "Man City"
        assert body["saturated"]["home_team_id"] == 13
        assert body["saturated"]["away_team"] == "Spurs"
        assert body["saturated"]["away_team_id"] == 18
        assert body["saturated"]["team"] == "Man City"
        assert body["saturated"]["team_id"] == 13
        assert body["saturated"]["opponent"] == "Spurs"
        assert body["saturated"]["opponent_id"] == 18
        assert body["saturated"]["is_home"] is True
        assert any(player["player_id"] == 701 for player in body["saturated"]["opp_players"])

        kwargs = mock_compute.call_args.kwargs
        assert kwargs["team_name"] == "Man City"
        assert kwargs["opponent_name"] == "Spurs"
        assert kwargs["is_home"] is True
        assert any(player["player_id"] == 701 for player in kwargs["opp_players"])

    def test_fixture_route_rejects_conflicting_fixture_context(self):
        client = TestClient(app)
        response = client.post(
            "/api/fdr/fixture",
            json={"fixture_id": 13, "team": "Arsenal"},
        )
        assert response.status_code == 400
        assert "does not belong" in response.json()["detail"]

    def test_team_fixtures_route_returns_next_fixture_list_for_team_alias(self):
        expected_responses = [
            self._fake_fdr_response(team="Spurs", opponent="Liverpool", is_home=False, team_id=18, opponent_id=12),
            self._fake_fdr_response(team="Spurs", opponent="Nott'm Forest", is_home=True, team_id=18, opponent_id=16),
        ]

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.side_effect = expected_responses
            client = TestClient(app)
            response = client.get("/api/fdr/team/Tottenham?next=2")

        assert response.status_code == 200
        body = response.json()
        assert isinstance(body, list)
        assert len(body) == 2
        assert body[0]["fdr"]["team_id"] == 18
        assert body[0]["fdr"]["opponent_id"] == 12
        assert body[0]["saturated"]["team"] == "Spurs"
        assert body[0]["saturated"]["team_id"] == 18
        assert body[0]["saturated"]["opponent"] == "Liverpool"
        assert "opp_ignored_absences" in body[0]["fdr"]
        assert body[1]["saturated"]["team"] == "Spurs"
        assert body[1]["saturated"]["team_id"] == 18
        assert body[1]["saturated"]["opponent"] == "Nott'm Forest"
        assert mock_compute.call_count == 2

    def test_team_fixtures_route_accepts_numeric_team_id(self):
        expected_fdr = self._fake_fdr_response(team="Liverpool", opponent="Spurs", team_id=12, opponent_id=18)

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.return_value = expected_fdr
            client = TestClient(app)
            response = client.get("/api/fdr/team/12?next=1")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        assert body[0]["saturated"]["team"] == "Liverpool"
        assert body[0]["saturated"]["team_id"] == 12
        assert body[0]["fdr"]["team_id"] == 12

    def test_team_fixtures_route_next_3_returns_expected_structure(self):
        expected_responses = [
            self._fake_fdr_response(team="Liverpool", opponent="Spurs", team_id=12, opponent_id=18),
            self._fake_fdr_response(team="Liverpool", opponent="Brighton", is_home=False, team_id=12, opponent_id=6),
            self._fake_fdr_response(team="Liverpool", opponent="Fulham", is_home=True, team_id=12, opponent_id=10),
        ]

        with patch("src.services.fixture_fdr.compute_fixture_fdr") as mock_compute:
            mock_compute.side_effect = expected_responses
            client = TestClient(app)
            response = client.get("/api/fdr/team/Liverpool?next=3")

        assert response.status_code == 200
        body = response.json()
        assert len(body) == 3
        assert all(set(item) == {"saturated", "fdr"} for item in body)
        assert body[0]["saturated"]["team"] == "Liverpool"
        assert body[1]["saturated"]["team"] == "Liverpool"
        assert body[2]["saturated"]["team"] == "Liverpool"

    def test_team_fixtures_route_returns_404_for_unknown_team(self):
        client = TestClient(app)
        response = client.get("/api/fdr/team/not-a-team?next=2")
        assert response.status_code == 404
        assert "Unknown team" in response.json()["detail"]

