"""Tests for official FPL fixture difficulty ingestion (fpl_official_fdr)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.main import app
from src.services.fpl_official_fdr import (
    build_fixture_difficulty_lookup,
    build_official_fpl_fields,
    get_official_fpl_difficulty_for_team,
    resolve_fpl_fixture_for_context,
)
from fastapi.testclient import TestClient

SAMPLE_RAW = [
    {
        "id": 1,
        "event": 31,
        "kickoff_time": "2026-04-04T15:00:00Z",
        "team_h": 13,
        "team_a": 8,
        "team_h_difficulty": 5,
        "team_a_difficulty": 2,
        "finished": False,
        "started": False,
        "team_h_score": None,
        "team_a_score": None,
    },
    {
        "id": 2,
        "event": 31,
        "kickoff_time": "2026-04-04T17:30:00Z",
        "team_h": 14,
        "team_a": 15,
        "team_h_difficulty": 3,
        "team_a_difficulty": 3,
        "finished": False,
        "started": False,
        "team_h_score": None,
        "team_a_score": None,
    },
]


def _norm_list(raw: list) -> list:
    from src.services.fpl_official_fdr import normalize_fpl_fixture

    return [normalize_fpl_fixture(x) for x in raw]


class TestFetchFplFixtures:
    def test_fetch_parses_json_array(self) -> None:
        import src.services.fpl_official_fdr as mod

        mod._fetch_ts = 0.0
        mod._cached_raw = []
        mod._stale_raw = []

        payload = [{"id": 42, "team_h": 3, "team_a": 4, "event": 1, "team_h_difficulty": 3, "team_a_difficulty": 3}]
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps(payload).encode("utf-8")
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = None

        with patch("urllib.request.urlopen", return_value=mock_cm):
            out = mod.fetch_fpl_fixtures()

        assert out == payload


class TestBuildFixtureDifficultyLookup:
    def test_fixture_id_to_difficulty(self):
        fixtures = _norm_list(SAMPLE_RAW)
        by_id, _ = build_fixture_difficulty_lookup(fixtures)
        assert by_id[1]["team_h_difficulty"] == 5
        assert by_id[1]["team_a_difficulty"] == 2

    def test_team_event_key(self):
        fixtures = _norm_list(SAMPLE_RAW)
        _, by_h_a_event = build_fixture_difficulty_lookup(fixtures)
        assert (13, 8, 31) in by_h_a_event


class TestSideAwareDifficulty:
    def test_home_uses_team_h_difficulty(self):
        fx = _norm_list(SAMPLE_RAW)[0]
        assert get_official_fpl_difficulty_for_team(fx, 13, True) == 5

    def test_away_uses_team_a_difficulty(self):
        fx = _norm_list(SAMPLE_RAW)[0]
        assert get_official_fpl_difficulty_for_team(fx, 8, False) == 2


class TestResolveFplFixtureForContext:
    def test_by_fixture_id(self):
        fixtures = _norm_list(SAMPLE_RAW)
        by_id, by_h_a_event = build_fixture_difficulty_lookup(fixtures)
        ctx = {
            "fixture_id": 1,
            "home_team_id": 13,
            "away_team_id": 8,
            "gameweek": 31,
            "team_id": 8,
            "is_home": False,
        }
        found = resolve_fpl_fixture_for_context(fixtures, by_id, by_h_a_event, ctx)
        assert found is not None
        assert found["id"] == 1

    def test_by_teams_and_gameweek(self):
        fixtures = _norm_list(SAMPLE_RAW)
        by_id, by_h_a_event = build_fixture_difficulty_lookup(fixtures)
        ctx = {
            "fixture_id": 99999,
            "home_team_id": 13,
            "away_team_id": 8,
            "gameweek": 31,
            "team_id": 13,
            "is_home": True,
        }
        found = resolve_fpl_fixture_for_context(fixtures, by_id, by_h_a_event, ctx)
        assert found is not None
        assert found["team_h_difficulty"] == 5

    def test_team_id_join_preferred_documented(self):
        """Lookup uses FPL team ids on the fixture row (team_h / team_a)."""
        fixtures = _norm_list(SAMPLE_RAW)
        by_id, by_h_a_event = build_fixture_difficulty_lookup(fixtures)
        ctx = {"home_team_id": 14, "away_team_id": 15, "gameweek": 31}
        found = resolve_fpl_fixture_for_context(fixtures, by_id, by_h_a_event, ctx)
        assert found["id"] == 2


class TestBuildOfficialFplFields:
    def test_empty_fetch_returns_unavailable(self):
        with patch("src.services.fpl_official_fdr.fetch_fpl_fixtures", return_value=[]):
            out = build_official_fpl_fields(
                {
                    "fixture_id": 1,
                    "home_team_id": 13,
                    "away_team_id": 8,
                    "team_id": 13,
                    "is_home": True,
                    "gameweek": 31,
                }
            )
        assert out["official_fpl_source"] == "unavailable"
        assert out["official_fpl_fdr"] is None

    def test_merged_fields_when_fetch_ok(self):
        with patch("src.services.fpl_official_fdr.fetch_fpl_fixtures", return_value=SAMPLE_RAW):
            out = build_official_fpl_fields(
                {
                    "fixture_id": 1,
                    "home_team_id": 13,
                    "away_team_id": 8,
                    "team_id": 8,
                    "is_home": False,
                    "gameweek": 31,
                }
            )
        assert out["official_fpl_source"] == "fpl_api"
        assert out["official_fpl_fdr"] == 2
        assert out["official_fpl_home_difficulty"] == 5
        assert out["official_fpl_away_difficulty"] == 2
        assert out["official_fpl_event"] == 31
        assert out["official_fpl_kickoff_time"] == "2026-04-04T15:00:00Z"


class TestOfficialFplEndpointIntegration:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_post_fixture_includes_official_when_mocked(self, client: TestClient):
        with patch("src.services.fpl_official_fdr.fetch_fpl_fixtures", return_value=SAMPLE_RAW):
            r = client.post(
                "/api/fdr/fixture",
                json={"fixture_id": 1, "team": "Man City", "opponent": "Crystal Palace", "is_home": True},
            )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "saturated" in body and "fdr" in body
        assert body["saturated"]["official_fpl_source"] == "fpl_api"
        assert body["saturated"]["official_fpl_fdr"] == 5
        assert body["saturated"]["official_fpl_home_difficulty"] == 5
        assert body["saturated"]["official_fpl_away_difficulty"] == 2
        assert "attack_fdr" in body["fdr"]
        assert "overall_fdr" in body["fdr"]

    def test_post_fpl_failure_still_returns_custom_fdr(self, client: TestClient):
        with patch("src.services.fpl_official_fdr.fetch_fpl_fixtures", return_value=[]):
            r = client.post(
                "/api/fdr/fixture",
                json={"fixture_id": 1, "team": "Man City", "opponent": "Crystal Palace", "is_home": True},
            )
        assert r.status_code == 200
        body = r.json()
        assert body["saturated"]["official_fpl_source"] == "unavailable"
        assert body["saturated"]["official_fpl_fdr"] is None
        assert body["fdr"]["overall_fdr"] is not None

    def test_team_next_includes_official_per_fixture(self, client: TestClient):
        with patch("src.services.fpl_official_fdr.fetch_fpl_fixtures", return_value=SAMPLE_RAW):
            r = client.get("/api/fdr/team/Man%20City", params={"next": 1})
        assert r.status_code == 200, r.text
        data = r.json()
        assert isinstance(data, list) and len(data) >= 1
        sat = data[0]["saturated"]
        assert "official_fpl_fdr" in sat
        assert "official_fpl_source" in sat
