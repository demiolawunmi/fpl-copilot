"""Tests for POST /api/airsenal/run (mocked subprocess)."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.services.airsenal_runner import AirsenalRunResponse


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _ok_response() -> AirsenalRunResponse:
    return AirsenalRunResponse(
        ok=True,
        action="update_db",
        steps=[{"step": "update_db", "returncode": 0, "stdout": "", "stderr": ""}],
    )


def test_run_success_without_api_key(client: TestClient) -> None:
    os.environ.pop("AIRSENAL_RUN_API_KEY", None)
    with patch(
        "src.main.run_airsenal_action",
        return_value=_ok_response(),
    ):
        r = client.post("/api/airsenal/run", json={"action": "update_db"})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["action"] == "update_db"


def test_run_requires_key_when_configured(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AIRSENAL_RUN_API_KEY", "secret")
    r = client.post("/api/airsenal/run", json={"action": "update_db"})
    assert r.status_code == 401

    with patch("src.main.run_airsenal_action", return_value=_ok_response()):
        r = client.post(
            "/api/airsenal/run",
            json={"action": "update_db"},
            headers={"X-Airsenal-Run-Key": "secret"},
        )
    assert r.status_code == 200


def test_run_validation_error_for_optimize_without_team(client: TestClient) -> None:
    from src.services.airsenal_runner import AirsenalRunError

    os.environ.pop("AIRSENAL_RUN_API_KEY", None)
    with patch(
        "src.main.run_airsenal_action",
        side_effect=AirsenalRunError(
            "fpl_team_id is required for optimize (or set .airsenal_home/FPL_TEAM_ID)"
        ),
    ):
        r = client.post("/api/airsenal/run", json={"action": "optimize"})
    assert r.status_code == 400
    assert "required" in r.json()["detail"].lower()
