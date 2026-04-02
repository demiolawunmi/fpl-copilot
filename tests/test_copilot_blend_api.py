from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import pytest

import src.main as main_module
from src.main import app


class _StubCopilotJobService:
    def submit_job(self, submit_payload):
        return {
            "schema_version": submit_payload["schema_version"],
            "correlation_id": submit_payload["correlation_id"],
            "job_id": "job-123",
            "status": "queued",
        }

    def get_job_status(self, job_id: str):
        if job_id == "job-queued":
            return {
                "schema_version": "1.0",
                "correlation_id": "corr-queued",
                "job_id": "job-queued",
                "status": "queued",
                "result_json": None,
                "error_json": None,
            }
        if job_id == "job-completed":
            return {
                "schema_version": "1.0",
                "correlation_id": "corr-done",
                "job_id": "job-completed",
                "status": "completed",
                "result_json": {
                    "schema_version": "1.0",
                    "correlation_id": "corr-done",
                    "core": {"summary": "Ready", "confidence": 0.8},
                    "recommended_transfers": [],
                    "ask_copilot": {
                        "answer": "Hold transfer.",
                        "rationale": ["Small edge"],
                        "confidence": 0.7,
                    },
                    "degraded_mode": {
                        "is_degraded": False,
                        "code": None,
                        "message": None,
                        "fallback_used": False,
                    },
                },
                "error_json": None,
            }
        return None


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: _StubCopilotJobService())
    return TestClient(app)


def test_submit_blend_job_returns_202_accepted_payload(client: TestClient) -> None:
    response = client.post(
        "/api/copilot/blend-jobs",
        json={
            "schema_version": "1.0",
            "correlation_id": "corr-submit",
            "source_weights": {"fplcopilot": 0.6, "airsenal": 0.4},
            "task": "hybrid",
            "force_refresh": False,
        },
    )

    assert response.status_code == 202
    assert response.json() == {
        "schema_version": "1.0",
        "correlation_id": "corr-submit",
        "job_id": "job-123",
        "status": "queued",
    }


def test_submit_blend_job_invalid_payload_returns_422(client: TestClient) -> None:
    response = client.post(
        "/api/copilot/blend-jobs",
        json={
            "schema_version": "1.0",
            "correlation_id": "corr-bad",
            "source_weights": {"fplcopilot": 0.3, "airsenal": 0.3},
            "task": "hybrid",
        },
    )

    assert response.status_code == 422


def test_get_blend_job_returns_typed_status_payload(client: TestClient) -> None:
    response = client.get("/api/copilot/blend-jobs/job-completed")

    assert response.status_code == 200
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["correlation_id"] == "corr-done"
    assert body["job_id"] == "job-completed"
    assert body["status"] == "completed"
    assert body["result"]["core"]["summary"] == "Ready"
    assert body["error"] is None


def test_get_blend_job_unknown_job_returns_structured_404(client: TestClient) -> None:
    response = client.get("/api/copilot/blend-jobs/job-missing")

    assert response.status_code == 404
    body = response.json()
    assert body["schema_version"] == "1.0"
    assert body["correlation_id"] == "job-missing"
    assert body["error"]["code"] == "NOT_FOUND"
    assert "not found" in body["error"]["message"].lower()
    assert "traceback" not in str(body).lower()
