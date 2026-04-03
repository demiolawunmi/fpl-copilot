from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import pytest
from pydantic import ValidationError

from src.main import (
    CopilotBlendSubmitRequest,
    CopilotHybridResultPayload,
    app,
)


def test_submit_contract_accepts_valid_payload() -> None:
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-123",
        "source_weights": {"elo": 0.6, "airsenal": 0.4},
        "task": "hybrid",
        "force_refresh": False,
    }

    model = CopilotBlendSubmitRequest.model_validate(payload)

    assert model.schema_version == "1.0"
    assert model.correlation_id == "corr-123"
    assert model.source_weights.elo == pytest.approx(0.6)
    assert model.source_weights.airsenal == pytest.approx(0.4)


def test_submit_contract_rejects_invalid_weights_sum() -> None:
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-123",
        "source_weights": {"elo": 0.7, "airsenal": 0.4},
        "task": "hybrid",
    }

    with pytest.raises(ValidationError) as exc_info:
        CopilotBlendSubmitRequest.model_validate(payload)

    assert "must sum to 1.0" in str(exc_info.value)


def test_hybrid_contract_accepts_valid_payload() -> None:
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-789",
        "core": {
            "summary": "Balanced squad with one priority move",
            "confidence": 0.82,
        },
        "recommended_transfers": [
            {
                "transfer_id": "t1",
                "out": {"player_id": 101, "player_name": "Player Out"},
                "in": {"player_id": 202, "player_name": "Player In"},
                "reason": "Improves expected points",
                "projected_points_delta": 4.6,
            }
        ],
        "ask_copilot": {
            "answer": "Make the transfer this week.",
            "rationale": ["Form uplift", "Fixture swing"],
            "confidence": 0.77,
        },
        "degraded_mode": {
            "is_degraded": False,
            "fallback_used": False,
        },
    }

    model = CopilotHybridResultPayload.model_validate(payload)

    assert model.schema_version == "1.0"
    assert model.correlation_id == "corr-789"
    assert model.core.summary
    assert model.recommended_transfers[0].out.player_id == 101
    assert model.ask_copilot.answer


def test_hybrid_contract_rejects_invalid_transfer_type() -> None:
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-789",
        "core": {
            "summary": "Balanced squad with one priority move",
            "confidence": 0.82,
        },
        "recommended_transfers": [
            {
                "transfer_id": "t1",
                "out": {"player_id": "bad", "player_name": "Player Out"},
                "in": {"player_id": 202, "player_name": "Player In"},
                "reason": "Improves expected points",
                "projected_points_delta": 4.6,
            }
        ],
        "ask_copilot": {
            "answer": "Make the transfer this week.",
            "rationale": ["Form uplift", "Fixture swing"],
            "confidence": 0.77,
        },
        "degraded_mode": {
            "is_degraded": True,
            "code": "SCHEMA_VALIDATION_FAILED",
            "message": "Fallback used",
            "fallback_used": True,
        },
    }

    with pytest.raises(ValidationError):
        CopilotHybridResultPayload.model_validate(payload)


def test_invalid_submit_payload_is_rejected_by_fastapi_validation() -> None:
    client = TestClient(app)

    response = client.post(
        "/api/copilot/blend-jobs",
        json={
            "schema_version": "1.0",
            "correlation_id": "corr-123",
            "source_weights": {"elo": 0.2, "airsenal": 0.2},
            "task": "hybrid",
        },
    )

    assert response.status_code == 422


def test_openapi_exposes_copilot_contract_schemas() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()

    submit_path = schema["paths"]["/api/copilot/blend-jobs"]["post"]
    status_path = schema["paths"]["/api/copilot/blend-jobs/{job_id}"]["get"]

    submit_ref = submit_path["requestBody"]["content"]["application/json"]["schema"]["$ref"]
    status_ref = status_path["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
    hybrid_ref = schema["components"]["schemas"]["CopilotBlendJobStatusResponse"]["properties"]["result"]["anyOf"][0]["$ref"]

    assert submit_ref.endswith("/CopilotBlendSubmitRequest")
    assert status_ref.endswith("/CopilotBlendJobStatusResponse")
    assert hybrid_ref.endswith("/CopilotHybridResultPayload")


def test_degraded_mode_code_none_when_not_degraded() -> None:
    """When is_degraded is False, code must be None."""
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-789",
        "core": {"summary": "All good", "confidence": 0.9},
        "recommended_transfers": [],
        "ask_copilot": {"answer": "Hold", "rationale": ["No edge"], "confidence": 0.9},
        "degraded_mode": {"is_degraded": False, "code": None, "fallback_used": False},
    }

    model = CopilotHybridResultPayload.model_validate(payload)
    assert model.degraded_mode.is_degraded is False
    assert model.degraded_mode.code is None
    assert model.degraded_mode.fallback_used is False


def test_degraded_payload_schema_defaults_for_missing_optional_fields() -> None:
    """Degraded payload with minimal degraded_mode fields validates correctly."""
    payload = {
        "schema_version": "1.0",
        "correlation_id": "corr-degraded",
        "core": {"summary": "Degraded", "confidence": 0.0},
        "recommended_transfers": [],
        "ask_copilot": {"answer": "Retry", "rationale": ["Error"], "confidence": 0.0},
        "degraded_mode": {"is_degraded": True, "code": "FALLBACK", "fallback_used": True},
    }

    model = CopilotHybridResultPayload.model_validate(payload)
    assert model.degraded_mode.is_degraded is True
    assert model.degraded_mode.code == "FALLBACK"
    assert model.degraded_mode.fallback_used is True
    assert model.degraded_mode.message is None
