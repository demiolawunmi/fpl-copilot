from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient
import pytest

import src.main as main_module
from src.main import app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setattr(main_module, "_get_copilot_job_service", lambda: object())
    return TestClient(app)


_BLEND_BODY = {
    "schema_version": "1.0",
    "correlation_id": "corr-chat",
    "message": "Who should I captain?",
    "messages": [],
    "blend_input": {
        "schema_version": "1.0",
        "correlation_id": "corr-blend",
        "source_weights": {"elo": 0.6, "airsenal": 0.4},
        "gameweek": 18,
        "task": "hybrid",
        "force_refresh": False,
    },
    "blend_result": {
        "schema_version": "1.0",
        "correlation_id": "corr-blend",
        "core": {"summary": "Balanced squad", "confidence": 0.75},
        "recommended_transfers": [],
        "ask_copilot": {
            "answer": "Consider your premium striker.",
            "rationale": ["Form + fixtures"],
            "confidence": 0.7,
        },
        "degraded_mode": {
            "is_degraded": False,
            "code": None,
            "message": None,
            "fallback_used": False,
        },
    },
}


def test_copilot_chat_returns_answer(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    from src.services import copilot_chat_service as chat_mod

    monkeypatch.setattr(
        chat_mod,
        "run_copilot_chat",
        lambda **kwargs: "Salah looks strong this week.",
    )

    response = client.post("/api/copilot/chat", json=_BLEND_BODY)
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == "Salah looks strong this week."
    assert data["correlation_id"] == "corr-chat"


def test_copilot_chat_503_when_service_raises(
    monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    from src.services.copilot_chat_service import CopilotChatError
    from src.services import copilot_chat_service as chat_mod

    def _fail(**kwargs: object) -> str:
        raise CopilotChatError("LLM down")

    monkeypatch.setattr(chat_mod, "run_copilot_chat", _fail)

    response = client.post("/api/copilot/chat", json=_BLEND_BODY)
    assert response.status_code == 503
    assert response.json()["detail"] == "LLM down"
