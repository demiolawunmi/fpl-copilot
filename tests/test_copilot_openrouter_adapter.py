from __future__ import annotations

import json
import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_openrouter_adapter import CopilotOpenRouterAdapter


class _Response:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _Session:
    def __init__(self, outcomes: list[_Response | Exception]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, object]] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, object], timeout: int):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


_CTX = {
    "schema_version": "1.0",
    "weights": {"elo": 0.7, "airsenal": 0.3},
    "gameweek": 31,
    "bank": 1.5,
    "free_transfers": 2,
    "sources": ["elo", "airsenal"],
    "current_squad": [
        {
            "player_id": 101,
            "fpl_api_id": 16,
            "player_name": "Saka",
            "team": "Arsenal",
            "position": "MID",
            "price": 10.2,
            "x_pts": 8.0,
            "elo_score": 1650.0,
            "airsenal_predicted_points": 8.0,
        }
    ],
    "blended_players": [
        {
            "player_id": 101,
            "player_name": "Saka",
            "team": "ARS",
            "position": "MID",
            "elo_score": 1650.0,
            "airsenal_predicted_points": 8.0,
        }
    ],
}


def test_openrouter_success_payload(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("OPENROUTER_MODEL", "google/gemma-2-9b-it:free")
    session = _Session(
        outcomes=[
            _Response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    {
                                        "core": {"summary": "Ready", "confidence": 0.8},
                                        "recommended_transfers": [],
                                        "ask_copilot": {
                                            "answer": "Hold",
                                            "rationale": ["small edge"],
                                            "confidence": 0.7,
                                        },
                                    }
                                )
                            }
                        }
                    ]
                }
            )
        ]
    )

    adapter = CopilotOpenRouterAdapter(session=session, sleep_fn=lambda _: None)
    result = adapter.generate_hybrid_payload(
        schema_version="1.0", correlation_id="corr-openrouter", model_context=_CTX
    )

    assert result["degraded_mode"]["is_degraded"] is False
    assert result["core"]["summary"] == "Ready"
    payload = session.calls[0]["json"]
    assert isinstance(payload, dict)
    assert payload.get("model") == "google/gemma-2-9b-it:free"
    messages = payload.get("messages")
    assert isinstance(messages, list)
    content = messages[0]["content"]
    assert "Target gameweek: 31" in content
    assert "Available bank: £1.5m" in content
    assert "Free transfers remaining: 2" in content


def test_openrouter_missing_api_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        CopilotOpenRouterAdapter()
        assert False, "expected runtime error"
    except RuntimeError as exc:
        assert "OPENROUTER_API_KEY" in str(exc)


def test_openrouter_invalid_json_degrades(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    session = _Session(
        outcomes=[
            _Response({"choices": [{"message": {"content": "not json"}}]}),
            _Response({"choices": [{"message": {"content": "still not json"}}]}),
            _Response({"choices": [{"message": {"content": "[]"}}]}),
        ]
    )
    adapter = CopilotOpenRouterAdapter(session=session, sleep_fn=lambda _: None)
    result = adapter.generate_hybrid_payload(
        schema_version="1.0", correlation_id="corr-openrouter-degraded", model_context=_CTX
    )
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
