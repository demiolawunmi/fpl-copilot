from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_gemini_adapter import CopilotGeminiAdapter


class _Response:
    def __init__(self, text: str) -> None:
        self.text = text


class _ModelsAPI:
    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self._outcomes = outcomes
        self._call_log = call_log

    def generate_content(self, *, model, contents, config):
        self._call_log.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _Client:
    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self.models = _ModelsAPI(outcomes=outcomes, call_log=call_log)


_ELO_CONTEXT = {
    "schema_version": "1.0",
    "weights": {"elo": 0.7, "airsenal": 0.3},
    "sources": ["elo", "airsenal"],
    "blended_players": [
        {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 8.0},
        {"player_id": 202, "player_name": "Haaland", "team": "MCI", "position": "FWD", "elo_score": 1850.5, "airsenal_predicted_points": 11.0},
    ],
}


def test_success_with_valid_gemini_response() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                """
                {
                    "core": {"summary": "Blend complete", "confidence": 0.82},
                    "recommended_transfers": [
                        {
                            "transfer_id": "t1",
                            "out": {"player_id": 11, "player_name": "Player Out"},
                            "in": {"player_id": 22, "player_name": "Player In"},
                            "reason": "Better projected minutes",
                            "projected_points_delta": 1.9
                        }
                    ],
                    "ask_copilot": {
                        "answer": "Prioritize value and minutes.",
                        "rationale": ["Weighted projection favored transfer"],
                        "confidence": 0.8
                    }
                }
                """
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-ok",
        model_context=_ELO_CONTEXT,
    )

    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-ok"
    assert result["degraded_mode"]["is_degraded"] is False
    assert result["recommended_transfers"][0]["in"]["player_id"] == 22
    assert call_log[0]["model"] == "gemini-2.5-flash"
    assert call_log[0]["config"]["response_mime_type"] == "application/json"


def test_prompt_includes_elo_weight_instructions() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-prompt",
        model_context=_ELO_CONTEXT,
    )

    prompt = call_log[0]["contents"]
    assert "ELO scores are weighted at 70%" in prompt
    assert "AIrsenal predictions at 30%" in prompt
    assert "Saka (ARS, MID): ELO=1650.0, AIrsenal=8.0" in prompt
    assert "Haaland (MCI, FWD): ELO=1850.5, AIrsenal=11.0" in prompt


def test_prompt_handles_elo_only_weighting() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-elo-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 1.0, "airsenal": 0.0},
            "sources": ["elo"],
            "blended_players": [
                {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 1650.0, "airsenal_predicted_points": 0.0},
            ],
        },
    )

    prompt = call_log[0]["contents"]
    assert "ELO scores are the sole signal (100%)" in prompt


def test_prompt_handles_airsenal_only_weighting() -> None:
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-airsenal-only",
        model_context={
            "schema_version": "1.0",
            "weights": {"elo": 0.0, "airsenal": 1.0},
            "sources": ["airsenal"],
            "blended_players": [
                {"player_id": 101, "player_name": "Saka", "team": "ARS", "position": "MID", "elo_score": 0.0, "airsenal_predicted_points": 8.0},
            ],
        },
    )

    prompt = call_log[0]["contents"]
    assert "AIrsenal predictions are the sole signal (100%)" in prompt


def test_timeout_retries_then_returns_structured_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[TimeoutError("t1"), TimeoutError("t2"), TimeoutError("t3")],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-timeout",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-timeout"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "LLM_TIMEOUT"
    assert result["degraded_mode"]["fallback_used"] is True


def _make_adapter_for_extract_tests():
    # Use the lightweight _Client stub so we don't trigger default client creation
    call_log: list[dict] = []
    return CopilotGeminiAdapter(client=_Client(outcomes=['{}'], call_log=call_log), sleep_fn=lambda _: None)


def test_extract_plain_json_only() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = '  {"x": 1, "y": "z"}  '\
        "\n"
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"x": 1, "y": "z"}


def test_extract_json_in_json_code_block() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = "Here you go:\n```json\n{\"a\": 1, \"b\": [1,2,3]}\n```\n"
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"a": 1, "b": [1, 2, 3]}


def test_extract_json_in_plain_code_block() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = "Some text\n```\n{\"c\": {\"d\": \"val\"}}\n```\n"
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"c": {"d": "val"}}


def test_extract_json_with_surrounding_text_and_nested_structures() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = (
        "Note:\n```json\n{\"outer\": {\"inner\": {\"val\": \"string with } brace\"}, \"num\": 3}}\n```\nThanks"
    )
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"outer": {"inner": {"val": "string with } brace"}, "num": 3}}


def test_extract_json_brace_counting_with_escaped_quotes_and_trailing_text() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = 'prefix text {"a": "escaped \\" quote", "b": {"c": 1}} trailing markdown ```'
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"a": "escaped \" quote", "b": {"c": 1}}


def test_malformed_json_retries_then_returns_schema_validation_failed_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=["not-json", "still-not-json", "[]"],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-malformed",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-malformed"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["ask_copilot"]["confidence"] == 0.0


def test_provider_error_retries_then_returns_provider_degraded_payload() -> None:
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[RuntimeError("upstream down"), RuntimeError("upstream down"), RuntimeError("upstream down")],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-provider",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["schema_version"] == "1.0"
    assert result["correlation_id"] == "corr-provider"
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "PROVIDER_ERROR"
    assert result["degraded_mode"]["fallback_used"] is True


def test_extract_json_in_code_block_with_crlf_line_endings() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = "Here you go:\r\n```json\r\n{\"a\": 1, \"b\": 2}\r\n```\r\nDone"
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"a": 1, "b": 2}


def test_extract_json_in_plain_code_block_with_crlf() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = "Some text\r\n```\r\n{\"c\": 3}\r\n```\r\n"
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"c": 3}


def test_extract_json_truncated_starts_with_brace() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = '{"key": "unterminated string'
    with __import__('pytest').raises(ValueError, match="Unterminated JSON"):
        adapter._extract_json(raw)


def test_extract_json_malformed_starts_with_brace() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = '{"key": "value", trailing}'
    with __import__('pytest').raises(ValueError):
        adapter._extract_json(raw)


def test_extract_json_no_brace_at_all() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = "This is just plain text with no JSON"
    with __import__('pytest').raises(ValueError, match="No JSON object found"):
        adapter._extract_json(raw)


def test_extract_json_malformed_in_code_block_falls_to_brace_counting() -> None:
    adapter = _make_adapter_for_extract_tests()
    raw = 'prefix {"valid": true} suffix'
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"valid": True}


# ---------------------------------------------------------------------------
# Timeout wiring tests
# ---------------------------------------------------------------------------

class _ModelsAPIWithTimeout:
    """Stub that accepts timeout and records it."""

    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self._outcomes = outcomes
        self._call_log = call_log

    def generate_content(self, *, model, contents, config, timeout):
        self._call_log.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
                "timeout": timeout,
            }
        )
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return _Response(outcome)


class _ClientWithTimeout:
    def __init__(self, outcomes: list[object], call_log: list[dict]) -> None:
        self.models = _ModelsAPIWithTimeout(outcomes=outcomes, call_log=call_log)


def test_adapter_passes_timeout_to_client() -> None:
    """Verify the adapter forwards a numeric timeout to the provider API."""
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_ClientWithTimeout(
            outcomes=['{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}'],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-timeout-wired",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 1
    assert "timeout" in call_log[0]
    assert isinstance(call_log[0]["timeout"], int)
    assert call_log[0]["timeout"] > 0
    assert call_log[0]["timeout"] == 25  # default GeminiAdapterConfig.timeout_seconds


def test_adapter_timeout_survives_retry_after_timeout_error() -> None:
    """After TimeoutError retries, the final successful call still carries timeout."""
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_ClientWithTimeout(
            outcomes=[
                TimeoutError("slow-1"),
                TimeoutError("slow-2"),
                '{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5}}',
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-timeout-retry",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    for entry in call_log:
        assert entry["timeout"] == 25
    assert result["degraded_mode"]["is_degraded"] is False


# ---------------------------------------------------------------------------
# No-JSON-found degraded flow
# ---------------------------------------------------------------------------

def test_extract_no_json_found_returns_error_and_triggers_degraded_flow() -> None:
    """Prose-only response with no JSON present → retries → SCHEMA_VALIDATION_FAILED."""
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                "I'm sorry, I cannot provide JSON data.",
                "Here is some text without any JSON structures at all.",
                "Still no JSON, just plain English prose.",
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-no-json",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert result["degraded_mode"]["fallback_used"] is True
    assert result["ask_copilot"]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# Multiple code blocks
# ---------------------------------------------------------------------------

def test_extract_handles_multiple_json_code_blocks_chooses_first_valid() -> None:
    """Multiple code blocks: first invalid, second valid → falls through to brace counting and finds valid JSON."""
    adapter = _make_adapter_for_extract_tests()
    raw = (
        "Here are two blocks:\n"
        "```json\n"
        "{invalid json here}\n"
        "```\n"
        "And the real one:\n"
        "```json\n"
        "{\"valid\": true, \"count\": 42}\n"
        "```\n"
    )
    with __import__('pytest').raises(ValueError, match="Malformed JSON extracted"):
        adapter._extract_json(raw)


def test_extract_multiple_code_blocks_first_valid_wins() -> None:
    """First code block is valid JSON → should be selected immediately."""
    adapter = _make_adapter_for_extract_tests()
    raw = (
        "```json\n"
        "{\"first\": \"block\"}\n"
        "```\n"
        "```json\n"
        "{\"second\": \"block\"}\n"
        "```\n"
    )
    extracted = adapter._extract_json(raw)
    assert __import__('json').loads(extracted) == {"first": "block"}


# ---------------------------------------------------------------------------
# Schema compliance: extra fields rejected
# ---------------------------------------------------------------------------

def test_response_with_extra_fields_is_stripped_by_validation() -> None:
    """Gemini returns extra fields → _validate_hybrid_payload strips them via selective .get() reconstruction."""
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                '{"core":{"summary":"ok","confidence":0.5},"recommended_transfers":[],"ask_copilot":{"answer":"ok","rationale":[],"confidence":0.5},"extra_field":"should not be here"}'
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-extra-fields",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 1
    assert result["degraded_mode"]["is_degraded"] is False
    assert "extra_field" not in result


# ---------------------------------------------------------------------------
# Malformed JSON fallback sets ask_copilot confidence to zero
# ---------------------------------------------------------------------------

def test_malformed_json_fallback_sets_ask_copilot_confidence_zero() -> None:
    """Partial structure missing ask_copilot → degraded payload with confidence 0.0."""
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                '{"core":{"summary":"partial"},"recommended_transfers":[]}',
                '{"core":{"summary":"partial"},"recommended_transfers":[]}',
                '{"core":{"summary":"partial"},"recommended_transfers":[]}',
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-partial-structure",
        model_context=_ELO_CONTEXT,
    )

    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert result["ask_copilot"]["confidence"] == 0.0
    assert result["ask_copilot"]["answer"] == "Model output format was invalid. Retry shortly."


# ---------------------------------------------------------------------------
# Provider returns error string (non-exception)
# ---------------------------------------------------------------------------

def test_provider_returns_error_string_triggers_provider_degraded() -> None:
    """Provider returns a plain-text error body → treated as provider error after retries."""
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                "Service Unavailable",
                "503 Service Unavailable",
                "Internal Server Error",
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-error-string",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"


# ---------------------------------------------------------------------------
# Valid-zero non-degraded outcome
# ---------------------------------------------------------------------------

def test_valid_zero_non_degraded_outcome_is_not_marked_degraded() -> None:
    """Zero transfers + zero confidence but properly formed → NOT degraded."""
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                '{"core":{"summary":"No transfers recommended","confidence":0.0},"recommended_transfers":[],"ask_copilot":{"answer":"Hold your team.","rationale":["No clear edge"],"confidence":0.0}}'
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-valid-zero",
        model_context=_ELO_CONTEXT,
    )

    assert result["degraded_mode"]["is_degraded"] is False
    assert result["degraded_mode"]["fallback_used"] is False
    assert result["degraded_mode"]["code"] is None
    assert result["recommended_transfers"] == []
    assert result["core"]["confidence"] == 0.0
    assert result["ask_copilot"]["confidence"] == 0.0


# ---------------------------------------------------------------------------
# CRLF markdown parsing in full adapter flow
# ---------------------------------------------------------------------------

def test_crlf_markdown_in_full_adapter_flow() -> None:
    """Gemini returns CRLF-terminated markdown → adapter extracts JSON correctly."""
    call_log: list[dict] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                "Here is the result:\r\n```json\r\n{\"core\":{\"summary\":\"CRLF test\",\"confidence\":0.9},\"recommended_transfers\":[],\"ask_copilot\":{\"answer\":\"Good\",\"rationale\":[\"CRLF handled\"],\"confidence\":0.9}}\r\n```\r\n"
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda _: None,
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-crlf",
        model_context=_ELO_CONTEXT,
    )

    assert result["degraded_mode"]["is_degraded"] is False
    assert result["core"]["summary"] == "CRLF test"
    assert result["core"]["confidence"] == 0.9


# ---------------------------------------------------------------------------
# Truncated JSON handling
# ---------------------------------------------------------------------------

def test_truncated_json_triggers_schema_validation_failed_after_retries() -> None:
    """Truncated JSON response → retries → SCHEMA_VALIDATION_FAILED degraded payload."""
    call_log: list[dict] = []
    sleeps: list[float] = []
    adapter = CopilotGeminiAdapter(
        client=_Client(
            outcomes=[
                '{"core":{"summary":"truncated","confidence":0.5},"recommended_transfers":[',
                '{"core":{"summary":"truncated","confidence":0.5},"recommended_transfers":[',
                '{"core":{"summary":"truncated","confidence":0.5},"recommended_transfers":[',
            ],
            call_log=call_log,
        ),
        sleep_fn=lambda seconds: sleeps.append(seconds),
    )

    result = adapter.generate_hybrid_payload(
        schema_version="1.0",
        correlation_id="corr-truncated",
        model_context=_ELO_CONTEXT,
    )

    assert len(call_log) == 3
    assert sleeps == [1, 2]
    assert result["degraded_mode"]["is_degraded"] is True
    assert result["degraded_mode"]["code"] == "SCHEMA_VALIDATION_FAILED"
    assert result["ask_copilot"]["confidence"] == 0.0
