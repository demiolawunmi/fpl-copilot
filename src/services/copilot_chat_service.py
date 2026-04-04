from __future__ import annotations

import json
import logging
import os
from typing import Any, Mapping

from src.services.copilot_gemini_adapter import CopilotGeminiAdapter
from src.services.copilot_openrouter_adapter import CopilotOpenRouterAdapter

logger = logging.getLogger(__name__)


class CopilotChatError(RuntimeError):
    """Raised when the chat LLM call fails after retries."""


_MAX_BLEND_INPUT_JSON = 32_000
_MAX_BLEND_RESULT_JSON = 48_000
_MAX_USER_MESSAGE = 4_000
_MAX_PRIOR_MESSAGES = 30


def _truncate_json_blob(label: str, payload: Mapping[str, Any], max_chars: int) -> str:
    raw = json.dumps(dict(payload), ensure_ascii=False, indent=2)
    if len(raw) <= max_chars:
        return raw
    return raw[: max_chars - 40] + f"\n...[{label} truncated, {len(raw)} chars total]"


def build_copilot_chat_system_prompt(
    blend_input: Mapping[str, Any],
    blend_result: Mapping[str, Any],
) -> str:
    instructions = (
        "You are FPL Copilot, a Fantasy Premier League assistant.\n"
        "The user has already run a model blend (ELO + AIrsenal). Answer follow-up questions using "
        "ONLY the blend context below. Do not invent fixtures, prices, or player names not present "
        "in the context.\n"
        "If asked about something outside this context, say you do not have that data in the saved blend.\n"
        "Keep answers concise and actionable. Use UK English for football terms."
    )
    inp = _truncate_json_blob("blend_input", blend_input, _MAX_BLEND_INPUT_JSON)
    res = _truncate_json_blob("blend_result", blend_result, _MAX_BLEND_RESULT_JSON)
    return (
        f"{instructions}\n\n"
        f"--- blend_input (JSON) ---\n{inp}\n\n"
        f"--- blend_result (JSON) ---\n{res}"
    )


def get_copilot_llm_adapter() -> CopilotGeminiAdapter | CopilotOpenRouterAdapter:
    """Same provider selection as ``CopilotJobService.from_dependencies``."""
    provider = os.environ.get("LLM_PROVIDER", "gemini").strip().lower()
    if provider == "gemini":
        return CopilotGeminiAdapter()
    if provider == "openrouter":
        return CopilotOpenRouterAdapter()
    raise ValueError("Unsupported LLM_PROVIDER. Expected 'gemini' or 'openrouter'.")


def run_copilot_chat(
    *,
    blend_input: Mapping[str, Any],
    blend_result: Mapping[str, Any],
    prior_messages: list[dict[str, str]],
    user_message: str,
) -> str:
    if len(user_message) > _MAX_USER_MESSAGE:
        raise CopilotChatError("Message too long.")
    if len(prior_messages) > _MAX_PRIOR_MESSAGES:
        raise CopilotChatError("Too many prior messages.")

    adapter = get_copilot_llm_adapter()
    system_prompt = build_copilot_chat_system_prompt(blend_input, blend_result)
    try:
        return adapter.generate_copilot_chat_reply(
            system_prompt=system_prompt,
            prior_messages=prior_messages,
            user_message=user_message.strip(),
        )
    except CopilotChatError:
        raise
    except Exception as exc:
        logger.warning("copilot chat failed: %s", exc, exc_info=True)
        raise CopilotChatError("The assistant could not complete your request. Try again shortly.") from exc
