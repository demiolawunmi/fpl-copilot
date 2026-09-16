from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from src.services.copilot_job_repository import redact_sensitive_fields

logger = logging.getLogger(__name__)


def _find_data_api_dir() -> Path:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        api_dir = candidate / "data" / "api"
        if api_dir.is_dir():
            return api_dir
    raise RuntimeError("Could not locate data/api directory for blend snapshots")


def _snapshot_filename(gameweek: Any, fpl_team_id: Any) -> str:
    """``gw_<gw>_copilot_blend_<fpl_team_id>.json`` or ``gw_<gw>_copilot_blend.json`` when no team id."""
    try:
        gw = int(gameweek)
    except (TypeError, ValueError):
        gw = 0
    if gw < 0:
        gw = 0
    try:
        tid = int(fpl_team_id)
    except (TypeError, ValueError):
        tid = 0
    if tid > 0:
        return f"gw_{gw}_copilot_blend_{tid}.json"
    return f"gw_{gw}_copilot_blend.json"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(dict(payload), indent=2, ensure_ascii=False)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(data + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_blend_snapshot(
    *,
    job_id: str,
    schema_version: str,
    correlation_id: str,
    input_payload: Mapping[str, Any],
    result_payload: Mapping[str, Any],
) -> Path | None:
    """Persist last successful hybrid blend to data/api for /api/files/... consumption."""
    try:
        data_api = _find_data_api_dir()
    except RuntimeError as exc:
        logger.warning("copilot blend snapshot skipped: %s", exc)
        return None

    fpl_team_id = input_payload.get("fpl_team_id")
    gameweek = input_payload.get("gameweek")
    filename = _snapshot_filename(gameweek, fpl_team_id)
    path = (data_api / filename).resolve()
    resolved_root = data_api.resolve()
    if path.parent != resolved_root:
        logger.warning("copilot blend snapshot path outside data/api: %s", path)
        return None

    snapshot: dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "job_id": job_id,
        "correlation_id": correlation_id,
        "schema_version": schema_version,
        "input": redact_sensitive_fields(dict(input_payload)),
        "result": dict(result_payload),
    }

    try:
        _atomic_write_json(path, snapshot)
    except OSError as exc:
        logger.warning("copilot blend snapshot write failed: %s", exc)
        return None

    return path
