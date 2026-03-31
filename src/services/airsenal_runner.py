"""
Run AIrsenal CLI commands the same way ``scripts/airsenal.sh`` does:
AIrsenal venv, ``AIRSENAL_HOME`` pointing at ``.airsenal_home``, optional
secrets from files in that directory.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


def find_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        if (candidate / "data" / "api").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root containing data/api")


def _read_optional_file(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").strip()


def build_airsenal_run_env(repo_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = repo_root / ".airsenal_home"
    env["AIRSENAL_HOME"] = str(home)
    for fname in ("AIRSENAL_DB_FILE", "FPL_TEAM_ID", "FPL_LOGIN", "FPL_PASSWORD"):
        val = _read_optional_file(home / fname)
        if val:
            env[fname] = val
    return env


def resolve_venv_bin(repo_root: Path) -> Path:
    venv_bin = repo_root / "AIrsenal" / ".venv" / "bin"
    if not venv_bin.is_dir():
        raise FileNotFoundError(
            f"AIrsenal venv not found at {venv_bin}. Create AIrsenal/.venv or install deps."
        )
    return venv_bin


def resolve_db_path(repo_root: Path, env: dict[str, str]) -> str:
    return env.get("AIRSENAL_DB_FILE") or str(repo_root / "data" / "airsenal" / "data.db")


def resolve_team_id(env: dict[str, str], override: Optional[int]) -> Optional[str]:
    if override is not None:
        return str(override)
    tid = env.get("FPL_TEAM_ID")
    return tid if tid else None


def _truncate(s: str, max_len: int = 24_000) -> str:
    if len(s) <= max_len:
        return s
    return s[:max_len] + "\n... [truncated]"


class AirsenalRunError(Exception):
    """Command failed or validation failed before running."""

    def __init__(
        self,
        message: str,
        *,
        completed: Optional[subprocess.CompletedProcess[str]] = None,
    ) -> None:
        super().__init__(message)
        self.completed = completed


class AirsenalRunRequest(BaseModel):
    action: Literal["update_db", "predict", "optimize", "export", "pipeline"]
    weeks_ahead: int = Field(3, ge=1, le=38)
    gameweek: str = Field(
        "auto",
        description="'auto' or an integer GW string for export (matches --gw).",
    )
    fpl_team_id: Optional[int] = Field(
        None,
        description="Override FPL team id; otherwise uses ~/.airsenal_home/FPL_TEAM_ID via env.",
    )


class AirsenalRunResponse(BaseModel):
    ok: bool
    action: str
    steps: list[dict[str, Any]]


def run_airsenal_action(
    req: AirsenalRunRequest,
    *,
    timeout_sec: Optional[int] = None,
) -> AirsenalRunResponse:
    repo_root = find_repo_root()
    env = build_airsenal_run_env(repo_root)
    venv_bin = resolve_venv_bin(repo_root)
    timeout = timeout_sec if timeout_sec is not None else int(os.environ.get("AIRSENAL_RUN_TIMEOUT_SEC", "3600"))
    db = resolve_db_path(repo_root, env)
    out_dir = repo_root / "data" / "api"
    team_id = resolve_team_id(env, req.fpl_team_id)

    steps: list[dict[str, Any]] = []

    def run_cmd(step: str, cmd: list[str]) -> None:
        try:
            cp = subprocess.run(
                cmd,
                cwd=str(repo_root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired as exc:
            raise AirsenalRunError(
                f"Command timed out after {timeout}s: {' '.join(cmd)}"
            ) from exc
        if cp.returncode != 0:
            raise AirsenalRunError(
                f"Command failed ({cp.returncode}): {' '.join(cmd)}",
                completed=cp,
            )
        steps.append(
            {
                "step": step,
                "returncode": cp.returncode,
                "stdout": _truncate(cp.stdout or ""),
                "stderr": _truncate(cp.stderr or ""),
            }
        )

    def run_update() -> None:
        run_cmd("update_db", [str(venv_bin / "airsenal_update_db")])

    def run_predict() -> None:
        run_cmd(
            "predict",
            [
                str(venv_bin / "airsenal_run_prediction"),
                "--weeks_ahead",
                str(req.weeks_ahead),
            ],
        )

    def run_optimize() -> None:
        if not team_id:
            raise AirsenalRunError(
                "fpl_team_id is required for optimize (or set .airsenal_home/FPL_TEAM_ID)"
            )
        run_cmd(
            "optimize",
            [
                str(venv_bin / "airsenal_run_optimization"),
                "--weeks_ahead",
                str(req.weeks_ahead),
                "--fpl_team_id",
                team_id,
            ],
        )

    def run_export() -> None:
        cmd = [
            str(venv_bin / "python"),
            str(repo_root / "adapters" / "airsenal_adapter.py"),
            "--db",
            db,
            "--out",
            str(out_dir),
            "--gw",
            req.gameweek,
        ]
        if team_id:
            cmd.extend(["--team-id", team_id])
        run_cmd("export", cmd)

    action = req.action
    if action == "update_db":
        run_update()
    elif action == "predict":
        run_predict()
    elif action == "optimize":
        run_optimize()
    elif action == "export":
        run_export()
    elif action == "pipeline":
        run_update()
        run_predict()
        run_optimize()
        run_export()

    return AirsenalRunResponse(ok=True, action=action, steps=steps)
