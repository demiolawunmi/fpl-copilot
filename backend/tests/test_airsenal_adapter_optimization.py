"""Minimal test for export_optimization JSON (SQLite)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapters.airsenal_adapter import connect, export_optimization  # noqa: E402


def test_export_optimization_writes_gw_team_file(tmp_path: Path) -> None:
    db_path = tmp_path / "mini.db"
    con = sqlite3.connect(db_path)
    con.execute(
        "CREATE TABLE player (player_id INTEGER PRIMARY KEY, name TEXT, display_name TEXT)"
    )
    con.execute(
        """
        CREATE TABLE transfer_suggestion (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER,
            in_or_out INTEGER,
            gameweek INTEGER,
            points_gain REAL,
            timestamp TEXT,
            season TEXT,
            fpl_team_id INTEGER,
            chip_played TEXT
        )
        """
    )
    con.execute("INSERT INTO player VALUES (1, 'Salah', 'M.Salah')")
    con.execute("INSERT INTO player VALUES (2, 'Saka', 'Saka')")
    ts = "2026-03-30T12:00:00"
    for pid, ioo, g in [(1, -1, 32), (2, 1, 32)]:
        con.execute(
            """
            INSERT INTO transfer_suggestion
            (player_id, in_or_out, gameweek, points_gain, timestamp, season, fpl_team_id, chip_played)
            VALUES (?, ?, ?, 18.1, ?, '2526', 8994418, 'None')
            """,
            (pid, ioo, g, ts),
        )
    con.commit()
    con.close()

    c = connect(str(db_path))
    export_optimization(c, tmp_path, 32, 8994418, "2526")
    c.close()

    out = tmp_path / "gw_32_optimization_8994418.json"
    assert out.is_file()
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["fpl_team_id"] == 8994418
    assert data["export_anchor_gameweek"] == 32
    assert data["total_points_gain_vs_baseline"] == pytest.approx(18.1)
    assert len(data["gameweeks"]) == 1
    gw0 = data["gameweeks"][0]
    assert gw0["gameweek"] == 32
    assert gw0["chip_played"] is None
    assert len(gw0["transfer_pairs"]) == 1
    assert gw0["transfer_pairs"][0]["player_out"]["name"] == "M.Salah"
    assert gw0["transfer_pairs"][0]["player_in"]["name"] == "Saka"
