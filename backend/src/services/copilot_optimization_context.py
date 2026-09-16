"""Load AIrsenal transfer optimization snapshot for Copilot LLM context.

Mirrors the payload written by ``adapters.airsenal_adapter.export_optimization`` to
``gw_<GW>_optimization_<FPL_TEAM_ID>.json``, sourced from ``transfer_suggestion``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional


def _norm_chip_played(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


def _has_table(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None


def _first_table(con: sqlite3.Connection, candidates: tuple[str, ...]) -> Optional[str]:
    for t in candidates:
        if _has_table(con, t):
            return t
    return None


def _q(con: sqlite3.Connection, sql: str, params: dict[str, Any] | tuple[Any, ...] | None = None) -> list[dict[str, Any]]:
    cur = con.execute(sql, params or ())
    rows = cur.fetchall()
    out: list[dict[str, Any]] = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out.append({k: r[k] for k in r.keys()})
        else:
            out.append(dict(r))
    return out


def load_optimization_for_team(
    db_path: str | Path,
    *,
    fpl_team_id: int,
    target_gameweek: Optional[int] = None,
    season: Optional[str] = None,
) -> dict[str, Any]:
    """Return optimization context for the latest AIrsenal run for this FPL entry team.

    If ``target_gameweek`` is set, ``gameweek_plan`` contains only that gameweek's
    transfers/chip (same structure as one element of ``gameweeks`` in the JSON export).
    """
    path = str(db_path)
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
    except sqlite3.Error:
        return {"available": False, "reason": "database_unreadable", "fpl_team_id": fpl_team_id}

    try:
        t = _first_table(con, ("transfer_suggestion", "transfersuggestion"))
        if not t:
            return {"available": False, "reason": "no_transfer_suggestion_table", "fpl_team_id": fpl_team_id}

        cols = {r[1] for r in con.execute(f"PRAGMA table_info({t});").fetchall()}
        if "fpl_team_id" not in cols:
            return {"available": False, "reason": "transfer_suggestion_missing_fpl_team_id", "fpl_team_id": fpl_team_id}

        params: dict[str, Any] = {"tid": fpl_team_id}
        season_filter = ""
        if season and "season" in cols:
            season_filter = " AND ts.season = :season "
            params["season"] = season

        row = _q(
            con,
            f"""
            SELECT MAX(ts.timestamp) AS ts
            FROM {t} ts
            WHERE ts.fpl_team_id = :tid
            {season_filter}
            """,
            params,
        )
        ts = row[0]["ts"] if row and row[0].get("ts") else None
        if not ts:
            return {
                "available": False,
                "reason": "no_optimization_rows_for_team",
                "fpl_team_id": fpl_team_id,
            }

        params2: dict[str, Any] = {"tid": fpl_team_id, "ts": ts}
        if season and "season" in cols:
            params2["season"] = season
        season_where = " AND ts.season = :season " if (season and "season" in cols) else ""

        sql = f"""
            SELECT
              ts.gameweek,
              ts.in_or_out,
              ts.points_gain,
              ts.timestamp,
              ts.season,
              ts.fpl_team_id,
              ts.chip_played,
              pl.player_id,
              COALESCE(pl.display_name, pl.name) AS name
            FROM {t} ts
            JOIN player pl ON pl.player_id = ts.player_id
            WHERE ts.fpl_team_id = :tid
              AND ts.timestamp = :ts
            {season_where}
            ORDER BY ts.gameweek, ts.in_or_out DESC, ts.points_gain DESC, pl.player_id;
        """
        rows = _q(con, sql, params2)

        if not rows:
            return {"available": False, "reason": "optimization_query_empty", "fpl_team_id": fpl_team_id}

        by_gw: dict[int, dict[str, Any]] = {}
        total_gain: Optional[float] = None
        season_out: Optional[str] = None

        for r in rows:
            g = int(r["gameweek"])
            entry = by_gw.setdefault(
                g,
                {
                    "gameweek": g,
                    "players_out": [],
                    "players_in": [],
                    "chip_played": _norm_chip_played(r.get("chip_played")),
                    "transfer_pairs": [],
                },
            )
            if r.get("chip_played") is not None:
                entry["chip_played"] = _norm_chip_played(r.get("chip_played"))
            if r.get("points_gain") is not None:
                total_gain = float(r["points_gain"])
            if r.get("season"):
                season_out = str(r["season"])

            ioo = r.get("in_or_out")
            if ioo is None:
                continue
            pinfo = {"player_id": int(r["player_id"]), "name": r["name"]}
            if int(ioo) > 0:
                entry["players_in"].append(pinfo)
            else:
                entry["players_out"].append(pinfo)

        gws_sorted = sorted(by_gw.keys())
        for g in gws_sorted:
            e = by_gw[g]
            outs = e["players_out"]
            ins = e["players_in"]
            pairs: list[dict[str, Any]] = []
            n = max(len(outs), len(ins))
            for i in range(n):
                pairs.append(
                    {
                        "player_out": outs[i] if i < len(outs) else None,
                        "player_in": ins[i] if i < len(ins) else None,
                    }
                )
            e["transfer_pairs"] = pairs

        all_gameweeks = [by_gw[g] for g in gws_sorted]

        gameweek_plan: Optional[dict[str, Any]] = None
        if target_gameweek is not None and target_gameweek in by_gw:
            gameweek_plan = by_gw[target_gameweek]

        return {
            "available": True,
            "fpl_team_id": fpl_team_id,
            "target_gameweek": target_gameweek,
            "timestamp": str(ts),
            "season": season_out,
            "source_table": t,
            "total_points_gain_vs_baseline": total_gain,
            "notes": (
                "total_points_gain_vs_baseline is (strategy total − baseline total) from the "
                "optimizer, repeated on transfer_suggestion rows. Per-transfer xPts are not stored here."
            ),
            "gameweeks": all_gameweeks,
            "gameweek_plan": gameweek_plan,
        }
    except sqlite3.OperationalError as exc:
        return {"available": False, "reason": f"sqlite_error:{exc}", "fpl_team_id": fpl_team_id}
    finally:
        con.close()
