#!/usr/bin/env python3
"""
AIrsenal -> JSON adapter

Exports small, UI-friendly JSON files so your app/LLM never touches SQL.
Designed to tolerate minor AIrsenal schema differences (table/column names).

Creates (when source tables exist):
  <out>/
    gw_<GW>_predictions.json
    gw_<GW>_fixtures_by_player.json
    gw_<GW>_transfers.json
    gw_<GW>_optimization_<FPL_TEAM_ID>.json  (requires --team-id; from transfer_suggestion)
    gw_<GW>_lineup_<FPL_TEAM_ID>.json  (requires --team-id; Squad.optimize_lineup via AIrsenal)
    injury_news.json
    form_last4.json
    bandwagons.json
    teams.json
    fixtures.json

Example:
  python adapters/airsenal_adapter.py \
    --db data/airsenal/data.db \
    --out data/api \
    --gw auto \
    --team-id 123456
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import sqlite3
import sys
from pathlib import Path

# Allow ``from adapters.…`` when run as ``python adapters/airsenal_adapter.py`` (cwd = repo root).
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# ------------------------------ util ---------------------------------

def log(msg: str) -> None:
    print(f"[adapter] {msg}")

def connect(db_path: str) -> sqlite3.Connection:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    return con

def has_table(con: sqlite3.Connection, name: str) -> bool:
    cur = con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?;", (name,))
    return cur.fetchone() is not None

def table_columns(con: sqlite3.Connection, name: str) -> List[str]:
    try:
        rows = con.execute(f"PRAGMA table_info({name});").fetchall()
        return [r["name"] if isinstance(r, sqlite3.Row) else r[1] for r in rows]
    except sqlite3.Error:
        return []

def first_table(con: sqlite3.Connection, candidates: Sequence[str]) -> Optional[str]:
    for t in candidates:
        if has_table(con, t):
            return t
    return None

def q(con: sqlite3.Connection, sql: str, params: Union[Tuple[Any, ...], Dict[str, Any], None] = None) -> List[Dict[str, Any]]:
    cur = con.execute(sql, params or ())
    rows = cur.fetchall()
    out: List[Dict[str, Any]] = []
    for r in rows:
        if isinstance(r, sqlite3.Row):
            out.append({k: r[k] for k in r.keys()})
        else:
            out.append(dict(r))
    return out

def write_json(outdir: Path, filename: str, payload: Any) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / filename
    with p.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def fixture_date_expr(con: sqlite3.Connection, alias: str = "f") -> str:
    """
    Different AIrsenal versions use 'date' or 'kickoff_time' in fixture.
    Return a SQL expression that always works with the provided alias.
    """
    cols = table_columns(con, "fixture")
    has_date = "date" in cols
    has_kick = "kickoff_time" in cols
    if has_date and has_kick:
        return f"COALESCE({alias}.date, {alias}.kickoff_time)"
    if has_date:
        return f"{alias}.date"
    if has_kick:
        return f"{alias}.kickoff_time"
    return "NULL"

def infer_current_season(con: sqlite3.Connection) -> Optional[str]:
    if not has_table(con, "fixture"):
        return None
    row = q(con, "SELECT MAX(season) AS season FROM fixture;")
    s = row[0]["season"] if row and row[0].get("season") else None
    return str(s) if s else None

def infer_season_for_gw(con: sqlite3.Connection, gw: int) -> Optional[str]:
    """
    GW numbers repeat each season, so we try to pick the most recent season that contains that GW.
    """
    if not has_table(con, "fixture"):
        return None
    row = q(con, "SELECT MAX(season) AS season FROM fixture WHERE gameweek = :gw;", {"gw": gw})
    s = row[0]["season"] if row and row[0].get("season") else None
    return str(s) if s else infer_current_season(con)

def detect_latest_playerpred_tag(con: sqlite3.Connection, gw: int, season: Optional[str]) -> Optional[str]:
    """
    player_prediction can hold multiple runs; tag identifies a run. Prefer latest tag for the GW (+ season if possible).
    """
    if not (has_table(con, "player_prediction") and has_table(con, "fixture")):
        return None
    cols = table_columns(con, "player_prediction")
    if "tag" not in cols:
        return None

    if season:
        row = q(con, """
            SELECT MAX(pp.tag) AS tag
            FROM player_prediction pp
            JOIN fixture f ON f.fixture_id = pp.fixture_id
            WHERE f.gameweek = :gw AND f.season = :season;
        """, {"gw": gw, "season": season})
    else:
        row = q(con, """
            SELECT MAX(pp.tag) AS tag
            FROM player_prediction pp
            JOIN fixture f ON f.fixture_id = pp.fixture_id
            WHERE f.gameweek = :gw;
        """, {"gw": gw})

    tag = row[0]["tag"] if row and row[0].get("tag") else None
    return str(tag) if tag else None

def detect_latest_transfers_timestamp(con: sqlite3.Connection, gw: int, fpl_team_id: Optional[int]) -> Optional[str]:
    """
    transfer_suggestion can hold multiple runs; prefer latest timestamp for the GW (+ team if present).
    """
    t = first_table(con, ["transfer_suggestion", "transfersuggestion"])
    if not t:
        return None
    cols = table_columns(con, t)
    if "timestamp" not in cols:
        return None

    if "fpl_team_id" in cols and fpl_team_id is not None:
        row = q(con, f"SELECT MAX(timestamp) AS ts FROM {t} WHERE gameweek=:gw AND fpl_team_id=:tid;", {"gw": gw, "tid": fpl_team_id})
    else:
        row = q(con, f"SELECT MAX(timestamp) AS ts FROM {t} WHERE gameweek=:gw;", {"gw": gw})

    ts = row[0]["ts"] if row and row[0].get("ts") else None
    return str(ts) if ts else None

# --------------------------- GW detection -----------------------------

def detect_calendar_next_gw(con: sqlite3.Connection) -> Optional[int]:
    """
    Use the current UTC date/time and scheduled fixtures to infer the next GW.
    Mirrors AIrsenal's fixture-based logic:
      - find the earliest future fixture GW in the current season
      - if that GW has already started, advance to the following GW
      - if no future dated fixtures exist, fall back to max scheduled GW + 1
    """
    if not has_table(con, "fixture"):
        return None

    season = infer_current_season(con)
    date_expr = fixture_date_expr(con, alias="f")
    if date_expr == "NULL":
        return None

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    params: Dict[str, Any] = {"now": now_utc}
    season_filter = ""
    if season:
        params["season"] = season
        season_filter = "AND f.season = :season"

    row = q(con, f"""
        SELECT MIN(f.gameweek) AS gw
        FROM fixture f
        WHERE f.gameweek IS NOT NULL
          AND {date_expr} IS NOT NULL
          {season_filter}
          AND datetime({date_expr}) > datetime(:now);
    """, params)

    if row and row[0].get("gw") is not None:
        next_gw = int(row[0]["gw"])
        started = q(con, f"""
            SELECT 1 AS started
            FROM fixture f
            WHERE f.gameweek = :gw
              AND {date_expr} IS NOT NULL
              {season_filter}
              AND datetime({date_expr}) < datetime(:now)
            LIMIT 1;
        """, {**params, "gw": next_gw})
        if started:
            next_gw += 1
        return next_gw

    row = q(con, f"""
        SELECT MAX(f.gameweek) AS gw
        FROM fixture f
        WHERE f.gameweek IS NOT NULL
          {season_filter};
    """, params)
    if row and row[0].get("gw") is not None:
        return int(row[0]["gw"]) + 1

    return None

def detect_next_gw(con: sqlite3.Connection) -> Optional[int]:
    """
    Try to infer the next GW to export predictions for.
    Strategy:
      - use current UTC date/time plus fixture dates to derive the next calendar GW
      - if prediction tables exist, choose the smallest predicted GW at or after that floor
      - otherwise fall back to the calendar-derived GW
    """
    season = infer_current_season(con)
    calendar_gw = detect_calendar_next_gw(con)

    if has_table(con, "predictedscore"):
        ps_cols = table_columns(con, "predictedscore")
        params: Dict[str, Any] = {}
        clauses = ["gameweek IS NOT NULL"]
        if calendar_gw is not None:
            clauses.append("gameweek >= :gw_floor")
            params["gw_floor"] = calendar_gw
        if season and "season" in ps_cols:
            clauses.append("season = :season")
            params["season"] = season

        row = q(con, f"SELECT MIN(gameweek) AS gw FROM predictedscore WHERE {' AND '.join(clauses)};", params)
        if row and row[0].get("gw") is not None:
            return int(row[0]["gw"])

    if has_table(con, "player_prediction") and has_table(con, "fixture"):
        params = {}
        clauses = ["f.gameweek IS NOT NULL"]
        if season:
            clauses.append("f.season = :season")
            params["season"] = season
        if calendar_gw is not None:
            clauses.append("f.gameweek >= :gw_floor")
            params["gw_floor"] = calendar_gw

        row = q(con, f"""
             SELECT MIN(f.gameweek) AS gw
             FROM player_prediction pp
             JOIN fixture f ON f.fixture_id = pp.fixture_id
            WHERE {' AND '.join(clauses)};
        """, params)
        if row and row[0].get("gw") is not None:
            return int(row[0]["gw"])

    return calendar_gw

# --------------------------- exports ----------------------------------

def export_predictions(con: sqlite3.Connection, outdir: Path, gw: int, season: Optional[str]) -> None:
    log("exporting predictions…")

    # Older schema
    if has_table(con, "predictedscore"):
        ps_cols = table_columns(con, "predictedscore")
        needed = {"predicted_points", "player_id", "gameweek"}
        if not needed.issubset(set(ps_cols)):
            log("WARN: predictedscore missing expected columns; skipping predictions export")
            return

        sql = """
            WITH base AS (
              SELECT
                ps.player_id,
                ps.gameweek,
                SUM(ps.predicted_points) AS xp
              FROM predictedscore ps
              WHERE ps.gameweek = :gw
              GROUP BY ps.player_id, ps.gameweek
            )
            SELECT
              b.player_id,
              COALESCE(pl.display_name, pl.name) AS name,
              pa.team AS team,
              tm.full_name AS team_full,
              pa.position AS position,
              pa.price AS price,
              b.gameweek,
              pa.chance_of_playing_next_round,
              pa.news,
              b.xp
            FROM base b
            LEFT JOIN player pl ON pl.player_id = b.player_id
            LEFT JOIN player_attributes pa
              ON pa.player_id = b.player_id
             AND pa.gameweek = b.gameweek
            LEFT JOIN team tm
              ON tm.name = pa.team AND tm.season = pa.season
            ORDER BY b.xp DESC;
        """
        rows = q(con, sql, {"gw": gw})
        write_json(outdir, f"gw_{gw}_predictions.json", rows)
        log(f"✓ gw_{gw}_predictions.json  ({len(rows)} rows)")
        return

    # Newer schema: player_prediction per fixture; aggregate to GW per player.
    if not (has_table(con, "player_prediction") and has_table(con, "fixture")):
        log("WARN: no predictions table found; skipping predictions export")
        return

    pp_cols = table_columns(con, "player_prediction")
    if "predicted_points" not in pp_cols:
        log("WARN: player_prediction missing predicted_points; skipping predictions export")
        return

    tag = detect_latest_playerpred_tag(con, gw, season)
    tag_filter = "AND pp.tag = :tag" if tag else ""
    params: Dict[str, Any] = {"gw": gw}
    if season:
        params["season"] = season
        season_filter = "AND f.season = :season"
    else:
        season_filter = ""
    if tag:
        params["tag"] = tag

    sql = f"""
        WITH base AS (
          SELECT
            pp.player_id,
            f.gameweek AS gameweek,
            SUM(pp.predicted_points) AS xp,
            MAX(f.season) AS season
          FROM player_prediction pp
          JOIN fixture f ON f.fixture_id = pp.fixture_id
          WHERE f.gameweek = :gw
          {season_filter}
          {tag_filter}
          GROUP BY pp.player_id, f.gameweek
        )
        SELECT
          b.player_id,
          COALESCE(pl.display_name, pl.name) AS name,
          pa.team AS team,
          tm.full_name AS team_full,
          pa.position AS position,
          pa.price AS price,
          b.gameweek,
          pa.chance_of_playing_next_round,
          pa.news,
          b.xp
        FROM base b
        LEFT JOIN player pl ON pl.player_id = b.player_id
        LEFT JOIN player_attributes pa
          ON pa.player_id = b.player_id
         AND pa.season = b.season
         AND pa.gameweek = b.gameweek
        LEFT JOIN team tm
          ON tm.name = pa.team AND tm.season = pa.season
        ORDER BY b.xp DESC;
    """
    rows = q(con, sql, params)
    write_json(outdir, f"gw_{gw}_predictions.json", rows)
    log(f"✓ gw_{gw}_predictions.json  ({len(rows)} rows){' (tag '+tag+')' if tag else ''}")

def export_fixtures_by_player(con: sqlite3.Connection, outdir: Path, gw: int, season: Optional[str]) -> None:
    log("exporting fixtures_by_player…")

    has_predictedscore = has_table(con, "predictedscore")
    has_player_pred = has_table(con, "player_prediction") and has_table(con, "fixture")
    if not (has_predictedscore or has_player_pred):
        log("WARN: no predictions table found; skipping fixtures_by_player")
        return

    date_expr = fixture_date_expr(con, alias="f")
    params: Dict[str, Any] = {"gw": gw}
    season_filter = ""
    if season:
        params["season"] = season
        season_filter = "AND f.season = :season"

    if has_player_pred:
        tag = detect_latest_playerpred_tag(con, gw, season)
        tag_filter = "AND pp.tag = :tag" if tag else ""
        if tag:
            params["tag"] = tag

        sql = f"""
            WITH attrs AS (
              SELECT pa.player_id, pa.season, pa.gameweek, pa.team
              FROM player_attributes pa
              WHERE pa.gameweek = :gw
            )
            SELECT
              pl.player_id,
              COALESCE(pl.display_name, pl.name) AS name,
              f.gameweek AS gw,
              f.fixture_id,
              CASE
                WHEN (a.team = f.home_team) THEN f.away_team ELSE f.home_team
              END AS opp,
              CASE
                WHEN (a.team = f.home_team) THEN 'H' ELSE 'A'
              END AS home_away,
              {date_expr} AS date
            FROM player_prediction pp
            JOIN fixture f ON f.fixture_id = pp.fixture_id
            JOIN player  pl ON pl.player_id = pp.player_id
            LEFT JOIN attrs a ON a.player_id = pl.player_id AND a.season = f.season AND a.gameweek = f.gameweek
            WHERE f.gameweek = :gw
              {season_filter}
              {tag_filter}
            GROUP BY pl.player_id, name, f.gameweek, f.fixture_id, opp, home_away, date
            ORDER BY name;
        """
        rows = q(con, sql, params)
    else:
        ps_cols = table_columns(con, "predictedscore")
        if "fixture_id" in ps_cols and has_table(con, "fixture"):
            sql = f"""
                WITH attrs AS (
                  SELECT pa.player_id, pa.season, pa.gameweek, pa.team
                  FROM player_attributes pa
                  WHERE pa.gameweek = :gw
                )
                SELECT
                  pl.player_id,
                  COALESCE(pl.display_name, pl.name) AS name,
                  ps.gameweek AS gw,
                  f.fixture_id,
                  CASE WHEN (a.team = f.home_team) THEN f.away_team ELSE f.home_team END AS opp,
                  CASE WHEN (a.team = f.home_team) THEN 'H' ELSE 'A' END AS home_away,
                  {date_expr} AS date
                FROM predictedscore ps
                JOIN player pl ON pl.player_id = ps.player_id
                LEFT JOIN fixture f ON f.fixture_id = ps.fixture_id
                LEFT JOIN attrs a ON a.player_id = pl.player_id AND a.season = f.season AND a.gameweek = ps.gameweek
                WHERE ps.gameweek = :gw
                GROUP BY pl.player_id, name, ps.gameweek, f.fixture_id, opp, home_away, date
                ORDER BY name;
            """
            rows = q(con, sql, {"gw": gw})
        else:
            sql = """
                SELECT DISTINCT
                  ps.player_id,
                  COALESCE(pl.display_name, pl.name) AS name,
                  ps.gameweek AS gw,
                  NULL AS fixture_id,
                  NULL AS opp,
                  NULL AS home_away,
                  NULL AS date
                FROM predictedscore ps
                JOIN player pl ON pl.player_id = ps.player_id
                WHERE ps.gameweek = :gw
                ORDER BY name;
            """
            rows = q(con, sql, {"gw": gw})

    write_json(outdir, f"gw_{gw}_fixtures_by_player.json", rows)
    log(f"✓ gw_{gw}_fixtures_by_player.json  ({len(rows)} rows)")

def export_transfers(con: sqlite3.Connection, outdir: Path, gw: int, fpl_team_id: Optional[int]) -> None:
    log("exporting transfers…")

    t = first_table(con, ["transfer_suggestion", "transfersuggestion"])
    if not t:
        log("WARN: no transfer suggestion table; skipping transfers export")
        return

    cols = table_columns(con, t)
    has_team_col = "fpl_team_id" in cols
    has_ts_col = "timestamp" in cols

    ts = detect_latest_transfers_timestamp(con, gw, fpl_team_id) if has_ts_col else None

    clauses = ["ts.gameweek = :gw"]
    params: Dict[str, Any] = {"gw": gw}

    if has_team_col and fpl_team_id is not None:
        clauses.append("ts.fpl_team_id = :tid")
        params["tid"] = fpl_team_id

    if has_ts_col and ts:
        clauses.append("ts.timestamp = :ts")
        params["ts"] = ts

    where = " AND ".join(clauses)

    sql = f"""
        SELECT
          ts.gameweek,
          ts.in_or_out,
          ts.points_gain,
          ts.timestamp AS timestamp,
          pl.player_id,
          COALESCE(pl.display_name, pl.name) AS name
        FROM {t} ts
        JOIN player pl ON pl.player_id = ts.player_id
        WHERE {where}
        ORDER BY ts.gameweek, ts.in_or_out DESC, ts.points_gain DESC;
    """
    rows = q(con, sql, params)

    by_gw: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        g = int(r["gameweek"])
        by_gw.setdefault(g, {"gw": g, "ins": [], "outs": [], "points_gain": None, "timestamp": r.get("timestamp")})
        if r.get("in_or_out") is None:
            continue
        if r["in_or_out"] > 0:
            by_gw[g]["ins"].append({"player_id": r["player_id"], "name": r["name"]})
            if r.get("points_gain") is not None:
                by_gw[g]["points_gain"] = r["points_gain"]
        else:
            by_gw[g]["outs"].append({"player_id": r["player_id"], "name": r["name"]})

    out_rows = [by_gw[gw]] if gw in by_gw else []
    write_json(outdir, f"gw_{gw}_transfers.json", out_rows)
    log(f"✓ gw_{gw}_transfers.json  ({len(out_rows)} rows){' (ts '+ts+')' if ts else ''}")


def _norm_chip_played(val: Any) -> Optional[str]:
    if val is None:
        return None
    s = str(val).strip()
    if s == "" or s.lower() in ("none", "null"):
        return None
    return s


def export_optimization(
    con: sqlite3.Connection,
    outdir: Path,
    gw: int,
    fpl_team_id: Optional[int],
    season: Optional[str],
) -> None:
    """
    Export the latest optimization run from ``transfer_suggestion`` as structured JSON.

    AIrsenal stores per-player rows (in/out per gameweek), ``chip_played``, and a
    single ``points_gain`` (strategy total minus baseline) repeated on each row.
    Baseline / strategy absolute totals and per-GW predicted scores are *not* stored
    in this table (those only exist in the optimizer's in-memory strategy / deleted
    JSON under ``AIRSENAL_HOME/airsopt`` during a run).
    """
    log("exporting optimization summary…")

    if fpl_team_id is None:
        log("WARN: no --team-id; skipping gw_*_optimization_*.json (need FPL team id)")
        return

    t = first_table(con, ["transfer_suggestion", "transfersuggestion"])
    if not t:
        log("WARN: no transfer_suggestion table; skipping optimization export")
        return

    cols = table_columns(con, t)
    if "fpl_team_id" not in cols:
        log("WARN: transfer_suggestion has no fpl_team_id; skipping optimization export")
        return

    params: Dict[str, Any] = {"tid": fpl_team_id}
    season_filter = ""
    if season and "season" in cols:
        season_filter = " AND ts.season = :season "
        params["season"] = season

    row = q(
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
        log(f"WARN: no transfer_suggestion rows for team {fpl_team_id}; skipping optimization export")
        return

    params2: Dict[str, Any] = {"tid": fpl_team_id, "ts": ts}
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
    rows = q(con, sql, params2)

    if not rows:
        log("WARN: optimization query returned no rows; skipping")
        return

    by_gw: Dict[int, Dict[str, Any]] = {}
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
        pairs: List[Dict[str, Any]] = []
        n = max(len(outs), len(ins))
        for i in range(n):
            pairs.append(
                {
                    "player_out": outs[i] if i < len(outs) else None,
                    "player_in": ins[i] if i < len(ins) else None,
                }
            )
        e["transfer_pairs"] = pairs

    payload: Dict[str, Any] = {
        "fpl_team_id": fpl_team_id,
        "export_anchor_gameweek": gw,
        "timestamp": str(ts),
        "season": season_out,
        "source_table": t,
        "total_points_gain_vs_baseline": total_gain,
        "notes": (
            "total_points_gain_vs_baseline is (strategy total − baseline total) from the "
            "optimizer, stored once per row in transfer_suggestion. Absolute baseline and "
            "strategy scores, per-gameweek points hit, and per-GW predicted scores are not "
            "persisted in SQLite; they only appear in the optimizer console output."
        ),
        "gameweeks": [by_gw[g] for g in gws_sorted],
    }

    write_json(outdir, f"gw_{gw}_optimization_{fpl_team_id}.json", payload)
    log(f"✓ gw_{gw}_optimization_{fpl_team_id}.json  ({len(gws_sorted)} GWs, ts {ts})")


def export_injury_news(con: sqlite3.Connection, outdir: Path, gw: int, season: Optional[str]) -> None:
    log("exporting injury_news…")

    if not has_table(con, "player_attributes"):
        log("WARN: no player_attributes; skipping injury news")
        write_json(outdir, "injury_news.json", [])
        return

    params: Dict[str, Any] = {"gw": gw}
    season_filter = ""
    if season:
        params["season"] = season
        season_filter = "AND f.season = :season"

    if has_table(con, "predictedscore"):
        sql = """
            SELECT DISTINCT
              pl.player_id,
              COALESCE(pl.display_name, pl.name) AS name,
              pa.team,
              pa.chance_of_playing_next_round AS chance_next_round,
              pa.news
            FROM predictedscore ps
            JOIN player pl ON pl.player_id = ps.player_id
            LEFT JOIN player_attributes pa
              ON pa.player_id = pl.player_id
             AND pa.gameweek = ps.gameweek
            WHERE ps.gameweek = :gw
              AND (
                (pa.chance_of_playing_next_round IS NOT NULL AND pa.chance_of_playing_next_round < 100)
                OR (pa.news IS NOT NULL AND TRIM(pa.news) <> '')
              )
            ORDER BY name;
        """
        rows = q(con, sql, {"gw": gw})
    elif has_table(con, "player_prediction") and has_table(con, "fixture"):
        tag = detect_latest_playerpred_tag(con, gw, season)
        tag_filter = "AND pp.tag = :tag" if tag else ""
        if tag:
            params["tag"] = tag

        sql = f"""
            SELECT DISTINCT
              pl.player_id,
              COALESCE(pl.display_name, pl.name) AS name,
              pa.team,
              pa.chance_of_playing_next_round AS chance_next_round,
              pa.news
            FROM player_prediction pp
            JOIN fixture f ON f.fixture_id = pp.fixture_id
            JOIN player pl ON pl.player_id = pp.player_id
            LEFT JOIN player_attributes pa
              ON pa.player_id = pl.player_id
             AND pa.season = f.season
             AND pa.gameweek = f.gameweek
            WHERE f.gameweek = :gw
              {season_filter}
              {tag_filter}
              AND (
                (pa.chance_of_playing_next_round IS NOT NULL AND pa.chance_of_playing_next_round < 100)
                OR (pa.news IS NOT NULL AND TRIM(pa.news) <> '')
              )
            ORDER BY name;
        """
        rows = q(con, sql, params)
    else:
        rows = []

    write_json(outdir, "injury_news.json", rows)
    log(f"✓ injury_news.json  ({len(rows)} rows)")

def export_form_last4(con: sqlite3.Connection, outdir: Path, gw: int, season: Optional[str]) -> None:
    log("exporting form_last4…")

    ps = first_table(con, ["player_score", "playerscore"])
    if not ps or not has_table(con, "fixture"):
        log("WARN: no score table; skipping form_last4")
        write_json(outdir, "form_last4.json", [])
        return

    ps_cols = table_columns(con, ps)
    pts_col = "points" if "points" in ps_cols else ("event_points" if "event_points" in ps_cols else None)
    min_col = "minutes" if "minutes" in ps_cols else None
    xgi_col = "expected_goal_involvements" if "expected_goal_involvements" in ps_cols else None
    xgc_col = "expected_goals_conceded" if "expected_goals_conceded" in ps_cols else None

    if not pts_col or not min_col:
        log("WARN: score table missing points/minutes; skipping form_last4")
        write_json(outdir, "form_last4.json", [])
        return

    hi = int(gw) - 1
    if hi < 1:
        write_json(outdir, "form_last4.json", [])
        log("✓ form_last4.json  (0 rows, no completed GWs before this GW)")
        return
    lo = max(1, hi - 3)

    sum_bits = [f"SUM(ps.{pts_col}) AS last4_points", f"SUM(ps.{min_col}) AS last4_minutes"]
    if xgi_col:
        sum_bits.append(f"SUM(ps.{xgi_col}) AS last4_xgi")
    if xgc_col:
        sum_bits.append(f"SUM(ps.{xgc_col}) AS last4_xgc")

    select_clause = ", ".join(["pl.player_id", "COALESCE(pl.display_name, pl.name) AS name"] + sum_bits)

    params: Dict[str, Any] = {"lo": lo, "hi": hi}
    season_filter = ""
    if season:
        params["season"] = season
        season_filter = "AND f.season = :season"

    sql = f"""
        SELECT
          {select_clause}
        FROM {ps} ps
        JOIN fixture f ON f.fixture_id = ps.fixture_id
        JOIN player  pl ON pl.player_id = ps.player_id
        WHERE f.gameweek BETWEEN :lo AND :hi
          {season_filter}
        GROUP BY pl.player_id, name
        ORDER BY last4_points DESC;
    """
    rows = q(con, sql, params)
    write_json(outdir, "form_last4.json", rows)
    log(f"✓ form_last4.json  ({len(rows)} rows, window GW {lo}-{hi}{' season '+season if season else ''})")

def export_bandwagons(con: sqlite3.Connection, outdir: Path, gw: int, season: Optional[str]) -> None:
    log("exporting bandwagons…")

    if not has_table(con, "player_attributes"):
        log("WARN: no player_attributes; skipping bandwagons")
        write_json(outdir, "bandwagons.json", [])
        return

    cols = table_columns(con, "player_attributes")
    needed = ["selected", "transfers_in", "transfers_out", "transfers_balance"]
    have = [c for c in needed if c in cols]
    if not have:
        log("WARN: player_attributes missing transfer columns; skipping bandwagons")
        write_json(outdir, "bandwagons.json", [])
        return

    select_bits = ["pl.player_id", "COALESCE(pl.display_name, pl.name) AS name", "pa.team", "pa.position", "pa.price"] + [f"pa.{c}" for c in have]
    order = "pa.transfers_balance DESC" if "transfers_balance" in have else "pa.selected DESC"
    params: Dict[str, Any] = {"gw": gw}
    season_filter = ""
    if season and "season" in cols:
        params["season"] = season
        season_filter = "AND pa.season = :season"

    sql = f"""
        SELECT {", ".join(select_bits)}
        FROM player_attributes pa
        JOIN player pl ON pl.player_id = pa.player_id
        WHERE pa.gameweek = :gw
          {season_filter}
        ORDER BY {order};
    """
    rows = q(con, sql, params)
    write_json(outdir, "bandwagons.json", rows)
    log(f"✓ bandwagons.json  ({len(rows)} rows)")

def export_teams(con: sqlite3.Connection, outdir: Path, season: Optional[str]) -> None:
    log("exporting teams.json…")
    if not has_table(con, "team"):
        log("WARN: no team table; skipping teams")
        write_json(outdir, "teams.json", [])
        return
    params: Dict[str, Any] = {}
    where = ""
    if season and "season" in table_columns(con, "team"):
        where = "WHERE season = :season"
        params["season"] = season
    rows = q(con, f"SELECT team_id, name, COALESCE(full_name, name) AS full_name, season FROM team {where};", params)
    write_json(outdir, "teams.json", rows)
    log(f"✓ teams.json  ({len(rows)} rows){' (season '+season+')' if season else ''}")

def export_fixtures(con: sqlite3.Connection, outdir: Path, season: Optional[str]) -> None:
    log("exporting fixtures.json…")
    if not has_table(con, "fixture"):
        log("WARN: no fixture table; skipping fixtures")
        write_json(outdir, "fixtures.json", [])
        return
    date_expr = fixture_date_expr(con, alias="f")
    params: Dict[str, Any] = {}
    where = ""
    if season:
        where = "WHERE f.season = :season"
        params["season"] = season
    sql = f"""
        SELECT
          f.fixture_id,
          {date_expr} AS date,
          f.gameweek,
          f.home_team,
          f.away_team,
          f.season
        FROM fixture f
        {where}
        ORDER BY date, f.gameweek, f.fixture_id;
    """
    rows = q(con, sql, params)
    write_json(outdir, "fixtures.json", rows)
    log(f"✓ fixtures.json  ({len(rows)} rows){' (season '+season+')' if season else ''}")

# ---------------------------- main ------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Export AIrsenal DB to JSON for UI.")
    ap.add_argument("--db", default="data/airsenal/data.db", help="Path to SQLite DB (AIRSENAL_DB_FILE).")
    ap.add_argument("--out", default="data/api", help="Output directory for JSON exports.")
    ap.add_argument("--gw", default="auto", help="'auto' or an integer GW to export.")
    ap.add_argument("--team-id", type=int, default=None, help="FPL team id (optional; improves transfer pairing).")
    args = ap.parse_args()

    db_path = args.db
    outdir = Path(args.out)

    if not Path(db_path).exists():
        raise SystemExit(f"DB not found at {db_path}")

    repo_root = _REPO_ROOT
    os.environ.setdefault("AIRSENAL_HOME", str(repo_root / ".airsenal_home"))
    os.environ["AIRSENAL_DB_FILE"] = str(Path(db_path).resolve())

    con = connect(db_path)

    # Determine GW
    if args.gw == "auto":
        gw = detect_next_gw(con)
        if gw is None:
            raise SystemExit("Could not detect next GW from DB. Try --gw <N>.")
        log(f"auto-detected GW = {gw}")
    else:
        gw = int(args.gw)

    season_for_gw = infer_season_for_gw(con, gw)
    current_season = infer_current_season(con)

    # For 'global' files, prefer current season; for per-GW files, use the GW's season.
    season_global = current_season
    season_target = season_for_gw

    export_predictions(con, outdir, gw, season_target)
    export_fixtures_by_player(con, outdir, gw, season_target)
    export_transfers(con, outdir, gw, args.team_id)
    export_optimization(con, outdir, gw, args.team_id, season_target)
    if args.team_id:
        try:
            from adapters.airsenal_lineup_export import export_recommended_lineup

            export_recommended_lineup(
                str(Path(db_path).resolve()),
                con,
                outdir,
                gw,
                args.team_id,
                season_target,
                repo_root,
            )
        except Exception as exc:
            log(f"WARN: lineup export failed: {exc}")
    export_injury_news(con, outdir, gw, season_target)
    export_form_last4(con, outdir, gw, season_target)
    export_bandwagons(con, outdir, gw, season_target)
    export_teams(con, outdir, season_global)
    export_fixtures(con, outdir, season_global)

if __name__ == "__main__":
    main()
