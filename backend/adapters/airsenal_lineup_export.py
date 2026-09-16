"""
Rebuild AIrsenal's recommended starting XI + captain / vice from SQLite:

1) Read latest ``transfer_suggestion`` run for the team (same rows as optimization JSON).
2) Build a strategy dict (players in/out per GW).
3) Load ``Squad`` via ``get_starting_squad`` (DB transactions), apply first-GW transfers.
4) Run ``get_expected_points`` (same as ``print_team_for_next_gw`` in AIrsenal).

Requires the AIrsenal package on ``PYTHONPATH`` (e.g. ``AIrsenal/.venv``).
``AIRSENAL_DB_FILE`` must match ``--db`` and be set **before** importing ``airsenal``.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from adapters.airsenal_adapter import (  # noqa: E402
    first_table,
    log,
    q,
    table_columns,
    write_json,
)


def _serialize_squad(squad: Any, tag: str, next_gw: int) -> Dict[str, Any]:
    players_out: List[Dict[str, Any]] = []
    for p in squad.players:
        pred = None
        if tag in p.predicted_points and next_gw in p.predicted_points[tag]:
            pred = p.predicted_points[tag][next_gw]
        players_out.append(
            {
                "player_id": p.player_id,
                "name": str(p.display_name or p.name),
                "team": p.team,
                "position": p.position,
                "is_starting": p.is_starting,
                "is_captain": p.is_captain,
                "is_vice_captain": p.is_vice_captain,
                "predicted_points": pred,
                "sub_position": p.sub_position,
            }
        )

    starters = [x for x in players_out if x["is_starting"]]
    subs = [x for x in players_out if not x["is_starting"]]
    subs.sort(key=lambda x: (x["sub_position"] is None, x["sub_position"] or 0))

    by_pos: Dict[str, List[Dict[str, Any]]] = {"GK": [], "DEF": [], "MID": [], "FWD": []}
    for x in players_out:
        pos = x["position"]
        if pos in by_pos:
            by_pos[pos].append(x)

    return {
        "players": players_out,
        "starters": starters,
        "subs": subs,
        "by_position": by_pos,
    }


def export_recommended_lineup(
    db_path: str,
    con: sqlite3.Connection,
    outdir: Path,
    gw: int,
    fpl_team_id: int,
    season: Optional[str],
    repo_root: Path,
) -> None:
    """
    Write ``gw_<gw>_lineup_<fpl_team_id>.json`` using AIrsenal ``Squad`` logic.

    Caller must set ``os.environ["AIRSENAL_DB_FILE"]`` to the same path as ``db_path``
    before this function imports ``airsenal`` (so SQLAlchemy uses the same SQLite file).
    """
    log("exporting recommended lineup (Squad.optimize_lineup)…")

    t = first_table(con, ["transfer_suggestion", "transfersuggestion"])
    if not t:
        log("WARN: no transfer_suggestion table; skipping lineup export")
        return

    cols = table_columns(con, t)
    if "fpl_team_id" not in cols:
        log("WARN: transfer_suggestion has no fpl_team_id; skipping lineup export")
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
        log(f"WARN: no transfer_suggestion rows for team {fpl_team_id}; skipping lineup export")
        return

    params2: Dict[str, Any] = {"tid": fpl_team_id, "ts": ts}
    if season and "season" in cols:
        params2["season"] = season

    season_where = " AND ts.season = :season " if (season and "season" in cols) else ""

    has_id = "id" in cols
    select_cols = (
        "ts.id, ts.gameweek, ts.in_or_out, ts.player_id"
        if has_id
        else "ts.gameweek, ts.in_or_out, ts.player_id"
    )
    order_clause = "ORDER BY ts.id ASC" if has_id else "ORDER BY ts.gameweek, ts.in_or_out DESC, ts.player_id"

    sql = f"""
        SELECT {select_cols}
        FROM {t} ts
        WHERE ts.fpl_team_id = :tid
          AND ts.timestamp = :ts
        {season_where}
        {order_clause};
    """
    rows = q(con, sql, params2)
    if not rows:
        log("WARN: lineup export query returned no rows; skipping")
        return

    by_gw: Dict[int, Dict[str, List[int]]] = {}
    for r in rows:
        g = int(r["gameweek"])
        ioo = int(r["in_or_out"])
        pid = int(r["player_id"])
        if g not in by_gw:
            by_gw[g] = {"outs": [], "ins": []}
        if ioo < 0:
            by_gw[g]["outs"].append(pid)
        else:
            by_gw[g]["ins"].append(pid)

    gws = sorted(by_gw.keys())
    for g in gws:
        outs, ins = by_gw[g]["outs"], by_gw[g]["ins"]
        if len(outs) != len(ins):
            log(
                f"WARN: GW{g}: players_out ({len(outs)}) vs players_in ({len(ins)}) mismatch; "
                "skipping lineup export"
            )
            return

    strat = {
        "points_per_gw": {str(g): 0.0 for g in gws},
        "players_out": {str(g): by_gw[g]["outs"] for g in gws},
        "players_in": {str(g): by_gw[g]["ins"] for g in gws},
    }

    next_gw = gws[0]

    os.environ["AIRSENAL_DB_FILE"] = str(Path(db_path).resolve())
    os.environ.setdefault("AIRSENAL_HOME", str(repo_root / ".airsenal_home"))

    try:
        from airsenal.framework.optimization_utils import get_starting_squad
        from airsenal.framework.season import CURRENT_SEASON
        from airsenal.framework.utils import get_latest_prediction_tag
    except ImportError as exc:
        log(f"WARN: cannot import airsenal (use AIrsenal venv): {exc}")
        return

    season_str = season or CURRENT_SEASON
    use_api = False

    try:
        squad = get_starting_squad(
            next_gw=next_gw,
            season=season_str,
            fpl_team_id=fpl_team_id,
            use_api=use_api,
        )
    except Exception as exc:
        log(f"WARN: get_starting_squad failed ({exc}); skipping lineup export")
        return

    for pid in strat["players_out"][str(next_gw)]:
        squad.remove_player(pid, gameweek=next_gw, use_api=use_api)
    for pid in strat["players_in"][str(next_gw)]:
        squad.add_player(pid, gameweek=next_gw)

    try:
        tag = get_latest_prediction_tag(season=season_str)
    except Exception as exc:
        log(f"WARN: get_latest_prediction_tag failed ({exc}); skipping lineup export")
        return

    try:
        total = squad.get_expected_points(next_gw, tag)
    except Exception as exc:
        log(f"WARN: get_expected_points failed ({exc}); skipping lineup export")
        return

    body = {
        "gameweek": next_gw,
        "export_anchor_gameweek": gw,
        "fpl_team_id": fpl_team_id,
        "prediction_tag": tag,
        "optimization_timestamp": str(ts),
        "expected_points_total": float(total),
        "method": "Squad.optimize_lineup (same rules as AIrsenal console)",
        "squad": _serialize_squad(squad, tag, next_gw),
    }

    write_json(outdir, f"gw_{gw}_lineup_{fpl_team_id}.json", body)
    log(f"✓ gw_{gw}_lineup_{fpl_team_id}.json  (GW{next_gw} expected ~{total:.1f} pts)")

