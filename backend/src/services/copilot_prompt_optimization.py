"""Format AIrsenal optimization snapshot for LLM prompts."""

from __future__ import annotations

from typing import Any


def format_airsenal_optimization_text(model_context: dict[str, Any]) -> str:
    opt = model_context.get("airsenal_optimization", {})
    if not opt.get("available"):
        return f"(no AIrsenal optimization snapshot: {opt.get('reason', 'unknown')})\n"

    gw_plan = opt.get("gameweek_plan")
    if not isinstance(gw_plan, dict):
        gw_plan = {}

    pairs_txt = ""
    for pair in gw_plan.get("transfer_pairs", [])[:8]:
        po = pair.get("player_out") or {}
        pi = pair.get("player_in") or {}
        pairs_txt += (
            f"    • OUT {po.get('name', '?')} (id {po.get('player_id', '?')}) → "
            f"IN {pi.get('name', '?')} (id {pi.get('player_id', '?')})\n"
        )

    chip = gw_plan.get("chip_played")
    gw_num = gw_plan.get("gameweek")

    if not gw_plan:
        plan_lines = "  (no plan row for target gameweek — full horizon in CONTEXT_JSON)\n"
    else:
        plan_lines = (
            f"- Target GW plan (gameweek {gw_num}):\n"
            f"  chip: {chip}\n"
            f"{pairs_txt if pairs_txt else '  (no paired transfers for this GW in snapshot)\n'}"
        )

    return (
        f"- FPL team id: {opt.get('fpl_team_id')}\n"
        f"- Optimizer timestamp: {opt.get('timestamp')}\n"
        f"- Total points gain vs baseline (full horizon): {opt.get('total_points_gain_vs_baseline')}\n"
        f"{plan_lines}"
    )
