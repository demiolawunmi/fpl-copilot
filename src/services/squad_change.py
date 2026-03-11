"""
Temporary squad-change adjustment layer.

This layer captures short-term disruption from recent loan and transfer
outgoings before ClubElo has had time to fully absorb the underlying team
strength change. Departures use the player's pre-departure importance when
available and decay away over time.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional

from src.services.injury_impact import player_importance

_COUNTED_DEPARTURE_TYPES = {"loan_out", "transfer_out"}
_DECAY_MATCH_CONSTANT = 3.0
_MIN_EFFECT_THRESHOLD = 0.01


def _parse_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _coerce_snapshot_datetime(snapshot_date: Optional[str]) -> datetime:
    if snapshot_date:
        parsed = _parse_datetime(snapshot_date)
        if parsed is not None:
            return parsed.astimezone(timezone.utc)
        try:
            return datetime.fromisoformat(f"{snapshot_date}T00:00:00+00:00")
        except ValueError:
            pass
    return datetime.now(timezone.utc)


def departure_decay(player: Dict, snapshot_date: Optional[str] = None) -> float:
    matches_since = player.get("matches_since_departure")
    if matches_since is not None:
        try:
            return math.exp(-max(0.0, float(matches_since)) / _DECAY_MATCH_CONSTANT)
        except (TypeError, ValueError):
            pass

    reference = _coerce_snapshot_datetime(snapshot_date)
    departure_time = _parse_datetime(player.get("departure_date") or player.get("last_updated"))
    if departure_time is None:
        return 1.0

    days_since = max(0.0, (reference - departure_time.astimezone(timezone.utc)).total_seconds() / 86400.0)
    approx_matches_since = days_since / 7.0
    return math.exp(-approx_matches_since / _DECAY_MATCH_CONSTANT)


def summarize_squad_changes(players: List[Dict], snapshot_date: Optional[str] = None) -> Dict[str, object]:
    if not players:
        return {
            "attack_loss": 0.0,
            "defence_loss": 0.0,
            "counted_absences": [],
            "ignored_absences": [],
        }

    att_sum = 0.0
    def_sum = 0.0
    counted_absences = []
    ignored_absences = []

    for player in players:
        absence_type = (player.get("absence_type") or "other").lower()
        debug_row = {
            "player_id": player.get("player_id"),
            "player_name": player.get("player_name"),
            "position": player.get("position"),
            "absence_type": absence_type,
            "status": player.get("status"),
            "source_news": player.get("source_news"),
        }

        if absence_type not in _COUNTED_DEPARTURE_TYPES:
            ignored_absences.append({
                **debug_row,
                "reason_counted_or_ignored": "not_counted_in_squad_change_layer",
                "attacking_impact": 0.0,
                "defensive_impact": 0.0,
            })
            continue

        minutes_season = float(player.get("minutes_season", 0) or 0)
        minutes_pre = float(player.get("minutes_last10_before_absence", 0) or 0)
        starter_prob = float(player.get("starter_probability", 0) or 0)

        if minutes_season < 600 and minutes_pre < 180 and starter_prob < 0.4:
            ignored_absences.append({
                **debug_row,
                "reason_counted_or_ignored": "insufficient_pre_departure_importance",
                "attacking_impact": 0.0,
                "defensive_impact": 0.0,
            })
            continue

        decay = departure_decay(player, snapshot_date)
        att_importance, def_importance = player_importance(player, use_pre_departure=True)

        att_contribution = decay * att_importance
        def_contribution = decay * def_importance

        if max(att_contribution, def_contribution) < _MIN_EFFECT_THRESHOLD:
            ignored_absences.append({
                **debug_row,
                "reason_counted_or_ignored": "below_effect_threshold",
                "attacking_impact": 0.0,
                "defensive_impact": 0.0,
            })
            continue

        att_sum += att_contribution
        def_sum += def_contribution

        counted_absences.append({
            **debug_row,
            "reason_counted_or_ignored": f"counted_in_squad_change_layer:matches_decay={round(decay, 4)}",
            "attacking_impact": round(att_contribution, 4),
            "defensive_impact": round(def_contribution, 4),
        })

    return {
        "attack_loss": 1.0 - math.exp(-att_sum),
        "defence_loss": 1.0 - math.exp(-def_sum),
        "counted_absences": counted_absences,
        "ignored_absences": ignored_absences,
    }



