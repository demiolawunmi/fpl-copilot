"""
Injury impact service.

Computes team-level attacking and defensive loss scores from a list of
player availability records.  Designed to be used on top of ClubElo Elo
ratings so that position-specific absences affect attack and defence
difficulty separately.

Player dict schema (all fields optional unless noted):
    position      (str, required) – one of the keys in ATT_ROLE
    minutes_last6 (float) – total minutes played in the last 6 GWs
    goals90       (float) – goals per 90 min
    assists90     (float) – assists per 90 min
    prob_available (float | None) – probability of being available [0, 1]
    status        (str | None)  – "available" | "doubtful" | "questionable"
                                   | "major_doubt" | "out" | "suspended"
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, cast

# ---------------------------------------------------------------------------
# Position role weights
# ---------------------------------------------------------------------------

ATT_ROLE: Dict[str, float] = {
    "GK":   0.05,
    "CB":   0.10,
    "FB":   0.35,
    "WB":   0.35,
    "DM":   0.30,
    "CM":   0.45,
    "AM":   0.80,
    "W":    0.80,
    "ST":   0.95,
}

DEF_ROLE: Dict[str, float] = {
    "GK":   1.00,
    "CB":   0.95,
    "FB":   0.70,
    "WB":   0.70,
    "DM":   0.60,
    "CM":   0.45,
    "AM":   0.20,
    "W":    0.20,
    "ST":   0.10,
}

# Fallback multipliers for unrecognised positions
_DEFAULT_ATT_ROLE = 0.45
_DEFAULT_DEF_ROLE = 0.45

_COUNTED_ABSENCE_TYPES = {"injury", "suspension", "doubtful", "questionable", "major_doubt"}
_RECENT_MINUTES_BASELINE = 540.0
_SEASON_MINUTES_BASELINE = 2160.0
_PRE_ABSENCE_MINUTES_BASELINE = 900.0
_ATT_PROD_CAP = 1.0
_KNOWN_FIRST_TEAM_FALLBACK = 0.3
_KNOWN_PLAYER_FALLBACK = 0.12

# Status → availability-loss mapping (1 − p_available)
_STATUS_MAP: Dict[str, float] = {
    "available":   0.00,
    "doubtful":    0.25,
    "questionable": 0.50,
    "major_doubt": 0.75,
    "out":         1.00,
    "suspended":   1.00,
}


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def availability_loss(
    prob_available: Optional[float] = None,
    status: Optional[str] = None,
) -> float:
    """Return the availability loss for a player (0 = fully fit, 1 = out).

    If *prob_available* is given it takes precedence over *status*.
    """
    if prob_available is not None:
        prob = max(0.0, min(1.0, float(prob_available)))
        return 1.0 - prob
    if status:
        return _STATUS_MAP.get(status.lower(), 0.0)
    return 0.0


def _player_minutes_share(player: Dict) -> float:
    minutes = max(0.0, float(player.get("minutes_last6") or 0.0))
    return min(1.0, minutes / _RECENT_MINUTES_BASELINE)


def _normalized_share(value: object, baseline: float) -> float:
    if value is None:
        return 0.0
    try:
        numeric = max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0
    if baseline <= 0.0:
        return 0.0
    return min(1.0, numeric / baseline)


def _player_attacking_output(player: Dict) -> float:
    goals90 = max(0.0, float(player.get("goals90") or 0.0))
    assists90 = max(0.0, float(player.get("assists90") or 0.0))
    return min(_ATT_PROD_CAP, 1.2 * goals90 + 0.8 * assists90)


def importance_basis(player: Dict, use_pre_departure: bool = False) -> tuple[float, str]:
    recent_minutes = 0.0 if use_pre_departure else _normalized_share(player.get("minutes_last6"), _RECENT_MINUTES_BASELINE)
    season_minutes = _normalized_share(
        player.get("minutes_season_before_departure") if use_pre_departure and player.get("minutes_season_before_departure") is not None else player.get("minutes_season"),
        _SEASON_MINUTES_BASELINE,
    )
    pre_absence_minutes = _normalized_share(
        player.get("minutes_last10_before_departure") if use_pre_departure and player.get("minutes_last10_before_departure") is not None else player.get("minutes_last10_before_absence"),
        _PRE_ABSENCE_MINUTES_BASELINE,
    )
    starter_probability = _normalized_share(
        player.get("pre_departure_starter_flag") if use_pre_departure and player.get("pre_departure_starter_flag") is not None else player.get("starter_probability"),
        1.0,
    )

    candidates = {
        "minutes_last6": recent_minutes,
        "minutes_season_before_departure" if use_pre_departure else "minutes_season": season_minutes,
        "minutes_last10_before_departure" if use_pre_departure else "minutes_last10_before_absence": pre_absence_minutes,
        "starter_probability": starter_probability,
    }
    best_source, best_value = max(candidates.items(), key=lambda item: item[1])
    if best_value > 0.0:
        return best_value, best_source

    if player.get("player_name") and player.get("team") and player.get("position"):
        return _KNOWN_FIRST_TEAM_FALLBACK, "known_first_team_fallback"
    if player.get("player_name") or player.get("position"):
        return _KNOWN_PLAYER_FALLBACK, "known_player_fallback"
    return 0.0, "no_importance_data"


def player_importance(player: Dict, use_pre_departure: bool = False) -> Tuple[float, float]:
    pos = (player.get("position") or "CM").upper()
    att_role = ATT_ROLE.get(pos, _DEFAULT_ATT_ROLE)
    def_role = DEF_ROLE.get(pos, _DEFAULT_DEF_ROLE)

    basis, _ = importance_basis(player, use_pre_departure=use_pre_departure)
    attacking_output = _player_attacking_output(player)

    attacking_importance = (0.65 * basis + 0.35 * attacking_output) * att_role
    defensive_importance = basis * def_role
    return attacking_importance, defensive_importance


def summarize_injury_impact(players: List[Dict]) -> Dict[str, object]:
    if not players:
        return {
            "att_loss": 0.0,
            "def_loss": 0.0,
            "counted_absences": [],
            "ignored_absences": [],
        }

    counted_absences = []
    ignored_absences = []
    att_sum = 0.0
    def_sum = 0.0

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

        if absence_type not in _COUNTED_ABSENCE_TYPES:
            ignored_absences.append(
                {
                    **debug_row,
                    "reason_counted_or_ignored": "not_counted_in_injury_layer",
                    "attacking_impact": 0.0,
                    "defensive_impact": 0.0,
                }
            )
            continue

        availability = availability_loss(player.get("prob_available"), player.get("status"))
        if availability <= 0.0:
            ignored_absences.append(
                {
                    **debug_row,
                    "reason_counted_or_ignored": "fully_available",
                    "attacking_impact": 0.0,
                    "defensive_impact": 0.0,
                }
            )
            continue

        _, basis_source = importance_basis(player)
        att_importance, def_importance = player_importance(player)
        att_contribution = availability * att_importance
        def_contribution = availability * def_importance
        att_sum += att_contribution
        def_sum += def_contribution
        counted_absences.append(
            {
                **debug_row,
                "availability_loss": round(availability, 4),
                "reason_counted_or_ignored": f"counted_in_injury_layer:{basis_source}",
                "attacking_impact": round(att_contribution, 4),
                "defensive_impact": round(def_contribution, 4),
            }
        )

    att_loss = 1.0 - math.exp(-att_sum)
    def_loss = 1.0 - math.exp(-def_sum)
    return {
        "att_loss": att_loss,
        "def_loss": def_loss,
        "counted_absences": counted_absences,
        "ignored_absences": ignored_absences,
    }


def team_injury_losses(players: List[Dict]) -> Tuple[float, float]:
    """Compute (att_loss, def_loss) for a team given a list of player dicts.

    Both values lie in [0, 1) and saturate naturally: a single key
    absentee matters a lot, many bench players do not explode the model.

    Returns
    -------
    tuple
        ``(att_loss, def_loss)`` where 0 means no impact and values
        approaching 1 mean severe losses.
    """
    summary = summarize_injury_impact(players)
    att_loss = cast(float, summary.get("att_loss", 0.0))
    def_loss = cast(float, summary.get("def_loss", 0.0))
    return float(att_loss), float(def_loss)
