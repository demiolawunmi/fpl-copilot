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
from typing import Dict, List, Optional, Tuple

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
    if not players:
        return 0.0, 0.0

    # Compute attacking production per player
    att_prod_values: List[float] = []
    for p in players:
        goals90 = float(p.get("goals90") or 0.0)
        assists90 = float(p.get("assists90") or 0.0)
        att_prod_values.append(1.2 * goals90 + 0.8 * assists90)

    total_att_prod = sum(att_prod_values) or 1e-6

    max_minutes = max((float(p.get("minutes_last6") or 0.0) for p in players), default=1.0)
    if max_minutes == 0.0:
        max_minutes = 1.0

    att_sum = 0.0
    def_sum = 0.0

    for idx, p in enumerate(players):
        pos = (p.get("position") or "CM").upper()
        att_role = ATT_ROLE.get(pos, _DEFAULT_ATT_ROLE)
        def_role = DEF_ROLE.get(pos, _DEFAULT_DEF_ROLE)

        min_share = min(1.0, float(p.get("minutes_last6") or 0.0) / max_minutes)
        att_share = att_prod_values[idx] / total_att_prod

        avail = availability_loss(
            p.get("prob_available"),
            p.get("status"),
        )

        att_imp = avail * (0.65 * min_share + 0.35 * att_share) * att_role
        def_imp = avail * min_share * def_role

        att_sum += att_imp
        def_sum += def_imp

    att_loss = 1.0 - math.exp(-att_sum)
    def_loss = 1.0 - math.exp(-def_sum)
    return att_loss, def_loss
