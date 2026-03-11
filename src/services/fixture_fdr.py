"""
Fixture Difficulty Rating (FDR) service.

Combines ClubElo strength ratings with position-weighted injury losses to
produce separate attack, defence and overall FDR scores on a 1–5 scale.

Formula reference
-----------------
Base Elo difficulty (for Team A facing Team B):
    dr_A  = Elo_A − Elo_B + H × home_A   (H = 55, home_A = +1 if home else −1)
    E_A   = 1 / (1 + 10^(−dr_A / 400))
    base_A = 0.5 − E_A                    (negative = easier, positive = harder)

Attack FDR raw score:
    rawAttack_A = base_A + 0.55 × AttLoss(A) − 0.85 × DefLoss(B)

Defence FDR raw score:
    rawDefence_A = base_A + 0.70 × DefLoss(A) − 0.85 × AttLoss(B)

Overall FDR raw score:
    rawOverall_A = 0.55 × rawAttack_A + 0.45 × rawDefence_A

Map to 1–5:
    FDR(x) = clamp(1, 5,  1 + 4 × sigmoid(3.0 × x))
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, cast

from src.services.club_elo import get_team_elo
from src.services.injury_impact import summarize_injury_impact
from src.services.injury_news import team_id_for_name
from src.services.squad_change import summarize_squad_changes

HOME_ELO_BONUS: float = 55.0
_SIGMOID_SLOPE: float = 3.0

# Attack / defence scaling weights
_ATT_OWN_LOSS_W: float = 0.55   # own attack loss increases attack difficulty
_ATT_OPP_DEF_LOSS_W: float = 0.85  # opp defensive loss decreases attack difficulty
_DEF_OWN_LOSS_W: float = 0.70   # own defensive loss increases defence difficulty
_DEF_OPP_ATT_LOSS_W: float = 0.85  # opp attacking loss decreases defence difficulty
_TEAM_SQUAD_ATTACK_W: float = 0.35
_TEAM_SQUAD_DEFENCE_W: float = 0.45
_OPP_SQUAD_ATTACK_W: float = 0.45
_OPP_SQUAD_DEFENCE_W: float = 0.45


# ---------------------------------------------------------------------------
# Core maths
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _summary_float(summary: Dict[str, object], key: str) -> float:
    return float(cast(float, summary.get(key, 0.0)))


def _summary_list(summary: Dict[str, object], key: str) -> list[Dict]:
    return list(cast(list[Dict], summary.get(key, [])))


def elo_base(
    elo_team: float,
    elo_opp: float,
    is_home: bool,
    home_bonus: float = HOME_ELO_BONUS,
) -> float:
    """Return the base difficulty raw score for *elo_team* against *elo_opp*.

    Positive  → fixture is harder than average for this team.
    Negative  → fixture is easier than average.
    """
    direction = 1.0 if is_home else -1.0
    dr = elo_team - elo_opp + home_bonus * direction
    expected = 1.0 / (1.0 + 10.0 ** (-dr / 400.0))
    return 0.5 - expected


def _merge_absence_debug(injury_summary: Dict[str, object], squad_summary: Dict[str, object], side: str) -> tuple[list[Dict], list[Dict]]:
    counted = [
        {**item, "layer": "injury", "side": side}
        for item in _summary_list(injury_summary, "counted_absences")
    ] + [
        {**item, "layer": "squad_change", "side": side}
        for item in _summary_list(squad_summary, "counted_absences")
    ]
    ignored = [
        {**item, "layer": "injury", "side": side}
        for item in _summary_list(injury_summary, "ignored_absences")
    ] + [
        {**item, "layer": "squad_change", "side": side}
        for item in _summary_list(squad_summary, "ignored_absences")
    ]
    return counted, ignored


def _sorted_key_absences(absences: list[Dict]) -> list[Dict]:
    return sorted(
        absences,
        key=lambda item: (
            max(float(item.get("attacking_impact", 0.0)), float(item.get("defensive_impact", 0.0))),
            item.get("player_name") or "",
        ),
        reverse=True,
    )


def raw_fdrs(
    team_elo: float,
    opp_elo: float,
    is_home: bool,
    team_players: Optional[List[Dict]] = None,
    opp_players: Optional[List[Dict]] = None,
    snapshot_date: Optional[str] = None,
) -> tuple[float, float, float]:
    """Compute (rawAttack, rawDefence, rawOverall) difficulty scores.

    Parameters
    ----------
    team_elo, opp_elo:
        ClubElo ratings.
    is_home:
        Whether *team* is the home side.
    team_players, opp_players:
        Lists of player dicts (see ``injury_impact.team_injury_losses``).
        Pass ``None`` or ``[]`` when injury data is unavailable.
    """
    base = elo_base(team_elo, opp_elo, is_home)

    team_injury = summarize_injury_impact(team_players or [])
    opp_injury = summarize_injury_impact(opp_players or [])
    team_squad = summarize_squad_changes(team_players or [], snapshot_date=snapshot_date)
    opp_squad = summarize_squad_changes(opp_players or [], snapshot_date=snapshot_date)

    team_att_loss = _summary_float(team_injury, "att_loss")
    team_def_loss = _summary_float(team_injury, "def_loss")
    opp_att_loss = _summary_float(opp_injury, "att_loss")
    opp_def_loss = _summary_float(opp_injury, "def_loss")
    team_squad_att = _summary_float(team_squad, "attack_loss")
    team_squad_def = _summary_float(team_squad, "defence_loss")
    opp_squad_att = _summary_float(opp_squad, "attack_loss")
    opp_squad_def = _summary_float(opp_squad, "defence_loss")

    raw_attack = (
        base
        + _ATT_OWN_LOSS_W * team_att_loss
        - _ATT_OPP_DEF_LOSS_W * opp_def_loss
        + _TEAM_SQUAD_ATTACK_W * team_squad_att
        - _OPP_SQUAD_DEFENCE_W * opp_squad_def
    )
    raw_defence = (
        base
        + _DEF_OWN_LOSS_W * team_def_loss
        - _DEF_OPP_ATT_LOSS_W * opp_att_loss
        + _TEAM_SQUAD_DEFENCE_W * team_squad_def
        - _OPP_SQUAD_ATTACK_W * opp_squad_att
    )
    raw_overall = 0.55 * raw_attack + 0.45 * raw_defence

    return raw_attack, raw_defence, raw_overall


def to_fdr(raw: float) -> float:
    """Map a raw difficulty score to the 1–5 FDR scale (continuous)."""
    return clamp(1.0 + 4.0 * sigmoid(_SIGMOID_SLOPE * raw), 1.0, 5.0)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def compute_fixture_fdr(
    team_name: str,
    opponent_name: str,
    is_home: bool,
    team_players: Optional[List[Dict]] = None,
    opp_players: Optional[List[Dict]] = None,
    snapshot_date: Optional[str] = None,
    elo_team: Optional[float] = None,
    elo_opp: Optional[float] = None,
) -> Dict:
    """Compute full FDR breakdown for a single fixture.

    Fetches ClubElo ratings automatically unless *elo_team* / *elo_opp*
    are supplied explicitly (useful for tests or when ratings are already
    cached).

    Returns a dict matching the documented response shape:
    {
        "team": ...,
        "opponent": ...,
        "is_home": ...,
        "elo_team": ...,
        "elo_opponent": ...,
        "base_raw": ...,
        "team_attack_loss": ...,
        "team_defence_loss": ...,
        "opp_attack_loss": ...,
        "opp_defence_loss": ...,
        "raw_attack": ...,
        "raw_defence": ...,
        "raw_overall": ...,
        "attack_fdr": ...,
        "defence_fdr": ...,
        "overall_fdr": ...,
        "attack_fdr_int": ...,
        "defence_fdr_int": ...,
        "overall_fdr_int": ...
    }
    """
    date_str = snapshot_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    if elo_team is None:
        elo_team = get_team_elo(team_name, date_str)
    if elo_opp is None:
        elo_opp = get_team_elo(opponent_name, date_str)

    base = elo_base(elo_team, elo_opp, is_home)

    team_injury = summarize_injury_impact(team_players or [])
    opp_injury = summarize_injury_impact(opp_players or [])
    team_squad = summarize_squad_changes(team_players or [], snapshot_date=date_str)
    opp_squad = summarize_squad_changes(opp_players or [], snapshot_date=date_str)

    team_att_loss = _summary_float(team_injury, "att_loss")
    team_def_loss = _summary_float(team_injury, "def_loss")
    opp_att_loss = _summary_float(opp_injury, "att_loss")
    opp_def_loss = _summary_float(opp_injury, "def_loss")
    team_squad_att = _summary_float(team_squad, "attack_loss")
    team_squad_def = _summary_float(team_squad, "defence_loss")
    opp_squad_att = _summary_float(opp_squad, "attack_loss")
    opp_squad_def = _summary_float(opp_squad, "defence_loss")

    team_counted_absences, team_ignored_absences = _merge_absence_debug(team_injury, team_squad, side="team")
    opp_counted_absences, opp_ignored_absences = _merge_absence_debug(opp_injury, opp_squad, side="opponent")
    key_absences_counted = _sorted_key_absences(team_counted_absences + opp_counted_absences)
    key_absences_ignored = _sorted_key_absences(team_ignored_absences + opp_ignored_absences)

    raw_attack = (
        base
        + _ATT_OWN_LOSS_W * team_att_loss
        - _ATT_OPP_DEF_LOSS_W * opp_def_loss
        + _TEAM_SQUAD_ATTACK_W * team_squad_att
        - _OPP_SQUAD_DEFENCE_W * opp_squad_def
    )
    raw_defence = (
        base
        + _DEF_OWN_LOSS_W * team_def_loss
        - _DEF_OPP_ATT_LOSS_W * opp_att_loss
        + _TEAM_SQUAD_DEFENCE_W * team_squad_def
        - _OPP_SQUAD_ATTACK_W * opp_squad_att
    )
    raw_overall = 0.55 * raw_attack + 0.45 * raw_defence

    attack_fdr = to_fdr(raw_attack)
    defence_fdr = to_fdr(raw_defence)
    overall_fdr = to_fdr(raw_overall)

    return {
        "team": team_name,
        "team_id": team_id_for_name(team_name),
        "opponent": opponent_name,
        "opponent_id": team_id_for_name(opponent_name),
        "is_home": is_home,
        "elo_team": round(elo_team, 2),
        "elo_opponent": round(elo_opp, 2),
        "base_raw": round(base, 4),
        "team_attack_loss": round(team_att_loss, 4),
        "team_defence_loss": round(team_def_loss, 4),
        "team_squad_change_attack_loss": round(team_squad_att, 4),
        "team_squad_change_defence_loss": round(team_squad_def, 4),
        "opp_attack_loss": round(opp_att_loss, 4),
        "opp_defence_loss": round(opp_def_loss, 4),
        "opp_squad_change_attack_loss": round(opp_squad_att, 4),
        "opp_squad_change_defence_loss": round(opp_squad_def, 4),
        "team_counted_absences": team_counted_absences,
        "team_ignored_absences": team_ignored_absences,
        "opp_counted_absences": opp_counted_absences,
        "opp_ignored_absences": opp_ignored_absences,
        "key_absences_counted": key_absences_counted,
        "key_absences_ignored": key_absences_ignored,
        "raw_attack": round(raw_attack, 4),
        "raw_defence": round(raw_defence, 4),
        "raw_overall": round(raw_overall, 4),
        "attack_fdr": round(attack_fdr, 2),
        "defence_fdr": round(defence_fdr, 2),
        "overall_fdr": round(overall_fdr, 2),
        "attack_fdr_int": round(attack_fdr),
        "defence_fdr_int": round(defence_fdr),
        "overall_fdr_int": round(overall_fdr),
    }
