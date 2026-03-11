# backend_repo/main.py
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, RootModel, model_validator

app = FastAPI()

# allow your frontend dev server + production domain later
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        # "https://your-frontend-domain.com",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _find_repo_root() -> Path:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        if (candidate / "data" / "api").is_dir():
            return candidate
    raise RuntimeError("Could not locate repo root containing data/api")

REPO_ROOT = _find_repo_root()
DATA_DIR = REPO_ROOT / "data" / "api"
RESOLVED_DATA_DIR = DATA_DIR.resolve()


def _serve_api_file(filename: str) -> FileResponse:
    """Safely resolve and return a file from DATA_DIR. Raises HTTPException on error."""
    # ensure .json suffix
    if not filename.endswith(".json"):
        filename = f"{filename}.json"
    p = (DATA_DIR / filename).resolve()
    # prevent directory traversal: parent must be the data dir
    if p.parent != RESOLVED_DATA_DIR:
        raise HTTPException(status_code=400, detail="Invalid file path")
    if not p.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(p)


@app.get("/api/airsenal/gw/{gw}/predictions")
def gw_predictions(gw: int):
    # serve gw_{gw}_predictions.json
    filename = f"gw_{gw}_predictions.json"
    return _serve_api_file(filename)


@app.get("/api/airsenal/gw/{gw}/fixtures_by_player")
def gw_fixtures_by_player(gw: int):
    """Return gw_{gw}_fixtures_by_player.json (e.g. gw 27)."""
    filename = f"gw_{gw}_fixtures_by_player.json"
    return _serve_api_file(filename)


@app.get("/api/airsenal/form_last4")
def form_last4():
    """Return form_last4.json"""
    return _serve_api_file("form_last4.json")


@app.get("/api/airsenal/bandwagons")
def bandwagons():
    """Return bandwagons.json"""
    return _serve_api_file("bandwagons.json")


@app.get("/api/files/{name}")
def files(name: str):
    filename = name if name.endswith(".json") else f"{name}.json"
    return _serve_api_file(filename)


# ---------------------------------------------------------------------------
# FDR endpoints
# ---------------------------------------------------------------------------

class PlayerRecord(BaseModel):
    """Player record for injury-adjusted FDR computation and explanations."""
    player_id: Optional[int] = None
    fpl_api_id: Optional[int] = None
    player_name: Optional[str] = None
    team: Optional[str] = None
    position: Optional[str] = None
    position_group: Optional[str] = None
    absence_type: Optional[str] = None
    minutes_last6: Optional[float] = None
    minutes_season: Optional[float] = None
    minutes_last10_before_absence: Optional[float] = None
    goals90: Optional[float] = None
    assists90: Optional[float] = None
    prob_available: Optional[float] = None
    status: Optional[str] = None
    starter_probability: Optional[float] = None
    matches_since_departure: Optional[float] = None
    source_news: Optional[str] = None
    last_updated: Optional[str] = None


class InjuryNewsRecord(BaseModel):
    player_id: Optional[int] = None
    fpl_api_id: Optional[int] = None
    player_name: Optional[str] = None
    team: str
    team_code: str
    position: Optional[str] = None
    position_group: Optional[str] = None
    absence_type: Optional[str] = None
    prob_available: Optional[float] = None
    status: Optional[str] = None
    starter_probability: Optional[float] = None
    matches_since_departure: Optional[float] = None
    source_news: Optional[str] = None
    last_updated: str


class InjuryNewsResponse(BaseModel):
    injuries: List[InjuryNewsRecord]


class EloRatingRecord(BaseModel):
    team_id: int
    team: str
    elo: float


class EloSnapshotResponse(BaseModel):
    snapshot_date: str
    ratings: List[EloRatingRecord]


class AbsenceDebugEntry(BaseModel):
    player_id: Optional[int] = None
    player_name: Optional[str] = None
    position: Optional[str] = None
    status: Optional[str] = None
    absence_type: Optional[str] = None
    side: Optional[str] = None
    layer: Optional[str] = None
    reason_counted_or_ignored: str
    attacking_impact: float
    defensive_impact: float


class FixtureFDRMetricsResponse(BaseModel):
    team: str
    team_id: Optional[int] = None
    opponent: str
    opponent_id: Optional[int] = None
    is_home: bool
    elo_team: float
    elo_opponent: float
    base_raw: float
    team_attack_loss: float
    team_defence_loss: float
    team_squad_change_attack_loss: float
    team_squad_change_defence_loss: float
    opp_attack_loss: float
    opp_defence_loss: float
    opp_squad_change_attack_loss: float
    opp_squad_change_defence_loss: float
    team_counted_absences: List[AbsenceDebugEntry]
    team_ignored_absences: List[AbsenceDebugEntry]
    opp_counted_absences: List[AbsenceDebugEntry]
    opp_ignored_absences: List[AbsenceDebugEntry]
    key_absences_counted: List[AbsenceDebugEntry]
    key_absences_ignored: List[AbsenceDebugEntry]
    raw_attack: float
    raw_defence: float
    raw_overall: float
    attack_fdr: float
    defence_fdr: float
    overall_fdr: float
    attack_fdr_int: int
    defence_fdr_int: int
    overall_fdr_int: int


class FixtureSaturatedResponse(BaseModel):
    fixture_id: Optional[int] = None
    snapshot_date: Optional[str] = None
    date: Optional[str] = None
    gameweek: Optional[int] = None
    season: Optional[str] = None
    home_team: str
    home_team_id: Optional[int] = None
    away_team: str
    away_team_id: Optional[int] = None
    team: str
    team_id: Optional[int] = None
    opponent: str
    opponent_id: Optional[int] = None
    is_home: bool
    team_players: List[PlayerRecord]
    opp_players: List[PlayerRecord]


class FixtureFDRResponse(BaseModel):
    saturated: FixtureSaturatedResponse
    fdr: FixtureFDRMetricsResponse


class TeamFixturesFDRResponse(RootModel[List[FixtureFDRResponse]]):
    pass


def _build_fixture_fdr_response(
    context: dict,
    snapshot_date: Optional[str],
    team_players: list[dict],
    opp_players: list[dict],
) -> FixtureFDRResponse:
    from src.services.fixture_fdr import compute_fixture_fdr

    fdr_result = compute_fixture_fdr(
        team_name=context["team"],
        opponent_name=context["opponent"],
        is_home=context["is_home"],
        team_players=team_players,
        opp_players=opp_players,
        snapshot_date=snapshot_date,
    )
    fdr_result.update(
        {
            "team_id": context.get("team_id"),
            "opponent_id": context.get("opponent_id"),
        }
    )

    saturated = {
        **context,
        "snapshot_date": snapshot_date,
        "team_players": team_players,
        "opp_players": opp_players,
    }
    return FixtureFDRResponse(
        saturated=FixtureSaturatedResponse.model_validate(saturated),
        fdr=FixtureFDRMetricsResponse.model_validate(fdr_result),
    )


class FixtureFDRRequest(BaseModel):
    fixture_id: Optional[int] = None
    team: Optional[str] = None
    opponent: Optional[str] = None
    is_home: Optional[bool] = None
    snapshot_date: Optional[str] = None
    team_players: Optional[List[PlayerRecord]] = None
    opp_players: Optional[List[PlayerRecord]] = None

    @model_validator(mode="after")
    def validate_input_mode(self) -> "FixtureFDRRequest":
        has_fixture_mode = self.fixture_id is not None
        has_team_mode = any(value is not None for value in (self.team, self.opponent, self.is_home))

        if not has_fixture_mode and not has_team_mode:
            raise ValueError("Provide either fixture_id or team/opponent/is_home")
        if not has_fixture_mode and not all(value is not None for value in (self.team, self.opponent, self.is_home)):
            raise ValueError("When not using fixture_id, provide team, opponent, and is_home together")
        if self.team and self.opponent and self.team == self.opponent:
            raise ValueError("team and opponent must be different")
        return self


@app.post("/api/fdr/fixture", response_model=FixtureFDRResponse)
def fixture_fdr(req: FixtureFDRRequest) -> FixtureFDRResponse:
    """Compute injury-adjusted FDR for a fixture with saturated context.

    Supported request modes:
    - ``fixture_id`` only, with optional ``team`` / ``opponent`` / ``is_home``
      hints validated against that fixture, or
    - ``team`` / ``opponent`` / ``is_home`` only.

    When team mode matches a fixture in ``fixtures.json``, the backend also
    fills ``fixture_id`` / date / gameweek / season into the saturated payload.

    If ``team_players`` or ``opp_players`` are omitted, the backend auto-loads
    current missing players for those teams from ``injury_news.json`` and then
    enriches them from ``data.db`` before computing FDR.
    """
    from src.services.injury_news import (
        canonical_team_name,
        enrich_player_records,
        get_team_injury_players,
        resolve_fixture_context,
        sanitize_snapshot_date,
    )

    try:
        context = resolve_fixture_context(
            fixture_id=req.fixture_id,
            team_name=req.team,
            opponent_name=req.opponent,
            is_home=req.is_home,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    snapshot_date = sanitize_snapshot_date(req.snapshot_date)
    provided_team_players = [p.model_dump() for p in req.team_players] if req.team_players is not None else None
    provided_opp_players = [p.model_dump() for p in req.opp_players] if req.opp_players is not None else None

    if provided_team_players is None:
        team_players = get_team_injury_players(context["team"])
    else:
        team_players = enrich_player_records(
            [
                {**player, "team": canonical_team_name(player.get("team") or context["team"])}
                for player in provided_team_players
            ]
        )

    if provided_opp_players is None:
        opp_players = get_team_injury_players(context["opponent"])
    else:
        opp_players = enrich_player_records(
            [
                {**player, "team": canonical_team_name(player.get("team") or context["opponent"])}
                for player in provided_opp_players
            ]
        )

    return _build_fixture_fdr_response(
        context=context,
        snapshot_date=snapshot_date,
        team_players=team_players,
        opp_players=opp_players,
    )



@app.get("/api/fdr/team/{team_name}", response_model=TeamFixturesFDRResponse)
def team_fixtures_fdr(
    team_name: str,
    next: int = Query(3, ge=1, le=38),
    snapshot_date: Optional[str] = None,
) -> TeamFixturesFDRResponse:
    """Return FDR results for a team's next N fixtures.

    ``team_name`` accepts either a readable team name/alias (for example
    ``Tottenham`` or ``Spurs``) or a numeric FPL ``team_id`` in the path.
    """
    from src.services.injury_news import (
        get_next_team_fixtures,
        get_team_injury_players,
        resolve_fixture_context,
        sanitize_snapshot_date,
    )

    try:
        fixtures = get_next_team_fixtures(team_name, limit=next)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    resolved_snapshot_date = sanitize_snapshot_date(snapshot_date)
    responses: list[FixtureFDRResponse] = []
    for fixture in fixtures:
        context = resolve_fixture_context(fixture_id=fixture["fixture_id"], team_name=team_name)
        team_players = get_team_injury_players(context["team"])
        opp_players = get_team_injury_players(context["opponent"])
        responses.append(
            _build_fixture_fdr_response(
                context=context,
                snapshot_date=resolved_snapshot_date,
                team_players=team_players,
                opp_players=opp_players,
            )
        )

    return TeamFixturesFDRResponse(responses)


@app.get("/api/fdr/injuries", response_model=InjuryNewsResponse, response_model_exclude_none=True)
def injury_news() -> InjuryNewsResponse:
    """Return normalized injury news for current Premier League teams.

    Data is sourced from ``data/api/injury_news.json`` and normalized with
    readable team names, availability probability, status, raw news text, and
    a file-derived ``last_updated`` timestamp.
    """
    from src.services.injury_news import load_current_injury_news

    return InjuryNewsResponse(
        injuries=[InjuryNewsRecord.model_validate(row) for row in load_current_injury_news()]
    )


@app.get("/api/fdr/elo", response_model=EloSnapshotResponse)
def elo_ratings(snapshot_date: Optional[str] = None) -> EloSnapshotResponse:
    """Return current ClubElo ratings for the current Premier League teams only.

    Pass ``?snapshot_date=YYYY-MM-DD`` to fetch ratings for a specific date
    (defaults to today UTC). Team names are normalized against ``teams.json`` so
    FPL names like ``Spurs`` and ``Man City`` line up with ClubElo.
    """
    from src.services.club_elo import fetch_premier_league_elo_snapshot

    snapshot = fetch_premier_league_elo_snapshot(snapshot_date)
    if not snapshot["ratings"]:
        raise HTTPException(status_code=503, detail="ClubElo ratings unavailable")
    return EloSnapshotResponse.model_validate(snapshot)

