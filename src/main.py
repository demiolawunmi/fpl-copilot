# backend_repo/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel

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
    """Minimal player record used for injury-adjusted FDR computation."""
    position: str
    minutes_last6: Optional[float] = None
    goals90: Optional[float] = None
    assists90: Optional[float] = None
    prob_available: Optional[float] = None
    status: Optional[str] = None


class FixtureFDRRequest(BaseModel):
    team: str
    opponent: str
    is_home: bool
    snapshot_date: Optional[str] = None
    team_players: Optional[List[PlayerRecord]] = None
    opp_players: Optional[List[PlayerRecord]] = None


@app.post("/api/fdr/fixture")
def fixture_fdr(req: FixtureFDRRequest):
    """Compute attack, defence and overall FDR for a single fixture.

    Uses ClubElo ratings (fetched automatically unless unavailable) and
    optional per-player injury data to produce a detailed difficulty
    breakdown.
    """
    from src.services.fixture_fdr import compute_fixture_fdr

    team_players = [p.model_dump() for p in req.team_players] if req.team_players else []
    opp_players = [p.model_dump() for p in req.opp_players] if req.opp_players else []

    result = compute_fixture_fdr(
        team_name=req.team,
        opponent_name=req.opponent,
        is_home=req.is_home,
        team_players=team_players,
        opp_players=opp_players,
        snapshot_date=req.snapshot_date,
    )
    return result


@app.get("/api/fdr/elo")
def elo_ratings(snapshot_date: Optional[str] = None):
    """Return the current ClubElo ratings for all clubs.

    Pass ``?snapshot_date=YYYY-MM-DD`` to fetch ratings for a specific date
    (defaults to today UTC).
    """
    from src.services.club_elo import fetch_elo_ratings

    ratings = fetch_elo_ratings(snapshot_date)
    if not ratings:
        raise HTTPException(status_code=503, detail="ClubElo ratings unavailable")
    return ratings
