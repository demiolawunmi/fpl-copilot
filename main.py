# backend_repo/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path

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

DATA_DIR = Path(__file__).resolve().parent / "data" / "api"
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
