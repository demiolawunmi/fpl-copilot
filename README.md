# FPL Copilot

Monorepo for **FPL Copilot** — an AI co-manager for Fantasy Premier League.

It combines the official FPL data and the AIrsenal expected-points models into
decisions: who to start, who to captain, who to buy and sell, and which model to
trust.

## Layout

```text
fpl-copilot/
├── backend/     # FastAPI API + services + adapters (+ AIrsenal submodule)
├── frontend/    # React 19 + Vite 7 + Tailwind v4 SPA
└── AGENTS.md    # agent/contributor knowledge base
```

- **backend/** — FastAPI app (`src/main.py`), domain services (`src/services/`),
  external adapters (`adapters/`), static JSON in `data/api/`, tests in `tests/`.
  AIrsenal is vendored as a git submodule at `backend/AIrsenal/`.
- **frontend/** — single-page app; dual API clients (own backend + official FPL
  proxy), design-system shell, and the Review/Plan/Explore screens.

## Quick start

```bash
# First clone: fetch the AIrsenal submodule
git submodule update --init --recursive

# Backend (FastAPI on :8000)
cd backend
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt   # or per your setup
source scripts/backend.sh

# Frontend (Vite on :5173, proxies /api -> :8000 and /fpl-api -> FPL)
cd frontend
npm install
npm run dev
```

See `backend/AGENTS.md` and `frontend/AGENTS.md` for package-specific commands and
conventions.

## Notes

- Auth is the FPL Team ID stored in `localStorage` (no OAuth/JWT).
- Secrets/config live in `backend/.env.local` and `backend/.airsenal_home/`
  (both gitignored).
- Backend blend/chat endpoints require an `OPENROUTER_API_KEY`.
