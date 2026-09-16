# PROJECT KNOWLEDGE BASE

**Repo:** `fpl-copilot` (monorepo)  
**Layout:** `backend/` (FastAPI) + `frontend/` (React 19 + Vite 7) + vendored AIrsenal

## HIERARCHY

- `backend/` (FastAPI + services + adapters)
- `frontend/` (React 19 + Vite 7)
- `backend/AIrsenal/` (vendored ML optimizer git submodule)

## OVERVIEW

- Monorepo: two first-party packages + vendored AIrsenal (ML FPL optimizer).
- Backend: FastAPI + in-process asyncio worker + static JSON serving + SQLite job queue.
- Frontend: React 19/Vite 7 SPA on a Tailwind v4 design system (ported from the product redesign), React Compiler, dual API clients (own backend + official FPL proxy).
- AIrsenal: integration target, not routine edit surface. Requires Python >=3.10,<3.13.

## STRUCTURE

```text
fpl-copilot/
├── AGENTS.md
├── backend/                  # FastAPI + services + adapters
│   ├── src/main.py           # All routes, lifespan, worker loop
│   ├── src/services/         # Domain logic (15 modules)
│   ├── adapters/             # External system bridges (3 modules)
│   ├── data/api/             # Static JSON served by backend
│   ├── tests/                # Flat test suite (19 files)
│   ├── scripts/              # Shell helpers (backend.sh, airsenal.sh)
│   └── AIrsenal/             # Vendored upstream (git submodule)
├── frontend/                 # React 19 + Vite 7 + Tailwind v4 design system
│   ├── src/App.tsx           # Routes + ProtectedRoute + Core/Squad providers
│   ├── src/api/              # Dual API clients (backend/ + fpl/)
│   ├── src/components/       # ds/ primitives, layout/ shell, player/, shared.tsx
│   ├── src/pages/            # 7 page components
│   ├── src/hooks/            # useCoreData (single fetch fan-out)
│   ├── src/domain/           # Normalized models + projections
│   └── src/context/          # TeamId, Theme, Toast, Core, Squad contexts
└── backend/AIrsenal/         # Vendored ML optimizer (separate venv)
```

## COMMANDS

### Backend (`backend/`)

```bash
# Dev server (interactive — prompts for host/port, loads .env.local)
cd backend && source scripts/backend.sh

# Dev server (non-interactive)
cd backend && . .venv/bin/activate && uvicorn src.main:app --reload --app-dir . --host 127.0.0.1 --port 8000

# Run ALL tests
cd backend && . .venv/bin/activate && pytest tests/

# Single test execution patterns (ESSENTIAL)
cd backend && . .venv/bin/activate && pytest tests/test_fdr.py # File
cd backend && . .venv/bin/activate && pytest tests/test_fdr.py::TestSigmoid # Class
cd backend && . .venv/bin/activate && pytest tests/test_fdr.py::TestSigmoid::test_at_zero # Method
cd backend && . .venv/bin/activate && pytest tests/ -k "injury" # Keyword search
```

### Frontend (`frontend/`)

```bash
# Dev server (with dual proxy: /api → FastAPI, /fpl-api → FPL official)
cd frontend && npm run dev

# Type-check + production build
cd frontend && npm run build

# Lint (ESLint 9 flat config)
cd frontend && npm run lint

# Preview production build
cd frontend && npm run preview
```

### AIrsenal (`backend/AIrsenal/`)

```bash
# Interactive shell (activates AIrsenal venv + AIRSENAL_HOME)
cd backend && source scripts/airsenal.sh

# AIrsenal tests (from AIrsenal directory, using its own venv)
cd backend/AIrsenal && . .venv/bin/activate && pytest airsenal/tests

# AIrsenal lint (Ruff)
cd backend/AIrsenal && . .venv/bin/activate && ruff check --fix .
```

## CODE STYLE & CONVENTIONS

### Backend (Python)

- **Naming:** `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants, `_leading_underscore` for private helpers.
- **Imports:** `from __future__ import annotations` required. Order: Stdlib → 3rd-party → Local.
- **Lazy imports:** Heavy services imported inside route handlers to minimize startup cost and circular deps.
- **Pydantic v2:** All API request/response schemas use Pydantic models with `ConfigDict(extra="forbid")`.
- **Typing:** Strict use of `Optional`, `list`, `dict`. No `Any` abuse.
- **Error Handling:** `HTTPException` with appropriate status codes. Custom exceptions (e.g. `AirsenalRunError`) for sub-processes.
- **Formatting:** No global linter/formatter; match surrounding style.

### Frontend (TypeScript/React)

- **Naming:** `PascalCase` components, `camelCase` hooks/utils (prefixed with `use`).
- **TypeScript:** Strict mode enabled. No `as any` or `@ts-ignore`.
- **React Compiler:** Auto-memoizes. **DO NOT** use manual `useMemo` or `useCallback`.
- **API Calls:** Must go through `src/api/backend/` or `src/api/fpl/`. No raw `fetch`.
- **Styling:** Tailwind v4 + the ported design-system classes in `src/index.css`. No manual CSS unless essential.
- **Imports:** No barrel exports for component directories (direct file imports only).

## ARCHITECTURE MAP (WHERE TO LOOK)

| Task | Location | Notes |
|------|----------|-------|
| Backend routes | `backend/src/main.py` | All routes inline, no APIRouter split |
| Background worker | `backend/src/main.py` (lifespan) | In-process asyncio, polls every 2s |
| Domain services | `backend/src/services/` | Lazy-imported, 15 modules |
| External adapters | `backend/adapters/` | Sibling to `src/`, not inside it |
| Backend tests | `backend/tests/` | Flat structure, class-based grouping |
| Frontend routing | `frontend/src/App.tsx` | ProtectedRoute wraps all except /login |
| UI Components | `frontend/src/components/` | `ds/` primitives + `layout/` shell |
| Data Hooks | `frontend/src/hooks/` | `useCoreData` (shared via `CoreProvider`) |

## ANTI-PATTERNS

- No direct file-serving endpoints bypassing `_serve_api_file` (backend).
- No AIrsenal commands from the backend app venv (use `backend/AIrsenal/.venv`).
- No treating vendored AIrsenal as first-party code.
- No secrets committed; use `backend/.env.local` or `backend/.airsenal_home/`.
- No `as any` / `@ts-ignore` in frontend.
- No direct FPL/backend API calls bypassing typed API clients.
- No manual `useMemo`/`useCallback` — React Compiler handles it.

## UNIQUE STYLES

- **Auth = FPL Team ID in localStorage** (no OAuth/JWT). `ProtectedRoute` wraps all routes.
- **In-process asyncio worker** instead of Celery/RQ. Polls every 2s.
- **Static JSON file serving** mixed with compute endpoints in single `main.py`.
- **Dual Vite proxy:** `/api` → FastAPI backend, `/fpl-api` → official FPL API.
- **No test framework in frontend** yet (Vitest is preferred if needed).
