# Official FPL fixture difficulty — pipeline

## Where it runs

1. **Fixture context** — `resolve_fixture_context` / `get_next_team_fixtures` in `src/services/injury_news.py` (unchanged: still driven by `fixtures.json` and team resolution).
2. **Official FPL difficulty** — `build_official_fpl_fields(context)` in `src/services/fpl_official_fdr.py` is called from `_build_fixture_fdr_response` in `src/main.py` **after** the context dict exists and **before** `compute_fixture_fdr` runs.
3. **Custom Copilot FDR** — `compute_fixture_fdr` in `src/services/fixture_fdr.py` (Elo + injury + squad-change) is unchanged; its output stays in the `fdr` object.

Official integers come only from `GET https://fantasy.premierleague.com/api/fixtures/` (cached 1h, stale fallback on error). They are merged into **`saturated`** as `official_fpl_*` fields, not into the custom `attack_fdr` / `defence_fdr` / `overall_fdr` metrics.

## Join keys

Lookup prefers **FPL fixture `id`** when it matches `context["fixture_id"]`. If that misses (e.g. internal id differs from FPL), the service falls back to **`(home_team_id, away_team_id, gameweek)`** using FPL team ids already on the saturated context.
