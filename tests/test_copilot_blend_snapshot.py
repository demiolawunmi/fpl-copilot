from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_write_blend_snapshot_per_team_file(tmp_path: Path, monkeypatch) -> None:
    from src.services import copilot_blend_snapshot as mod

    monkeypatch.setattr(mod, "_find_data_api_dir", lambda: tmp_path)

    out = mod.write_blend_snapshot(
        job_id="job-abc",
        schema_version="1.0",
        correlation_id="corr-1",
        input_payload={
            "gameweek": 18,
            "fpl_team_id": 8994418,
            "source_weights": {"elo": 0.6, "airsenal": 0.4},
            "task": "hybrid",
        },
        result_payload={
            "schema_version": "1.0",
            "correlation_id": "corr-1",
            "core": {"summary": "ok", "confidence": 0.8},
            "recommended_transfers": [],
            "ask_copilot": {"answer": "hold", "rationale": [], "confidence": 0.8},
            "degraded_mode": {
                "is_degraded": False,
                "fallback_used": False,
            },
        },
    )
    assert out is not None
    target = tmp_path / "gw_18_copilot_blend_8994418.json"
    assert target.is_file()


def test_write_blend_snapshot_global_when_no_team_id(tmp_path: Path, monkeypatch) -> None:
    from src.services import copilot_blend_snapshot as mod

    monkeypatch.setattr(mod, "_find_data_api_dir", lambda: tmp_path)

    out = mod.write_blend_snapshot(
        job_id="job-x",
        schema_version="1.0",
        correlation_id="corr-2",
        input_payload={
            "gameweek": 18,
            "source_weights": {"elo": 0.5, "airsenal": 0.5},
            "task": "hybrid",
        },
        result_payload={
            "schema_version": "1.0",
            "correlation_id": "corr-2",
            "core": {"summary": "ok", "confidence": 0.5},
            "recommended_transfers": [],
            "ask_copilot": {"answer": "ok", "rationale": [], "confidence": 0.5},
            "degraded_mode": {"is_degraded": False, "fallback_used": False},
        },
    )
    assert out is not None
    assert (tmp_path / "gw_18_copilot_blend.json").is_file()
