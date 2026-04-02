from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_blend_sql import CopilotSqlBlendAssembler


def _seed_blend_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE copilot_source_player_scores (
            source TEXT NOT NULL,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            projected_points REAL NOT NULL
        );
        """
    )
    con.executemany(
        """
        INSERT INTO copilot_source_player_scores (source, player_id, player_name, projected_points)
        VALUES (?, ?, ?, ?);
        """,
        [
            ("fplcopilot", 101, "Saka", 10.0),
            ("airsenal", 101, "Saka", 8.0),
            ("fplcopilot", 202, "Haaland", 9.0),
            ("airsenal", 202, "Haaland", 11.0),
            ("fplcopilot", 303, "Palmer", 5.0),
            ("airsenal", 303, "Palmer", 5.0),
        ],
    )
    con.commit()
    con.close()


def test_happy_path_weighted_blend_060_040(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)
    context = assembler.assemble_model_context(
        source_weights={"fplcopilot": 0.6, "airsenal": 0.4},
        limit=10,
    )

    assert context["schema_version"] == "1.0"
    assert context["weights"] == {"fplcopilot": 0.6, "airsenal": 0.4}

    players = context["blended_players"]
    assert [p["player_id"] for p in players] == [202, 101, 303]
    assert players[0]["blended_projected_points"] == pytest.approx(9.8)
    assert players[1]["blended_projected_points"] == pytest.approx(9.2)
    assert players[2]["blended_projected_points"] == pytest.approx(5.0)


def test_blend_is_deterministic_for_identical_inputs(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)
    first = assembler.assemble_model_context(
        source_weights={"fplcopilot": 0.6, "airsenal": 0.4},
        player_name_contains="a",
        limit=5,
    )
    second = assembler.assemble_model_context(
        source_weights={"fplcopilot": 0.6, "airsenal": 0.4},
        player_name_contains="a",
        limit=5,
    )

    assert first == second


def test_invalid_weight_rejected_clearly(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)

    with pytest.raises(ValueError, match="must sum to 1.0"):
        assembler.assemble_model_context(source_weights={"fplcopilot": 0.8, "airsenal": 0.4})


def test_negative_weight_rejected_clearly(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)

    with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
        assembler.assemble_model_context(source_weights={"fplcopilot": 1.1, "airsenal": -0.1})


def test_malicious_source_identifier_rejected_before_sql_execution(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)

    with pytest.raises(ValueError, match="must include exactly"):
        assembler.assemble_model_context(
            source_weights={
                "fplcopilot": 0.6,
                "airsenal'); DROP TABLE copilot_source_player_scores; --": 0.4,
            }
        )

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT COUNT(*) FROM copilot_source_player_scores;"
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] == 6


def test_malicious_filter_input_cannot_alter_sql_behavior(tmp_path: Path) -> None:
    db_path = tmp_path / "blend.db"
    _seed_blend_db(db_path)

    assembler = CopilotSqlBlendAssembler(db_path)
    malicious = "x'); DROP TABLE copilot_source_player_scores; --"

    context = assembler.assemble_model_context(
        source_weights={"fplcopilot": 0.6, "airsenal": 0.4},
        player_name_contains=malicious,
        limit=10,
    )

    assert context["blended_players"] == []

    con = sqlite3.connect(db_path)
    row = con.execute(
        "SELECT COUNT(*) FROM copilot_source_player_scores;"
    ).fetchone()
    con.close()

    assert row is not None
    assert row[0] == 6
