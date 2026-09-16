from pathlib import Path
import sqlite3
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.services.copilot_airsenal_extractor import CopilotAirsenalExtractor


def _seed_airsenal_db(db_path: Path) -> None:
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE player (
            player_id INTEGER NOT NULL,
            fpl_api_id INTEGER,
            name VARCHAR(100) NOT NULL,
            display_name VARCHAR(100),
            opta_code VARCHAR,
            PRIMARY KEY (player_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE player_prediction (
            id INTEGER NOT NULL,
            fixture_id INTEGER,
            predicted_points FLOAT NOT NULL,
            tag VARCHAR(100) NOT NULL,
            player_id INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(player_id) REFERENCES player (player_id)
        );
        """
    )
    con.executemany(
        """
        INSERT INTO player (player_id, fpl_api_id, name, display_name)
        VALUES (?, ?, ?, ?);
        """,
        [
            (101, 11, "Saka", "Bukayo Saka"),
            (202, 22, "Haaland", "Erling Haaland"),
            (303, 33, "Palmer", "Cole Palmer"),
        ],
    )
    con.executemany(
        """
        INSERT INTO player_prediction (id, fixture_id, predicted_points, tag, player_id)
        VALUES (?, ?, ?, ?, ?);
        """,
        [
            (1, 1, 3.0, "tag-2024-01-01", 101),
            (2, 2, 4.0, "tag-2024-01-01", 101),
            (3, 1, 5.0, "tag-2024-01-01", 202),
            (4, 1, 2.0, "tag-2024-01-01", 303),
            (5, 3, 6.0, "tag-2024-01-15", 101),
            (6, 4, 7.0, "tag-2024-01-15", 101),
            (7, 3, 8.0, "tag-2024-01-15", 202),
            (8, 4, 3.0, "tag-2024-01-15", 202),
            (9, 3, 4.0, "tag-2024-01-15", 303),
        ],
    )
    con.commit()
    con.close()


def test_returns_predictions_for_latest_tag_only(tmp_path: Path) -> None:
    db_path = tmp_path / "airsenal.db"
    _seed_airsenal_db(db_path)

    extractor = CopilotAirsenalExtractor(db_path)
    predictions = extractor.get_player_predictions()

    player_ids = {p["player_id"] for p in predictions}
    assert player_ids == {101, 202, 303}

    saka = next(p for p in predictions if p["player_id"] == 101)
    assert saka["predicted_points"] == pytest.approx(13.0)
    assert saka["fpl_api_id"] == 11
    assert saka["player_name"] == "Bukayo Saka"

    haaland = next(p for p in predictions if p["player_id"] == 202)
    assert haaland["predicted_points"] == pytest.approx(11.0)

    palmer = next(p for p in predictions if p["player_id"] == 303)
    assert palmer["predicted_points"] == pytest.approx(4.0)


def test_returns_correct_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "airsenal.db"
    _seed_airsenal_db(db_path)

    extractor = CopilotAirsenalExtractor(db_path)
    predictions = extractor.get_player_predictions()

    assert len(predictions) == 3
    for pred in predictions:
        assert set(pred.keys()) == {"player_id", "fpl_api_id", "player_name", "predicted_points"}
        assert isinstance(pred["player_id"], int)
        assert isinstance(pred["fpl_api_id"], int)
        assert isinstance(pred["player_name"], str)
        assert isinstance(pred["predicted_points"], float)


def test_empty_db_returns_empty_list(tmp_path: Path) -> None:
    db_path = tmp_path / "empty.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE player (
            player_id INTEGER NOT NULL,
            fpl_api_id INTEGER,
            name VARCHAR(100) NOT NULL,
            display_name VARCHAR(100),
            opta_code VARCHAR,
            PRIMARY KEY (player_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE player_prediction (
            id INTEGER NOT NULL,
            fixture_id INTEGER,
            predicted_points FLOAT NOT NULL,
            tag VARCHAR(100) NOT NULL,
            player_id INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(player_id) REFERENCES player (player_id)
        );
        """
    )
    con.commit()
    con.close()

    extractor = CopilotAirsenalExtractor(db_path)
    predictions = extractor.get_player_predictions()

    assert predictions == []


def test_sums_predictions_across_multiple_fixtures(tmp_path: Path) -> None:
    db_path = tmp_path / "airsenal.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE player (
            player_id INTEGER NOT NULL,
            fpl_api_id INTEGER,
            name VARCHAR(100) NOT NULL,
            display_name VARCHAR(100),
            opta_code VARCHAR,
            PRIMARY KEY (player_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE player_prediction (
            id INTEGER NOT NULL,
            fixture_id INTEGER,
            predicted_points FLOAT NOT NULL,
            tag VARCHAR(100) NOT NULL,
            player_id INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(player_id) REFERENCES player (player_id)
        );
        """
    )
    con.execute(
        "INSERT INTO player (player_id, name, display_name) VALUES (1, 'Salah', 'Mohamed Salah');"
    )
    con.executemany(
        """
        INSERT INTO player_prediction (id, fixture_id, predicted_points, tag, player_id)
        VALUES (?, ?, ?, ?, ?);
        """,
        [
            (1, 10, 5.5, "tag-abc", 1),
            (2, 11, 3.5, "tag-abc", 1),
            (3, 12, 4.0, "tag-abc", 1),
        ],
    )
    con.commit()
    con.close()

    extractor = CopilotAirsenalExtractor(db_path)
    predictions = extractor.get_player_predictions()

    assert len(predictions) == 1
    assert predictions[0]["player_id"] == 1
    assert predictions[0]["player_name"] == "Mohamed Salah"
    assert predictions[0]["predicted_points"] == pytest.approx(13.0)


def test_uses_display_name_fallback_to_name(tmp_path: Path) -> None:
    db_path = tmp_path / "airsenal.db"
    con = sqlite3.connect(db_path)
    con.execute(
        """
        CREATE TABLE player (
            player_id INTEGER NOT NULL,
            fpl_api_id INTEGER,
            name VARCHAR(100) NOT NULL,
            display_name VARCHAR(100),
            opta_code VARCHAR,
            PRIMARY KEY (player_id)
        );
        """
    )
    con.execute(
        """
        CREATE TABLE player_prediction (
            id INTEGER NOT NULL,
            fixture_id INTEGER,
            predicted_points FLOAT NOT NULL,
            tag VARCHAR(100) NOT NULL,
            player_id INTEGER,
            PRIMARY KEY (id),
            FOREIGN KEY(player_id) REFERENCES player (player_id)
        );
        """
    )
    con.execute(
        "INSERT INTO player (player_id, name, display_name) VALUES (1, 'De Bruyne', NULL);"
    )
    con.execute(
        "INSERT INTO player_prediction (id, fixture_id, predicted_points, tag, player_id) "
        "VALUES (1, 1, 7.0, 'tag-x', 1);"
    )
    con.commit()
    con.close()

    extractor = CopilotAirsenalExtractor(db_path)
    predictions = extractor.get_player_predictions()

    assert len(predictions) == 1
    assert predictions[0]["player_name"] == "De Bruyne"
