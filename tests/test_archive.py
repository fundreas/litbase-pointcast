"""The committed daily archive: what the API can never give back."""

from __future__ import annotations

import gzip
from datetime import date

import pandas as pd

from kickbase_xp import archive, db

DAY = date(2026, 9, 10)
ROWS = [
    (DAY.isoformat(), "p1", 0, 5_000_000, 5, "1"),
    (DAY.isoformat(), "p2", 1, 2_000_000, 2, "2"),
]


def _seed(conn):
    db.upsert_status_snapshots(conn, ROWS)
    conn.commit()


def test_day_number_matches_the_kickbase_encoding():
    # Kickbase indexes market values by days since the Unix epoch.
    assert archive.day_number(date(1970, 1, 1)) == 0
    assert archive.day_number(date(2026, 9, 10)) == 20706


def test_snapshot_round_trips_through_a_fresh_database(tmp_path):
    source = db.connect(tmp_path / "a.sqlite")
    _seed(source)
    archive.write_snapshot(source, tmp_path / "snapshots", DAY)
    source.close()

    # A fresh CI runner: no SQLite file, only the committed snapshots.
    target = db.connect(tmp_path / "b.sqlite")
    assert archive.load_snapshots(target, tmp_path / "snapshots") == 1
    got = pd.read_sql_query(
        "SELECT player_id, status, market_value, lineup_prob, team_id"
        " FROM status_snapshots ORDER BY player_id",
        target,
    )
    assert list(got["player_id"]) == ["p1", "p2"]
    assert list(got["status"]) == [0, 1]
    assert list(got["team_id"]) == ["1", "2"]
    target.close()


def test_snapshots_also_extend_the_market_value_series(tmp_path):
    """The API only serves 365 days; the archive is what reaches further back."""
    source = db.connect(tmp_path / "a.sqlite")
    _seed(source)
    archive.write_snapshot(source, tmp_path / "snapshots", DAY)
    source.close()

    target = db.connect(tmp_path / "b.sqlite")
    archive.load_snapshots(target, tmp_path / "snapshots")
    mv = pd.read_sql_query("SELECT * FROM market_values ORDER BY player_id", target)
    assert list(mv["day"]) == [archive.day_number(DAY)] * 2
    assert list(mv["value"]) == [5_000_000.0, 2_000_000.0]
    target.close()


def test_a_fresher_api_value_is_not_overwritten_by_the_archive(tmp_path):
    source = db.connect(tmp_path / "a.sqlite")
    _seed(source)
    archive.write_snapshot(source, tmp_path / "snapshots", DAY)
    source.close()

    target = db.connect(tmp_path / "b.sqlite")
    db.upsert_market_values(target, [("p1", archive.day_number(DAY), 9_999_999.0)])
    target.commit()
    archive.load_snapshots(target, tmp_path / "snapshots")
    value = target.execute(
        "SELECT value FROM market_values WHERE player_id = 'p1'"
    ).fetchone()[0]
    assert value == 9_999_999.0
    target.close()


def test_loading_is_idempotent(tmp_path):
    source = db.connect(tmp_path / "a.sqlite")
    _seed(source)
    archive.write_snapshot(source, tmp_path / "snapshots", DAY)
    source.close()

    target = db.connect(tmp_path / "b.sqlite")
    archive.load_snapshots(target, tmp_path / "snapshots")
    archive.load_snapshots(target, tmp_path / "snapshots")
    n = target.execute("SELECT COUNT(*) FROM status_snapshots").fetchone()[0]
    assert n == 2
    target.close()


def test_missing_directory_is_not_an_error(tmp_path):
    conn = db.connect(tmp_path / "a.sqlite")
    assert archive.load_snapshots(conn, tmp_path / "nope") == 0
    conn.close()


def test_unrecognised_files_are_skipped(tmp_path):
    snaps = tmp_path / "snapshots"
    snaps.mkdir()
    (snaps / "README.md").write_text("not a snapshot", encoding="utf-8")
    with gzip.open(snaps / "not-a-date.csv.gz", "wt", encoding="utf-8") as fh:
        fh.write("player_id\np1\n")
    conn = db.connect(tmp_path / "a.sqlite")
    assert archive.load_snapshots(conn, snaps) == 0
    conn.close()


def test_snapshot_file_is_small_and_readable(tmp_path):
    source = db.connect(tmp_path / "a.sqlite")
    db.upsert_status_snapshots(
        source,
        [(DAY.isoformat(), f"p{i}", 0, 1_000_000 + i, 3, "1") for i in range(461)],
    )
    source.commit()
    path = archive.write_snapshot(source, tmp_path / "snapshots", DAY)
    source.close()

    # A full league-day costs a handful of kilobytes -- the whole point of
    # committing these instead of a rewritten 16 MB database.
    assert path.stat().st_size < 20_000
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = fh.readline().strip()
    assert header == ",".join(archive.FIELDS)
