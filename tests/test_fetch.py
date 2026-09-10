"""Data layer: the API-shaped payloads land in SQLite the way features expect."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kickbase_xp import db, features
from kickbase_xp.fetch import run_fetch

TABLE = [{"tid": "1", "tn": "Alpha"}, {"tid": "2", "tn": "Beta"}]

SQUADS = {
    "1": {"it": [{"i": "p1", "n": "Adams", "pos": 3, "st": 0, "mv": 5_000_000, "ap": 90,
                  "tp": 900, "prob": 5}]},
    "2": {"it": [{"i": "p2", "n": "Bauer", "pos": 4, "st": 1, "mv": 2_000_000, "ap": 40,
                  "tp": 400, "prob": 2}]},
}

# One Bundesliga season and one 2. Bundesliga season, exactly as the real feed
# interleaves them -- note the higher season id on the *lower* division.
PERFORMANCE = {
    "p1": [
        {"sid": "42", "ti": "2026/2027", "n": "Bundesliga", "ph": [
            {"mi": "m1", "day": 1, "p": 120, "mp": "90'", "md": "2026-08-28T18:30:00Z",
             "t1": "1", "t2": "2", "t1g": 1, "t2g": 0, "pt": "1", "st": 5, "mdst": 2},
            {"mi": "m2", "day": 2, "md": "2026-09-13T15:30:00Z",
             "t1": "2", "t2": "1", "st": 0, "mdst": 0},
        ]},
        {"sid": "43", "ti": "2026/2027", "n": "2. Bundesliga", "ph": [
            {"mi": "x1", "day": 1, "p": 999, "mp": "90'", "md": "2026-08-01T18:30:00Z",
             "t1": "1", "t2": "2", "pt": "1", "st": 5, "mdst": 2},
        ]},
    ],
    "p2": [
        {"sid": "42", "ti": "2026/2027", "n": "Bundesliga", "ph": [
            {"mi": "m1", "day": 1, "mp": "0'", "md": "2026-08-28T18:30:00Z",
             "t1": "1", "t2": "2", "t1g": 1, "t2g": 0, "pt": "2", "st": 4, "mdst": 2},
            {"mi": "m2", "day": 2, "md": "2026-09-13T15:30:00Z",
             "t1": "2", "t2": "1", "st": 0, "mdst": 0},
        ]},
    ],
}

MATCHDAYS = {
    "day": 2,
    "it": [
        {"day": 1, "it": [{"mi": "m1", "day": 1, "dt": "2026-08-28T18:30:00Z",
                           "t1": "1", "t2": "2", "t1g": 1, "t2g": 0, "st": 2}]},
        {"day": 2, "it": [{"mi": "m2", "day": 2, "dt": "2026-09-13T15:30:00Z",
                           "t1": "2", "t2": "1", "st": 0}]},
    ],
}

MARKET_VALUES = {
    "p1": [{"dt": 20700, "mv": 5_000_000.0}, {"dt": 20701, "mv": 5_100_000.0}],
    "p2": [{"dt": 20700, "mv": 2_000_000.0}],
}


class FakeClient:
    """Stands in for KickbaseClient; records what the pipeline asked for."""

    competition_id = "1"

    def __init__(self):
        self.calls: list[str] = []

    def table(self):
        self.calls.append("table")
        return TABLE

    def team_profile(self, team_id):
        self.calls.append(f"team_profile:{team_id}")
        return SQUADS[team_id]

    def matchdays(self):
        self.calls.append("matchdays")
        return MATCHDAYS

    def player_performance(self, player_id):
        self.calls.append(f"performance:{player_id}")
        return PERFORMANCE[player_id]

    def player_market_value(self, player_id, timeframe=365):
        self.calls.append(f"mv:{player_id}")
        return MARKET_VALUES[player_id]


@pytest.fixture
def fetched(tmp_path):
    conn = db.connect(tmp_path / "f.sqlite")
    client = FakeClient()
    # An explicit snapshot dir: the default points at the repo's real archive.
    result = run_fetch(client, conn, snapshot_dir=tmp_path / "snapshots")
    yield conn, client, result
    conn.close()


def test_every_table_is_populated(fetched):
    conn, _, result = fetched
    counts = result["counts"]
    assert counts["teams"] == 2
    assert counts["players"] == 2
    assert counts["matches"] == 2
    assert counts["market_values"] == 3
    assert counts["status_snapshots"] == 2


def test_current_season_ignores_other_competitions(fetched):
    """Season 43 is 2. Bundesliga and has the higher id -- 42 must still win."""
    _, _, result = fetched
    assert result["season_id"] == "42"
    assert result["current_matchday"] == 2


def test_competition_is_recorded_per_row(fetched):
    conn, _, _ = fetched
    rows = pd.read_sql_query(
        "SELECT competition, COUNT(*) n FROM performances GROUP BY competition", conn
    ).set_index("competition")["n"]
    assert rows["Bundesliga"] == 4
    assert rows["2. Bundesliga"] == 1


def test_features_exclude_the_other_competition(fetched):
    conn, _, _ = fetched
    matrix = features.build_matrix(
        conn, now=pd.Timestamp("2026-09-01", tz="UTC"), max_seasons=None
    )
    assert set(matrix["season_id"]) == {"42"}
    # The 999-point 2. Bundesliga match must not colour p1's form.
    p1_md2 = matrix[(matrix["player_id"] == "p1") & (matrix["matchday"] == 2)].iloc[0]
    assert p1_md2["pts_last"] == 120


def test_minutes_of_zero_mean_did_not_play(fetched):
    conn, _, _ = fetched
    matrix = features.build_matrix(
        conn, now=pd.Timestamp("2026-09-01", tz="UTC"), max_seasons=None
    )
    p2_md1 = matrix[(matrix["player_id"] == "p2") & (matrix["matchday"] == 1)].iloc[0]
    assert p2_md1["played"] == 0.0
    assert p2_md1["points"] == 0.0  # absent from the pitch scores nothing, not NaN


def test_a_snapshot_is_written_for_today(fetched):
    _, _, result = fetched
    path = Path(result["snapshot"])
    assert path.exists()
    assert path.name.endswith(".csv.gz")


def test_refetching_reaches_a_steady_state(tmp_path):
    """Every write is an upsert, so re-running must not accumulate rows.

    The first run is the one exception, and deliberately so: it writes
    today's snapshot, which the second run replays as an extra market-value
    point per player (today's value, which the API's own series does not
    carry until tomorrow). From run two on, nothing moves.
    """
    conn = db.connect(tmp_path / "f.sqlite")
    snaps = tmp_path / "snapshots"
    first = run_fetch(FakeClient(), conn, snapshot_dir=snaps)
    second = run_fetch(FakeClient(), conn, snapshot_dir=snaps)
    third = run_fetch(FakeClient(), conn, snapshot_dir=snaps)

    assert second["counts"] == third["counts"]
    extra = second["counts"]["market_values"] - first["counts"]["market_values"]
    assert extra == 2  # one per player, for today
    for table in ("teams", "players", "matches", "performances", "status_snapshots"):
        assert first["counts"][table] == second["counts"][table]
    conn.close()
