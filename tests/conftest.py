"""Synthetic two-team league used by the feature and model tests.

Small enough to reason about by hand, complete enough that every join in
`features.py` has something to bite on.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from kickbase_xp import db
from kickbase_xp.config import DEFAULT_SNAPSHOT_DIR

COMPETITION = "Bundesliga"
SEASON = "42"
PREV_SEASON = "34"
# Weekly matchdays from SEASON_START. Chosen so that with NOW as "today",
# matchdays 1-10 are finished and matchday 11 (2026-10-03) is still to come.
SEASON_START = datetime(2026, 7, 25, 15, 30, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)

TEAMS = [("1", "Alpha"), ("2", "Beta")]
# (player_id, team_id, position)
SQUAD = [
    ("p1", "1", 1),
    ("p2", "1", 2),
    ("p3", "1", 3),
    ("p4", "1", 4),
    ("p5", "2", 1),
    ("p6", "2", 2),
    ("p7", "2", 3),
    ("p8", "2", 4),
]

# Ten matchdays played, matchday 11 scheduled.
PLAYED_MATCHDAYS = 10
SCHEDULED_MATCHDAY = 11


def _kickoff(matchday: int, season_offset_days: int = 0) -> datetime:
    return SEASON_START + timedelta(days=7 * (matchday - 1) + season_offset_days)


def _perf_rows() -> list[tuple]:
    rows = []
    for season, offset in ((PREV_SEASON, -400), (SEASON, 0)):
        for md in range(1, PLAYED_MATCHDAYS + 1):
            kickoff = _kickoff(md, offset)
            home, away = ("1", "2") if md % 2 else ("2", "1")
            for pid, team, pos in SQUAD:
                # p4 is a rotation player: only appears on even matchdays.
                appears = not (pid == "p4" and md % 2 == 1)
                minutes = 90 if appears else 0
                points = (40 + 12 * pos + 7 * md) if appears else None
                rows.append(
                    (
                        pid,
                        season,
                        md,
                        f"{season}-{md}",
                        COMPETITION,
                        f"20{season}",
                        points,
                        minutes,
                        kickoff.isoformat(),
                        home,
                        away,
                        2,
                        1,
                        team,
                        5 if appears else 4,
                        2,
                    )
                )
    # The fixture we predict.
    kickoff = _kickoff(SCHEDULED_MATCHDAY)
    for pid, team, _pos in SQUAD:
        rows.append(
            (
                pid,
                SEASON,
                SCHEDULED_MATCHDAY,
                f"{SEASON}-{SCHEDULED_MATCHDAY}",
                COMPETITION,
                f"20{SEASON}",
                None,
                None,
                kickoff.isoformat(),
                "1",
                "2",
                None,
                None,
                None,  # scheduled fixtures carry no player-team field
                0,
                0,
            )
        )
    return rows


def _mv_rows() -> list[tuple]:
    base_day = (SEASON_START.date() - datetime(1970, 1, 1).date()).days
    rows = []
    for pid, _team, pos in SQUAD:
        for d in range(-60, 90):
            rows.append((pid, base_day + d, 1_000_000.0 * pos * (1 + d / 500.0)))
    return rows


@pytest.fixture(autouse=True)
def _guard_the_real_archive():
    """Fail any test that writes into the repository's committed archive.

    `data/snapshots/` is real, committed data. Several entry points default
    to it, so a test that forgets to pass `snapshot_dir=tmp_path/...` will
    silently write fake players into it -- which is exactly how `p1` and
    `p2` once ended up in a released snapshot.
    """
    real = Path(DEFAULT_SNAPSHOT_DIR)
    before = {p: p.stat().st_mtime_ns for p in real.glob("*")} if real.exists() else {}
    yield
    after = {p: p.stat().st_mtime_ns for p in real.glob("*")} if real.exists() else {}
    assert after == before, (
        f"test touched the committed archive at {real}; "
        "pass an explicit snapshot_dir under tmp_path"
    )


@pytest.fixture
def conn(tmp_path) -> sqlite3.Connection:
    connection = db.connect(tmp_path / "test.sqlite")
    now = NOW.isoformat()
    db.upsert_teams(connection, [(tid, name, now) for tid, name in TEAMS])
    db.upsert_players(
        connection,
        [
            (pid, "First", f"Last{pid}", team, pos, 0, 1_000_000 * pos, 100, 1000, 3, now)
            for pid, team, pos in SQUAD
        ],
    )
    db.upsert_performances(connection, _perf_rows())
    db.upsert_market_values(connection, _mv_rows())
    connection.commit()
    yield connection
    connection.close()
