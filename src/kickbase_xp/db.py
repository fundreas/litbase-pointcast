"""SQLite persistence for the committed history file.

The database is the pipeline's memory: the Kickbase performance endpoint
re-serves full history on every call, but market-value history is capped at
365 days and player status is only ever available as "right now". Persisting
nightly snapshots is what turns those into a time series.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

SCHEMA = """
PRAGMA journal_mode = WAL;

CREATE TABLE IF NOT EXISTS teams (
    team_id     TEXT PRIMARY KEY,
    name        TEXT,
    updated_at  TEXT
);

CREATE TABLE IF NOT EXISTS players (
    player_id     TEXT PRIMARY KEY,
    first_name    TEXT,
    last_name     TEXT,
    team_id       TEXT,
    position      INTEGER,
    status        INTEGER,
    market_value  INTEGER,
    avg_points    INTEGER,
    total_points  INTEGER,
    lineup_prob   INTEGER,   -- Kickbase/LigaInsider tier; stored, unused in v1
    updated_at    TEXT
);

-- One row per player per matchday the Kickbase feed knows about, all seasons.
-- `points` is NULL when the player did not take part.
-- `competition` matters: the feed replays a player's 2. Bundesliga and
-- foreign-league seasons here too, under their own season ids interleaved
-- with the Bundesliga ones. Training on them would mix scoring regimes.
CREATE TABLE IF NOT EXISTS performances (
    player_id      TEXT NOT NULL,
    season_id      TEXT NOT NULL,
    matchday       INTEGER NOT NULL,
    match_id       TEXT,
    competition    TEXT,
    season_title   TEXT,
    points         INTEGER,
    minutes        INTEGER,
    kickoff        TEXT,
    home_team_id   TEXT,
    away_team_id   TEXT,
    home_goals     INTEGER,
    away_goals     INTEGER,
    player_team_id TEXT,
    perf_status    INTEGER,
    md_status      INTEGER,
    PRIMARY KEY (player_id, season_id, matchday)
);
CREATE INDEX IF NOT EXISTS idx_perf_season_md ON performances (season_id, matchday);
CREATE INDEX IF NOT EXISTS idx_perf_player ON performances (player_id);

-- Fixtures of the running season, including scheduled ones.
CREATE TABLE IF NOT EXISTS matches (
    match_id     TEXT PRIMARY KEY,
    season_id    TEXT,
    matchday     INTEGER NOT NULL,
    kickoff      TEXT,
    home_team_id TEXT,
    away_team_id TEXT,
    home_goals   INTEGER,
    away_goals   INTEGER,
    status       INTEGER
);
CREATE INDEX IF NOT EXISTS idx_matches_md ON matches (matchday);

-- Market value time series. `day` = days since the Unix epoch (Kickbase's own
-- encoding), kept as-is so re-fetches are idempotent.
CREATE TABLE IF NOT EXISTS market_values (
    player_id TEXT NOT NULL,
    day       INTEGER NOT NULL,
    value     REAL,
    PRIMARY KEY (player_id, day)
);

-- Nightly status snapshot. Not used as a training feature in v1 (no history
-- on day one), but accumulating it is what makes that possible in v2.
CREATE TABLE IF NOT EXISTS status_snapshots (
    snapshot_date TEXT NOT NULL,
    player_id     TEXT NOT NULL,
    status        INTEGER,
    market_value  INTEGER,
    lineup_prob   INTEGER,
    team_id       TEXT,
    PRIMARY KEY (snapshot_date, player_id)
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


# Columns added after the first release. The history file is committed and
# outlives the code that wrote it, so widen it in place rather than asking
# for a full re-fetch.
MIGRATIONS: tuple[tuple[str, str, str], ...] = (
    ("performances", "competition", "TEXT"),
    ("performances", "season_title", "TEXT"),
)


# Indexes over migrated columns, created once the columns exist.
POST_MIGRATION_INDEXES = (
    "CREATE INDEX IF NOT EXISTS idx_perf_competition ON performances (competition)",
)


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, coltype in MIGRATIONS:
        existing = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")
    for statement in POST_MIGRATION_INDEXES:
        conn.execute(statement)
    conn.commit()


def connect(path: Path | str) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


def _upsert(
    conn: sqlite3.Connection,
    table: str,
    columns: Sequence[str],
    rows: Iterable[Sequence[Any]],
) -> int:
    rows = list(rows)
    if not rows:
        return 0
    placeholders = ", ".join("?" * len(columns))
    sql = (
        f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    )
    conn.executemany(sql, rows)
    return len(rows)


def upsert_teams(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(conn, "teams", ("team_id", "name", "updated_at"), rows)


def upsert_players(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(
        conn,
        "players",
        (
            "player_id",
            "first_name",
            "last_name",
            "team_id",
            "position",
            "status",
            "market_value",
            "avg_points",
            "total_points",
            "lineup_prob",
            "updated_at",
        ),
        rows,
    )


def upsert_performances(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(
        conn,
        "performances",
        (
            "player_id",
            "season_id",
            "matchday",
            "match_id",
            "competition",
            "season_title",
            "points",
            "minutes",
            "kickoff",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "player_team_id",
            "perf_status",
            "md_status",
        ),
        rows,
    )


def upsert_matches(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(
        conn,
        "matches",
        (
            "match_id",
            "season_id",
            "matchday",
            "kickoff",
            "home_team_id",
            "away_team_id",
            "home_goals",
            "away_goals",
            "status",
        ),
        rows,
    )


def upsert_market_values(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(conn, "market_values", ("player_id", "day", "value"), rows)


def upsert_status_snapshots(conn: sqlite3.Connection, rows: Iterable[Sequence[Any]]) -> int:
    return _upsert(
        conn,
        "status_snapshots",
        ("snapshot_date", "player_id", "status", "market_value", "lineup_prob", "team_id"),
        rows,
    )


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def get_meta(conn: sqlite3.Connection, key: str, default: str | None = None) -> str | None:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def vacuum(conn: sqlite3.Connection) -> None:
    """Keep the committed file small and its diffs deterministic."""
    conn.commit()
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("VACUUM")


def table_counts(conn: sqlite3.Connection) -> dict[str, int]:
    tables = (
        "teams",
        "players",
        "performances",
        "matches",
        "market_values",
        "status_snapshots",
    )
    return {
        t: conn.execute(f"SELECT COUNT(*) AS n FROM {t}").fetchone()["n"] for t in tables
    }
