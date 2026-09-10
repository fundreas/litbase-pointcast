"""The durable half of the history: one small append-only file per day.

The plan calls for committing the SQLite file so history survives ephemeral
Action runners. The intent is right, the mechanism does not scale: the
database is ~16 MB (4 MB gzipped) and a nightly commit of a rewritten binary
blob adds that much to the repository *every night*, because git cannot delta
a re-VACUUMed or re-compressed file. A year of that is over a gigabyte.

So split the history by whether the API can re-serve it:

* **Re-derivable** -- performances, fixtures, squads. The performance endpoint
  replays a player's entire career on every call, so `data/history.sqlite` is
  a pure cache. It is not committed.
* **Perishable** -- what a player's status and market value were *on a given
  day*. The API only ever answers "right now", and market-value history is
  capped at 365 days. Miss a day and it is gone for good.

The perishable part is exactly one row per player per day, so it goes into
`data/snapshots/YYYY-MM-DD.csv.gz`: ~10 KB, written once, never rewritten.
Git stores each file a single time, the archive is human-readable, and a year
costs a few megabytes instead of a few gigabytes.

Over time these snapshots also grow past the API's own 365-day market-value
window, which makes the feed's history *better* than what Kickbase exposes.
"""

from __future__ import annotations

import csv
import gzip
import logging
import sqlite3
from datetime import date, datetime, timezone
from pathlib import Path

from . import db

log = logging.getLogger(__name__)

FIELDS = ("player_id", "team_id", "status", "market_value", "lineup_prob")
EPOCH = date(1970, 1, 1)


def snapshot_dir(root: Path | str) -> Path:
    return Path(root)


def snapshot_path(root: Path | str, day: date) -> Path:
    return Path(root) / f"{day.isoformat()}.csv.gz"


def day_number(day: date) -> int:
    """Days since the Unix epoch -- Kickbase's own market-value index."""
    return (day - EPOCH).days


def write_snapshot(conn: sqlite3.Connection, root: Path | str, day: date | None = None) -> Path:
    """Freeze one day's status and market value for every known player."""
    day = day or datetime.now(timezone.utc).date()
    rows = conn.execute(
        "SELECT player_id, team_id, status, market_value, lineup_prob"
        " FROM status_snapshots WHERE snapshot_date = ? ORDER BY player_id",
        (day.isoformat(),),
    ).fetchall()
    path = snapshot_path(root, day)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(FIELDS)
        for row in rows:
            writer.writerow([row[f] if row[f] is not None else "" for f in FIELDS])
    log.info("snapshot %s: %d players -> %s", day, len(rows), path)
    return path


def load_snapshots(conn: sqlite3.Connection, root: Path | str) -> int:
    """Replay every committed snapshot into the working database.

    Market values land in `market_values` alongside the API's own 365-day
    window; the snapshots are what extends that window backwards as the
    archive ages.
    """
    root = Path(root)
    if not root.exists():
        return 0
    status_rows: list[tuple] = []
    mv_rows: list[tuple] = []
    loaded = 0
    for path in sorted(root.glob("*.csv.gz")):
        try:
            day = date.fromisoformat(path.name.removesuffix(".csv.gz"))
        except ValueError:
            log.warning("skipping unrecognised snapshot file %s", path.name)
            continue
        loaded += 1
        dnum = day_number(day)
        with gzip.open(path, "rt", newline="", encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                pid = rec.get("player_id")
                if not pid:
                    continue
                status = _int_or_none(rec.get("status"))
                mv = _int_or_none(rec.get("market_value"))
                status_rows.append(
                    (
                        day.isoformat(),
                        pid,
                        status,
                        mv,
                        _int_or_none(rec.get("lineup_prob")),
                        rec.get("team_id") or None,
                    )
                )
                if mv is not None:
                    mv_rows.append((pid, dnum, float(mv)))

    # Snapshots are the authority for their own day, but the API's own
    # market-value series is finer-grained, so let a later API fetch win by
    # inserting rather than replacing where a value already exists.
    db.upsert_status_snapshots(conn, status_rows)
    conn.executemany(
        "INSERT OR IGNORE INTO market_values (player_id, day, value) VALUES (?, ?, ?)",
        mv_rows,
    )
    conn.commit()
    if loaded:
        log.info(
            "archive: replayed %d snapshot files (%d status rows, %d market values)",
            loaded,
            len(status_rows),
            len(mv_rows),
        )
    return loaded


def _int_or_none(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None
