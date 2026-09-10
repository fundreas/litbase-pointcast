"""Milestone 1 -- pull Kickbase v4 data into the committed SQLite history."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from . import archive, db
from .api import KickbaseClient
from .config import DEFAULT_COMPETITION_NAME, DEFAULT_SNAPSHOT_DIR

log = logging.getLogger(__name__)

_MINUTES_RE = re.compile(r"(\d+)")


def parse_minutes(raw: Any) -> int | None:
    """Kickbase reports minutes as strings like ``"90'"`` or ``"0'"``.

    Stoppage time pushes these past 90 (``"107'"``); that is kept as reported,
    it carries a real signal about who stayed on the pitch.
    """
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    m = _MINUTES_RE.search(str(raw))
    return int(m.group(1)) if m else None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _today() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def fetch_teams(client: KickbaseClient, conn) -> list[dict[str, Any]]:
    table = client.table()
    now = _now()
    db.upsert_teams(conn, [(t["tid"], t.get("tn"), now) for t in table])
    log.info("teams: %d", len(table))
    return table


def fetch_squads(client: KickbaseClient, conn, teams: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Team profiles give the full player universe plus today's status and MV."""
    now = _now()
    today = _today()
    players: list[dict[str, Any]] = []
    for team in teams:
        profile = client.team_profile(team["tid"])
        for p in profile.get("it", []):
            p.setdefault("tid", team["tid"])
            players.append(p)

    db.upsert_players(
        conn,
        [
            (
                p["i"],
                None,  # first name only comes from the player detail endpoint
                p.get("n"),
                p.get("tid"),
                p.get("pos"),
                p.get("st"),
                p.get("mv"),
                p.get("ap"),
                p.get("tp"),
                p.get("prob"),
                now,
            )
            for p in players
        ],
    )
    db.upsert_status_snapshots(
        conn,
        [
            (today, p["i"], p.get("st"), p.get("mv"), p.get("prob"), p.get("tid"))
            for p in players
        ],
    )
    log.info("players: %d", len(players))
    return players


def fetch_matchdays(client: KickbaseClient, conn, season_id: str) -> dict[str, Any]:
    payload = client.matchdays()
    rows = []
    for md in payload.get("it", []):
        for m in md.get("it", []):
            rows.append(
                (
                    m["mi"],
                    season_id,
                    m.get("day", md.get("day")),
                    m.get("dt"),
                    m.get("t1"),
                    m.get("t2"),
                    m.get("t1g"),
                    m.get("t2g"),
                    m.get("st"),
                )
            )
    db.upsert_matches(conn, rows)
    log.info("matches: %d (current matchday %s)", len(rows), payload.get("day"))
    return payload


def _performance_rows(player_id: str, seasons: list[dict[str, Any]]) -> list[tuple]:
    """Flatten the per-season history.

    Season ids are global, not per-competition: a player's 2. Bundesliga or
    MLS spells arrive in the same list with ids interleaved among the
    Bundesliga ones. `n` is the only thing telling them apart, so it is
    recorded per row and filtered on later.
    """
    rows = []
    for season in seasons:
        sid = season.get("sid")
        competition = season.get("n")
        title = season.get("ti")
        for e in season.get("ph", []):
            minutes = parse_minutes(e.get("mp"))
            rows.append(
                (
                    player_id,
                    sid,
                    e.get("day"),
                    e.get("mi"),
                    competition,
                    title,
                    e.get("p"),
                    minutes,
                    e.get("md"),
                    e.get("t1"),
                    e.get("t2"),
                    e.get("t1g"),
                    e.get("t2g"),
                    e.get("pt"),
                    e.get("st"),
                    e.get("mdst"),
                )
            )
    return rows


def fetch_performances(
    client: KickbaseClient,
    conn,
    players: list[dict[str, Any]],
    *,
    competition_name: str = DEFAULT_COMPETITION_NAME,
    progress_every: int = 50,
) -> str | None:
    """Per-player history. Returns the id of the newest season seen.

    The endpoint replays every season the player ever appeared in, so a cold
    start needs no special backfill path -- the first run *is* the backfill.
    Only seasons of `competition_name` count toward "newest": season ids run
    across competitions, so a promoted player's 2. Bundesliga season can
    carry a higher id than the running Bundesliga one.
    """
    latest_season: str | None = None
    latest_key = -1
    total = 0
    for i, p in enumerate(players, 1):
        seasons = client.player_performance(p["i"])
        rows = _performance_rows(p["i"], seasons)
        total += db.upsert_performances(conn, rows)
        for season in seasons:
            sid = season.get("sid")
            if season.get("n") != competition_name or sid is None:
                continue
            if int(sid) > latest_key:
                latest_key, latest_season = int(sid), sid
        if i % progress_every == 0:
            conn.commit()
            log.info("performance: %d/%d players (%d rows)", i, len(players), total)
    conn.commit()
    log.info("performance: %d players, %d rows", len(players), total)
    return latest_season


def fetch_market_values(
    client: KickbaseClient, conn, players: list[dict[str, Any]], *, timeframe: int = 365,
    progress_every: int = 50,
) -> int:
    total = 0
    for i, p in enumerate(players, 1):
        series = client.player_market_value(p["i"], timeframe)
        total += db.upsert_market_values(
            conn, [(p["i"], e["dt"], e["mv"]) for e in series if e.get("dt") is not None]
        )
        if i % progress_every == 0:
            conn.commit()
            log.info("market values: %d/%d players (%d points)", i, len(players), total)
    conn.commit()
    log.info("market values: %d players, %d points", len(players), total)
    return total


def run_fetch(
    client: KickbaseClient,
    conn,
    *,
    competition_name: str = DEFAULT_COMPETITION_NAME,
    snapshot_dir: Path | str | None = DEFAULT_SNAPSHOT_DIR,
    skip_performances: bool = False,
    skip_market_values: bool = False,
    player_limit: int | None = None,
) -> dict[str, Any]:
    """Full data-layer refresh. Idempotent: everything is an upsert."""
    if snapshot_dir is not None:
        archive.load_snapshots(conn, snapshot_dir)

    teams = fetch_teams(client, conn)
    players = fetch_squads(client, conn, teams)
    if player_limit:
        players = players[:player_limit]

    latest_season = None
    if not skip_performances:
        latest_season = fetch_performances(
            client, conn, players, competition_name=competition_name
        )
    if latest_season is None:
        latest_season = db.get_meta(conn, "current_season_id") or "0"

    payload = fetch_matchdays(client, conn, latest_season)

    if not skip_market_values:
        fetch_market_values(client, conn, players)

    snapshot_file = None
    if snapshot_dir is not None:
        snapshot_file = archive.write_snapshot(conn, snapshot_dir)

    db.set_meta(conn, "current_season_id", str(latest_season))
    db.set_meta(conn, "current_matchday", str(payload.get("day", "")))
    db.set_meta(conn, "last_fetch_at", _now())
    db.vacuum(conn)

    counts = db.table_counts(conn)
    log.info("fetch complete: %s", counts)
    return {
        "season_id": latest_season,
        "current_matchday": payload.get("day"),
        "counts": counts,
        "snapshot": str(snapshot_file) if snapshot_file else None,
    }
