"""Actual-points rankings for the running season.

Pure bookkeeping over the `performances` table the fetch already fills: no
model and no extra API calls. Two scopes for every matchday that has kicked
off:

- `matchday` -- points scored on that matchday alone
- `season`   -- points summed from matchday 1 through that matchday

Each is ranked across the whole league and within every position.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

import pandas as pd

from . import db
from .config import DEFAULT_COMPETITION_NAME, POSITIONS

TOP_N = 100

# A matchday counts as complete once its last match kicked off this long ago.
# Kickbase may still correct points days later; the nightly rebuild picks
# those up, so "complete" only means no match is still live or scheduled.
MATCH_WINDOW = timedelta(hours=3)


@dataclass
class MatchdayStatus:
    matchday: int
    complete: bool
    matches_played: int
    matches_total: int


@dataclass
class SeasonRankings:
    season_id: str
    matchdays: list[MatchdayStatus]
    # matchday -> one row per player who scored, unranked and uncut
    per_matchday: dict[int, pd.DataFrame] = field(default_factory=dict)
    cumulative: dict[int, pd.DataFrame] = field(default_factory=dict)

    @property
    def latest(self) -> MatchdayStatus | None:
        return self.matchdays[-1] if self.matchdays else None


def load_season(
    conn: sqlite3.Connection,
    season_id: str,
    *,
    competition_name: str = DEFAULT_COMPETITION_NAME,
) -> pd.DataFrame:
    """One row per player per fixture of the season, played or not.

    Only players on a squad today are in the feed, so someone who left the
    league mid-season is missing from the matchdays they did play.
    """
    df = pd.read_sql_query(
        """
        SELECT pf.player_id, pf.matchday, pf.match_id, pf.points, pf.minutes, pf.kickoff,
               COALESCE(pf.player_team_id, p.team_id) AS team_id,
               p.team_id AS current_team_id,
               p.first_name, p.last_name, p.position
        FROM performances pf
        JOIN players p ON p.player_id = pf.player_id
        WHERE pf.season_id = ?
          AND pf.matchday IS NOT NULL AND pf.kickoff IS NOT NULL
          AND (pf.competition IS NULL OR pf.competition = ?)
        """,
        conn,
        params=(str(season_id), competition_name),
    )
    names = pd.read_sql_query("SELECT team_id, name FROM teams", conn).set_index("team_id")[
        "name"
    ]
    df["team_name"] = df["team_id"].map(names)
    df["current_team_name"] = df["current_team_id"].map(names)
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, format="ISO8601")
    df["matchday"] = df["matchday"].astype(int)
    df["position_label"] = df["position"].map(POSITIONS)
    return df


def matchday_status(df: pd.DataFrame, now: datetime) -> list[MatchdayStatus]:
    """Matchdays with at least one kicked-off match, oldest first."""
    fixtures = df.drop_duplicates("match_id")[["matchday", "match_id", "kickoff"]]
    out = []
    for md, group in fixtures.groupby("matchday"):
        played = int((group["kickoff"] <= now).sum())
        if played == 0:
            continue
        out.append(
            MatchdayStatus(
                matchday=int(md),
                complete=bool((group["kickoff"] + MATCH_WINDOW <= now).all()),
                matches_played=played,
                matches_total=len(group),
            )
        )
    return out


def matchday_table(df: pd.DataFrame, matchday: int, now: datetime) -> pd.DataFrame:
    """Everyone who scored on `matchday`, listed under the club they played for."""
    rows = df[(df["matchday"] == matchday) & (df["kickoff"] <= now) & df["points"].notna()]
    return rows[
        [
            "player_id",
            "first_name",
            "last_name",
            "position_label",
            "team_id",
            "team_name",
            "points",
            "minutes",
        ]
    ].reset_index(drop=True)


def season_table(df: pd.DataFrame, through: int, now: datetime) -> pd.DataFrame:
    """Season totals from matchday 1 through `through`.

    Listed under the player's current club: a total can span two clubs, and
    the current one is where a manager can still buy the player.
    """
    rows = df[(df["matchday"] <= through) & (df["kickoff"] <= now) & df["points"].notna()]
    return rows.groupby("player_id", as_index=False).agg(
        first_name=("first_name", "first"),
        last_name=("last_name", "first"),
        position_label=("position_label", "first"),
        team_id=("current_team_id", "first"),
        team_name=("current_team_name", "first"),
        points=("points", "sum"),
        minutes=("minutes", "sum"),
        appearances=("points", "size"),
    )


def ranked(table: pd.DataFrame, *, position: str | None = None, top: int = TOP_N) -> pd.DataFrame:
    """Competition ranking ("1224"), cut after rank `top`.

    Ties share a rank and are kept whole at the cut, so a list can run a few
    rows past `top`. Within a tie, rows fall back to player id so the output
    is byte-stable between runs.
    """
    if position is not None:
        table = table[table["position_label"] == position]
    table = table.sort_values(["points", "player_id"], ascending=[False, True]).copy()
    table["rank"] = table["points"].rank(method="min", ascending=False).astype(int)
    return table[table["rank"] <= top].reset_index(drop=True)


def build_rankings(
    conn: sqlite3.Connection,
    season_id: str,
    *,
    now: datetime | None = None,
    competition_name: str = DEFAULT_COMPETITION_NAME,
) -> SeasonRankings:
    """Rank every matchday that had kicked off as of `now`.

    `now` defaults to the last fetch rather than the wall clock: the history
    only knows points as of then, and a re-publish from a stale history
    (`skip_fetch`) must not present unfetched matchdays as finished and empty.
    """
    if now is None:
        last_fetch = db.get_meta(conn, "last_fetch_at")
        now = datetime.fromisoformat(last_fetch) if last_fetch else datetime.now(timezone.utc)
    df = load_season(conn, season_id, competition_name=competition_name)
    result = SeasonRankings(season_id=str(season_id), matchdays=matchday_status(df, now))
    for status in result.matchdays:
        md = status.matchday
        result.per_matchday[md] = matchday_table(df, md, now)
        result.cumulative[md] = season_table(df, md, now)
    return result
