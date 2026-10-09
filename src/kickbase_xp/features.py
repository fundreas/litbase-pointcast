"""Milestone 2 -- build the feature matrix from the SQLite history.

Every feature is computed from information available *strictly before* the
kickoff of the row it describes. The single mechanism enforcing that is a
per-player ``shift(1)`` over a chronologically sorted frame: nothing about a
match can leak into its own features. `tests/test_features.py` pins it.

The frame covers finished and scheduled matchdays alike. Finished rows are the
training set, the scheduled rows of the target matchday are what we predict --
one code path, so training and inference cannot drift apart.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .config import (
    DEFAULT_COMPETITION_NAME,
    PERF_STATUS_BENCH,
    PERF_STATUS_SCHEDULED,
    PERF_STATUS_STARTED,
    PERF_STATUS_SUB,
)

log = logging.getLogger(__name__)

STARTER_MINUTES = 60

# Windows used for the rolling form/playing-time features.
SHORT_WINDOW = 3
MEDIUM_WINDOW = 5
LONG_WINDOW = 10

# Shrinkage strength for team/opponent strength estimates: how many matchdays
# of own evidence it takes to outweigh the prior by half.
STRENGTH_PRIOR_WEIGHT = 3.0

# Seasons of history to train on by default. See `build_matrix`.
DEFAULT_MAX_SEASONS = 4

FEATURE_COLUMNS = [
    # form
    "pts_mean_3",
    "pts_mean_5",
    "pts_mean_10",
    "pts_median_3",
    "pts_median_5",
    "pts_std_5",
    "pts_std_10",
    "pts_last",
    "pts_app_mean_3",
    "pts_app_mean_5",
    "season_pts_mean",
    "career_app_pts_mean",
    # playing time
    "min_mean_3",
    "min_mean_5",
    "min_last",
    "play_share_5",
    "play_share_10",
    "start_share_5",
    "start_share_10",
    "squad_share_5",
    "started_last",
    "played_last",
    "squad_last",
    "start_streak",
    "season_play_share",
    "season_start_share",
    "season_games",
    "career_games",
    "days_since_last_match",
    # role
    "position",
    "matchday",
    # competition for places inside the team
    "start_rank_pos",
    "team_pos_starters_5",
    "start_rank_vs_slots",
    "team_rotation_5",
    # market
    "log_mv",
    "mv_trend_7",
    "mv_trend_30",
    # fixture
    "is_home",
    "opp_allowed_pos",
    "opp_allowed_all",
    "team_scored_pos",
    "team_scored_all",
    "opp_scored_all",
]

CATEGORICAL_COLUMNS = ["position"]

# Columns carried alongside the features for bookkeeping and publishing.
ID_COLUMNS = [
    "player_id",
    "season_id",
    "matchday",
    "match_id",
    "kickoff",
    "team_id",
    "opponent_team_id",
    "completed",
    "points",
    "minutes",
    "played",
    "started",
    "in_squad",
    "perf_status",
]


# --------------------------------------------------------------------- load


def load_raw(
    conn: sqlite3.Connection, competition_name: str = DEFAULT_COMPETITION_NAME
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the history for one competition.

    The competition filter is not optional housekeeping: a player promoted
    from the 2. Bundesliga carries that league's matchdays under season ids
    interleaved with the Bundesliga ones, and its points are not on the same
    scale. Rows written before the `competition` column existed are kept
    (NULL passes the filter) so an older history file still works.
    """
    perf = pd.read_sql_query(
        """
        SELECT player_id, season_id, matchday, match_id, points, minutes, kickoff,
               home_team_id, away_team_id, player_team_id, perf_status
        FROM performances
        WHERE matchday IS NOT NULL AND kickoff IS NOT NULL
          AND (competition IS NULL OR competition = ?)
        """,
        conn,
        params=(competition_name,),
    )
    players = pd.read_sql_query(
        "SELECT player_id, team_id, position, status, market_value, lineup_prob,"
        " first_name, last_name FROM players",
        conn,
    )
    mv = pd.read_sql_query(
        "SELECT player_id, day, value FROM market_values ORDER BY player_id, day", conn
    )
    return perf, players, mv


# ----------------------------------------------------------------- assemble


def _prepare_base(perf: pd.DataFrame, players: pd.DataFrame, now: datetime) -> pd.DataFrame:
    df = perf.merge(
        players[["player_id", "team_id", "position"]],
        on="player_id",
        how="inner",
        suffixes=("", "_current"),
    )
    df["kickoff"] = pd.to_datetime(df["kickoff"], utc=True, format="ISO8601")

    # Scheduled fixtures carry no `pt`; fall back to the player's current club.
    df["team_id"] = df["player_team_id"].fillna(df["team_id"])
    df = df[df["team_id"].notna()]

    df["is_home"] = (df["team_id"] == df["home_team_id"]).astype("int8")
    df["opponent_team_id"] = np.where(
        df["is_home"] == 1, df["away_team_id"], df["home_team_id"]
    )

    df["completed"] = df["kickoff"] < now
    df = _add_labels(df)
    # Points conditional on appearing -- NaN when the player did not appear, so
    # rolling means over it describe form *given playing*.
    df["points_app"] = np.where(df["played"] == 1.0, df["points"], np.nan)

    df["season_order"] = pd.to_numeric(df["season_id"], errors="coerce").fillna(-1)
    df = df.sort_values(["player_id", "kickoff", "matchday"]).reset_index(drop=True)
    return df


def _add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """`started` / `played` / `in_squad`, read off the feed's per-match status.

    The feed's `st` is the real lineup (see `config.PERF_STATUS_*`); minutes
    only back it up -- `>= 60` was the old starter proxy and misses every
    starter subbed off early. A kicked-off row still carrying `st == 0` and no
    minutes is a fixture the history has not caught up with (a stale fetch):
    its labels and points are *unknown*, not "did not play".
    """
    st = df["perf_status"] if "perf_status" in df else pd.Series(np.nan, index=df.index)
    has_status = st.notna() & (st != PERF_STATUS_SCHEDULED)
    known = df["completed"] & (has_status | df["minutes"].notna())

    # A known finished match with no minutes entry: the player was not involved.
    df["minutes"] = df["minutes"].where(df["completed"], other=np.nan)
    df.loc[known & df["minutes"].isna(), "minutes"] = 0.0
    df.loc[~known, "minutes"] = np.nan
    df["points"] = df["points"].where(known, other=np.nan)
    df.loc[known & df["points"].isna(), "points"] = 0.0

    mins = df["minutes"]
    on_pitch = mins > 0
    started = np.where(has_status, st == PERF_STATUS_STARTED, mins >= STARTER_MINUTES)
    played = np.where(has_status, st.isin([PERF_STATUS_STARTED, PERF_STATUS_SUB]) | on_pitch, on_pitch)
    in_squad = np.where(
        has_status,
        st.isin([PERF_STATUS_STARTED, PERF_STATUS_SUB, PERF_STATUS_BENCH]) | on_pitch,
        on_pitch,
    )
    for name, value in (("started", started), ("played", played), ("in_squad", in_squad)):
        df[name] = np.where(known, value.astype(float), np.nan)
    return df


# ------------------------------------------------------------ player window


def _add_player_history(df: pd.DataFrame) -> pd.DataFrame:
    grp = df.groupby("player_id", sort=False)

    def roll(col: str, window: int, how: str) -> pd.Series:
        """Trailing statistic over the *previous* `window` rows of each player."""
        shifted = grp[col].shift(1)
        return (
            shifted.groupby(df["player_id"])
            .rolling(window, min_periods=1)
            .agg(how)
            .reset_index(level=0, drop=True)
        )

    df["pts_mean_3"] = roll("points", SHORT_WINDOW, "mean")
    df["pts_mean_5"] = roll("points", MEDIUM_WINDOW, "mean")
    df["pts_mean_10"] = roll("points", LONG_WINDOW, "mean")
    df["pts_median_3"] = roll("points", SHORT_WINDOW, "median")
    df["pts_median_5"] = roll("points", MEDIUM_WINDOW, "median")
    df["pts_std_5"] = roll("points", MEDIUM_WINDOW, "std")
    df["pts_std_10"] = roll("points", LONG_WINDOW, "std")
    df["min_mean_3"] = roll("minutes", SHORT_WINDOW, "mean")
    df["min_mean_5"] = roll("minutes", MEDIUM_WINDOW, "mean")
    df["play_share_5"] = roll("played", MEDIUM_WINDOW, "mean")
    df["play_share_10"] = roll("played", LONG_WINDOW, "mean")
    df["start_share_5"] = roll("started", MEDIUM_WINDOW, "mean")
    df["start_share_10"] = roll("started", LONG_WINDOW, "mean")
    df["squad_share_5"] = roll("in_squad", MEDIUM_WINDOW, "mean")
    df["pts_last"] = grp["points"].shift(1)
    df["min_last"] = grp["minutes"].shift(1)
    # Last match's role. P(start | started last) is ~0.8 against ~0.2
    # otherwise, the single strongest lineup signal in the history.
    df["started_last"] = grp["started"].shift(1)
    df["played_last"] = grp["played"].shift(1)
    df["squad_last"] = grp["in_squad"].shift(1)

    # Consecutive starts strictly before this row. An unknown row breaks the
    # run, which errs towards "less nailed".
    prev_start = grp["started"].shift(1)
    is_start = prev_start == 1.0
    block = (~is_start).groupby(df["player_id"]).cumsum()
    df["start_streak"] = is_start.astype(float).groupby([df["player_id"], block]).cumsum()

    # Form over the last N appearances (ignores matchdays spent off the pitch).
    for window, name in ((SHORT_WINDOW, "pts_app_mean_3"), (MEDIUM_WINDOW, "pts_app_mean_5")):
        app = df["points_app"]
        tmp = (
            app.groupby(df["player_id"])
            .apply(lambda s: s.dropna().rolling(window, min_periods=1).mean())
            .reset_index(level=0, drop=True)
        )
        tmp = tmp.reindex(df.index)
        df[name] = tmp.groupby(df["player_id"]).shift(1).groupby(df["player_id"]).ffill()

    # Career-to-date (expanding, all seasons).
    df["career_games"] = (
        df["played"].groupby(df["player_id"]).shift(1).groupby(df["player_id"]).cumsum()
    )
    df["career_app_pts_mean"] = (
        df["points_app"]
        .groupby(df["player_id"])
        .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
        .reset_index(level=0, drop=True)
        .reindex(df.index)
    )

    # Season-to-date (expanding, resets each season).
    skey = [df["player_id"], df["season_id"]]
    for col, name in (
        ("points", "season_pts_mean"),
        ("played", "season_play_share"),
        ("started", "season_start_share"),
    ):
        df[name] = (
            df[col]
            .groupby(skey)
            .apply(lambda s: s.shift(1).expanding(min_periods=1).mean())
            .reset_index(level=[0, 1], drop=True)
            .reindex(df.index)
        )
    df["season_games"] = (
        df["played"].groupby(skey).shift(1).groupby(skey).cumsum().reindex(df.index)
    )

    prev_kickoff = grp["kickoff"].shift(1)
    df["days_since_last_match"] = (df["kickoff"] - prev_kickoff).dt.total_seconds() / 86400.0
    return df


# ------------------------------------------------------------ team context


def _asof_team(
    df: pd.DataFrame, right: pd.DataFrame, by: list[str], cols: list[str]
) -> pd.DataFrame:
    """Attach the latest team tally from a *strictly earlier* kickoff.

    `allow_exact_matches=False` is the leak guard: a row never sees the tally
    of its own matchday, scheduled rows get the latest finished one for free.
    """
    left = df[["kickoff", *by]].copy()
    left["_row"] = np.arange(len(left))
    left = left.sort_values("kickoff", kind="mergesort")
    right = right.sort_values("kickoff", kind="mergesort")
    merged = pd.merge_asof(
        left,
        right[["kickoff", *by, *cols]],
        on="kickoff",
        by=by,
        direction="backward",
        allow_exact_matches=False,
    )
    return merged.set_index("_row")[cols].sort_index().set_axis(df.index)


def team_games(df: pd.DataFrame) -> pd.DataFrame:
    """One row per finished (team, season, matchday) with a known lineup.

    Carries the kickoff, the starting XI as a frozenset of player ids and the
    number of starters per position. `lineups.py` reads formations off this.
    """
    done = df[df["completed"] & df["started"].notna()]
    if done.empty:
        return pd.DataFrame(
            columns=["team_id", "season_id", "matchday", "kickoff", "xi", 1, 2, 3, 4]
        )
    keys = ["team_id", "season_id", "matchday"]
    games = done.groupby(keys, as_index=False)["kickoff"].min()
    starters = done[(done["started"] == 1.0) & done["position"].notna()].copy()
    starters["position"] = starters["position"].astype(int)
    xi = starters.groupby(keys)["player_id"].agg(frozenset).rename("xi").reset_index()
    counts = (
        starters.groupby(keys + ["position"]).size().unstack("position", fill_value=0)
        .reindex(columns=[1, 2, 3, 4], fill_value=0)
        .reset_index()
    )
    counts.columns.name = None
    out = games.merge(xi, on=keys, how="left").merge(counts, on=keys, how="left")
    out["xi"] = out["xi"].apply(lambda s: s if isinstance(s, frozenset) else frozenset())
    for pos in (1, 2, 3, 4):
        out[pos] = out[pos].fillna(0).astype(int)
    return out.sort_values(["team_id", "kickoff"]).reset_index(drop=True)


def _add_team_context(df: pd.DataFrame) -> pd.DataFrame:
    """Where a player stands against his teammates at the same position.

    Every input here is already a shifted, pre-kickoff quantity or a tally of
    strictly earlier matchdays, so the cross-player view leaks nothing.
    """
    pos_key = df["position"].astype("float").fillna(-1.0)
    group = [df["season_id"], df["matchday"], df["team_id"], pos_key]
    df["start_rank_pos"] = (
        df["start_share_5"].fillna(-1.0).groupby(group).rank(method="min", ascending=False)
    )

    games = team_games(df)
    if games.empty:
        for col in ("team_pos_starters_5", "start_rank_vs_slots", "team_rotation_5"):
            df[col] = np.nan
        return df

    # Starters per position, rolled over the team's last five games.
    long = games.melt(
        id_vars=["team_id", "kickoff"], value_vars=[1, 2, 3, 4],
        var_name="position", value_name="n",
    )
    long["position"] = long["position"].astype(float)
    long = long.sort_values(["team_id", "position", "kickoff"])
    long["team_pos_starters_5"] = long.groupby(["team_id", "position"])["n"].transform(
        lambda s: s.rolling(MEDIUM_WINDOW, min_periods=1).mean()
    )
    df["_pos_key"] = pos_key
    df["team_pos_starters_5"] = _asof_team(
        df, long.rename(columns={"position": "_pos_key"}), ["team_id", "_pos_key"],
        ["team_pos_starters_5"],
    )["team_pos_starters_5"]
    df = df.drop(columns=["_pos_key"])
    df["start_rank_vs_slots"] = df["start_rank_pos"] - df["team_pos_starters_5"]

    # Rotation: share of the XI that is new against the team's previous game.
    def churn(g: pd.DataFrame) -> pd.Series:
        prev = g["xi"].shift(1)
        vals = [
            (1.0 - len(cur & p) / len(cur)) if isinstance(p, frozenset) and cur else np.nan
            for cur, p in zip(g["xi"], prev)
        ]
        return pd.Series(vals, index=g.index)

    games["churn"] = pd.concat([churn(g) for _, g in games.groupby("team_id", sort=False)])
    games["team_rotation_5"] = games.groupby("team_id")["churn"].transform(
        lambda s: s.rolling(MEDIUM_WINDOW, min_periods=1).mean()
    )
    df["team_rotation_5"] = _asof_team(df, games, ["team_id"], ["team_rotation_5"])[
        "team_rotation_5"
    ]
    return df


# ---------------------------------------------------------- team strength


def _league_priors(played: pd.DataFrame) -> pd.DataFrame:
    """League-wide mean points per position, from *previous seasons only*.

    A single mean over the whole dataset would be a quiet leak: matchday 34's
    results would nudge the prior that matchday 1's features are shrunk
    toward. Accumulating season by season and shifting keeps it honest, at the
    cost of a NaN prior in the very first season on record.
    """
    per_season = (
        played.groupby(["position", "season_order"])["points"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values(["position", "season_order"])
    )
    g = per_season.groupby("position", sort=False)
    prior_sum = g["sum"].cumsum() - per_season["sum"]
    prior_cnt = g["count"].cumsum() - per_season["count"]
    per_season["league_pos_prior"] = np.where(
        prior_cnt > 0, prior_sum / prior_cnt.replace(0, np.nan), np.nan
    )

    overall = (
        played.groupby("season_order")["points"]
        .agg(["sum", "count"])
        .reset_index()
        .sort_values("season_order")
    )
    o_sum = overall["sum"].cumsum() - overall["sum"]
    o_cnt = overall["count"].cumsum() - overall["count"]
    overall["league_all_prior"] = np.where(
        o_cnt > 0, o_sum / o_cnt.replace(0, np.nan), np.nan
    )
    return per_season[["position", "season_order", "league_pos_prior"]].merge(
        overall[["season_order", "league_all_prior"]], on="season_order", how="left"
    )


def _todate(agg: pd.DataFrame, pos_col: str, all_col: str, priors: pd.DataFrame) -> pd.DataFrame:
    """Season-to-date mean of a per-matchday team/position tally.

    Expanding within the season and shifted by one matchday (so a team's own
    result never describes its own fixture), then shrunk toward a prior: the
    same team's value last season, falling back to the league's previous
    seasons. Without the prior, matchday 1 would be pure noise and matchday 2
    a single observation.
    """
    keys = ["team_id", "position"]
    agg = agg.sort_values(keys + ["season_order", "matchday"]).reset_index(drop=True)

    # Prior 1: this team at this position, last season.
    season_level = (
        agg.groupby(keys + ["season_order"], as_index=False)[[pos_col, all_col]]
        .mean()
        .sort_values(keys + ["season_order"])
    )
    for col in (pos_col, all_col):
        season_level[f"prior_{col}"] = season_level.groupby(keys, sort=False)[col].shift(1)
    agg = agg.merge(
        season_level[keys + ["season_order", f"prior_{pos_col}", f"prior_{all_col}"]],
        on=keys + ["season_order"],
        how="left",
    )
    # Prior 2: the league at this position, in seasons already played out.
    agg = agg.merge(priors, on=["position", "season_order"], how="left")
    agg[f"prior_{pos_col}"] = agg[f"prior_{pos_col}"].fillna(agg["league_pos_prior"])
    agg[f"prior_{all_col}"] = agg[f"prior_{all_col}"].fillna(agg["league_all_prior"])

    grp = agg.groupby(keys + ["season_order"], sort=False)
    for col in (pos_col, all_col):
        shifted = grp[col].shift(1)
        run = shifted.groupby([agg["team_id"], agg["position"], agg["season_order"]])
        run_sum = run.expanding(min_periods=1).sum().reset_index(level=[0, 1, 2], drop=True)
        run_cnt = run.expanding(min_periods=1).count().reset_index(level=[0, 1, 2], drop=True)
        run_sum = run_sum.reindex(agg.index).fillna(0.0)
        run_cnt = run_cnt.reindex(agg.index).fillna(0.0)
        prior = agg[f"prior_{col}"]
        shrunk = (run_sum + STRENGTH_PRIOR_WEIGHT * prior) / (run_cnt + STRENGTH_PRIOR_WEIGHT)
        # No prior available (the earliest season on record): fall back to the
        # unshrunk running mean, which is simply undefined before matchday 2.
        plain = run_sum.divide(run_cnt.replace(0.0, np.nan))
        agg[f"{col}_todate"] = shrunk.where(prior.notna(), plain)
    return agg.drop(columns=["league_pos_prior", "league_all_prior"])


def _team_position_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Attach team/opponent strength, measured in Kickbase points only.

    "Points conceded to forwards" needs no xG feed and no league table: it is
    literally the average score of the *opposing* team's forwards in that
    team's past fixtures. Same data, better aligned to the target.
    """
    strength_cols = [
        "team_scored_pos",
        "team_scored_all",
        "opp_allowed_pos",
        "opp_allowed_all",
        "opp_scored_all",
    ]
    played = df[(df["completed"]) & (df["played"] == 1.0)]
    if played.empty:
        return df.assign(**dict.fromkeys(strength_cols, np.nan))

    priors = _league_priors(played)

    def tally(team_col: str, pos_name: str, all_name: str) -> pd.DataFrame:
        """Mean points by (season, matchday, `team_col`, position) and overall."""
        by_pos = (
            played.groupby(
                ["season_order", "season_id", "matchday", team_col, "position"], as_index=False
            )["points"]
            .mean()
            .rename(columns={"points": pos_name, team_col: "team_id"})
        )
        by_team = (
            played.groupby(["season_order", "season_id", "matchday", team_col], as_index=False)[
                "points"
            ]
            .mean()
            .rename(columns={"points": all_name, team_col: "team_id"})
        )
        merged = by_pos.merge(
            by_team, on=["season_order", "season_id", "matchday", "team_id"], how="left"
        )
        return _todate(merged, pos_name, all_name, priors)

    # What a team's players produce...
    scored = tally("team_id", "scored", "scored_all")
    # ...and what its opponents produce against it, i.e. what it concedes.
    conceded = tally("opponent_team_id", "conceded", "conceded_all")

    out = df.merge(
        scored[
            ["season_id", "matchday", "team_id", "position", "scored_todate", "scored_all_todate"]
        ].rename(
            columns={"scored_todate": "team_scored_pos", "scored_all_todate": "team_scored_all"}
        ),
        on=["season_id", "matchday", "team_id", "position"],
        how="left",
    )
    out = out.merge(
        conceded[
            [
                "season_id",
                "matchday",
                "team_id",
                "position",
                "conceded_todate",
                "conceded_all_todate",
            ]
        ].rename(
            columns={
                "team_id": "opponent_team_id",
                "conceded_todate": "opp_allowed_pos",
                "conceded_all_todate": "opp_allowed_all",
            }
        ),
        on=["season_id", "matchday", "opponent_team_id", "position"],
        how="left",
    )
    out = out.merge(
        scored[["season_id", "matchday", "team_id", "scored_all_todate"]]
        .drop_duplicates(subset=["season_id", "matchday", "team_id"])
        .rename(
            columns={"team_id": "opponent_team_id", "scored_all_todate": "opp_scored_all"}
        ),
        on=["season_id", "matchday", "opponent_team_id"],
        how="left",
    )

    # Scheduled fixtures have no tally of their own, and a team that fielded
    # nobody at some position in a past matchday has a gap: carry the most
    # recent season-to-date estimate forward in both cases.
    out = out.sort_values(["team_id", "position", "kickoff"])
    for col in ("team_scored_pos", "team_scored_all"):
        out[col] = out.groupby(["team_id", "position"], sort=False)[col].ffill()
    out = out.sort_values(["opponent_team_id", "position", "kickoff"])
    for col in ("opp_allowed_pos", "opp_allowed_all", "opp_scored_all"):
        out[col] = out.groupby(["opponent_team_id", "position"], sort=False)[col].ffill()
    return out.sort_values(["player_id", "kickoff"]).reset_index(drop=True)


# ------------------------------------------------------------ market value


def _add_market_value(df: pd.DataFrame, mv: pd.DataFrame) -> pd.DataFrame:
    # Kickbase indexes market values by days since the Unix epoch.
    df["kickoff_day"] = (
        (df["kickoff"] - pd.Timestamp("1970-01-01", tz="UTC")).dt.days.astype("int64")
    )
    if mv.empty:
        df["log_mv"] = np.nan
        df["mv_trend_7"] = np.nan
        df["mv_trend_30"] = np.nan
        return df

    mv = mv.dropna(subset=["day"]).copy()
    mv["day"] = mv["day"].astype("int64")
    mv = mv.sort_values(["day", "player_id"])

    def as_of(offset: int, name: str) -> pd.Series:
        left = df[["player_id", "kickoff_day"]].copy()
        # -1: use the value from the day *before* kickoff, never the same day.
        left["lookup_day"] = left["kickoff_day"] - 1 - offset
        # merge_asof needs a sorted key and hands back a fresh RangeIndex, so
        # carry the original positions through explicitly.
        left["_row"] = np.arange(len(left))
        left = left.sort_values("lookup_day", kind="mergesort")
        merged = pd.merge_asof(
            left,
            mv.rename(columns={"day": "lookup_day", "value": name}),
            on="lookup_day",
            by="player_id",
            direction="backward",
            tolerance=45,
        )
        return (
            merged.set_index("_row")[name].sort_index().set_axis(df.index)
        )

    df["_mv"] = as_of(0, "_mv")
    df["_mv7"] = as_of(7, "_mv7")
    df["_mv30"] = as_of(30, "_mv30")

    df["log_mv"] = np.log1p(df["_mv"])
    df["mv_trend_7"] = df["_mv"] / df["_mv7"] - 1.0
    df["mv_trend_30"] = df["_mv"] / df["_mv30"] - 1.0
    df.loc[~np.isfinite(df["mv_trend_7"]), "mv_trend_7"] = np.nan
    df.loc[~np.isfinite(df["mv_trend_30"]), "mv_trend_30"] = np.nan
    return df.drop(columns=["_mv7", "_mv30"])


# ------------------------------------------------------------------ public


def build_matrix(
    conn: sqlite3.Connection,
    *,
    now: datetime | None = None,
    min_season: int | None = None,
    max_seasons: int | None = DEFAULT_MAX_SEASONS,
    competition_name: str = DEFAULT_COMPETITION_NAME,
) -> pd.DataFrame:
    """Return one row per player-matchday with leak-free features.

    Parameters
    ----------
    min_season:
        Drop rows from seasons older than this Kickbase season id.
    max_seasons:
        Keep only the N most recent seasons present in the data. Older seasons
        are thin -- the feed only replays players who are *still* on a squad
        today, so 2015 looks like a league of 40 survivors -- and Kickbase has
        changed its scoring since. Rows are dropped after the features are
        computed, so the surviving rows keep their full history behind them.
    """
    now = now or datetime.now(timezone.utc)
    perf, players, mv = load_raw(conn, competition_name)
    if perf.empty:
        return pd.DataFrame(columns=ID_COLUMNS + FEATURE_COLUMNS)

    df = _prepare_base(perf, players, now)
    df = _add_player_history(df)
    df = _add_team_context(df)
    df = _team_position_strength(df)
    df = _add_market_value(df, mv)

    season_order = pd.to_numeric(df["season_id"], errors="coerce").fillna(-1)
    if min_season is not None:
        df = df[season_order >= min_season]
        season_order = season_order.loc[df.index]
    if max_seasons is not None and max_seasons > 0:
        keep = sorted(season_order.unique())[-max_seasons:]
        df = df[season_order.isin(keep)]

    df["position"] = df["position"].astype("float").astype("Int64")
    for col in FEATURE_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    keep = ID_COLUMNS + [c for c in FEATURE_COLUMNS if c not in ID_COLUMNS] + ["kickoff_day"]
    out = df[[c for c in dict.fromkeys(keep) if c in df.columns]].copy()
    log.info(
        "feature matrix: %d rows (%d completed) x %d features",
        len(out),
        int(out["completed"].sum()),
        len(FEATURE_COLUMNS),
    )
    return out.reset_index(drop=True)


def training_rows(matrix: pd.DataFrame, *, before: pd.Timestamp | None = None) -> pd.DataFrame:
    """Kicked-off rows whose outcome is known (see `_add_labels`)."""
    rows = matrix[matrix["completed"] & matrix["played"].notna()]
    if before is not None:
        rows = rows[rows["kickoff"] < before]
    return rows


def prediction_rows(matrix: pd.DataFrame, season_id: str, matchday: int) -> pd.DataFrame:
    return matrix[
        (matrix["season_id"].astype(str) == str(season_id))
        & (matrix["matchday"] == matchday)
    ].copy()
