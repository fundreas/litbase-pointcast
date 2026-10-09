"""Feature-matrix behaviour, above all: no information from the future."""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd
import pytest

from kickbase_xp import features
from kickbase_xp.fetch import parse_minutes

from .conftest import NOW, PLAYED_MATCHDAYS, PREV_SEASON, SCHEDULED_MATCHDAY, SEASON


@pytest.fixture
def matrix(conn) -> pd.DataFrame:
    return features.build_matrix(conn, now=NOW, max_seasons=None)


def test_parse_minutes_handles_kickbase_formats():
    assert parse_minutes("90'") == 90
    assert parse_minutes("0'") == 0
    assert parse_minutes("107'") == 107  # stoppage time is kept, not clipped
    assert parse_minutes(None) is None
    assert parse_minutes(45) == 45
    assert parse_minutes("") is None


def test_scheduled_matchday_is_present_and_incomplete(matrix):
    upcoming = matrix[matrix["matchday"] == SCHEDULED_MATCHDAY]
    assert len(upcoming) == 8
    assert not upcoming["completed"].any()
    assert upcoming["points"].isna().all()
    # The player's club is filled in from the roster when the fixture has none.
    assert upcoming["team_id"].notna().all()
    assert set(upcoming["opponent_team_id"]) == {"1", "2"}


def test_rolling_features_exclude_the_current_matchday(matrix):
    """The row for matchday N must not know matchday N's own points."""
    p1 = matrix[(matrix["player_id"] == "p1") & (matrix["season_id"] == SEASON)]
    p1 = p1.sort_values("matchday").set_index("matchday")
    # From matchday 4 on, the 3-match window sits entirely inside the season.
    for md in range(4, PLAYED_MATCHDAYS + 1):
        prior = p1.loc[md - 3 : md - 1, "points"]
        assert p1.loc[md, "pts_mean_3"] == pytest.approx(prior.mean())
        assert p1.loc[md, "pts_last"] == pytest.approx(p1.loc[md - 1, "points"])


def test_rolling_window_carries_across_the_season_break(matrix):
    """Deliberate: matchday 1 has form from last season rather than nothing.

    `matchday` and `season_games` are in the feature set so the model can
    learn how much to discount that carry-over.
    """
    p1 = matrix[matrix["player_id"] == "p1"].sort_values("kickoff")
    first_of_season = p1[p1["season_id"] == SEASON].iloc[0]
    last_of_previous = p1[p1["season_id"] == PREV_SEASON].iloc[-1]
    assert first_of_season["pts_last"] == pytest.approx(last_of_previous["points"])
    # ...but never its own result.
    assert first_of_season["pts_last"] != first_of_season["points"]


def test_shuffling_future_points_leaves_features_untouched(conn):
    """A direct leakage probe: rewrite the last matchday's results and every
    feature for earlier matchdays must be byte-identical."""
    before = features.build_matrix(conn, now=NOW, max_seasons=None)
    conn.execute(
        "UPDATE performances SET points = points * -3 WHERE season_id = ? AND matchday = ?",
        (SEASON, PLAYED_MATCHDAYS),
    )
    conn.commit()
    after = features.build_matrix(conn, now=NOW, max_seasons=None)

    key = ["player_id", "season_id", "matchday"]
    earlier = before["matchday"] < PLAYED_MATCHDAYS
    cols = key + [c for c in features.FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        before[earlier][cols].reset_index(drop=True),
        after[after["matchday"] < PLAYED_MATCHDAYS][cols].reset_index(drop=True),
    )


def test_play_and_start_shares_track_a_rotation_player(matrix):
    """p4 only plays on even matchdays -- the share features must show it."""
    p4 = matrix[(matrix["player_id"] == "p4") & (matrix["season_id"] == SEASON)]
    p4 = p4.sort_values("matchday").set_index("matchday")
    # Looking back at matchdays 2..6 from matchday 7: appearances on 2, 4, 6.
    assert p4.loc[7, "play_share_5"] == pytest.approx(3 / 5)
    assert p4.loc[7, "start_share_5"] == pytest.approx(3 / 5)
    # A regular starter sits at 1.0.
    p1 = matrix[(matrix["player_id"] == "p1") & (matrix["season_id"] == SEASON)]
    assert p1.set_index("matchday").loc[7, "play_share_5"] == pytest.approx(1.0)


def test_labels_come_from_the_lineup_status(matrix):
    """`st` is the lineup: 5 started, 3 came on, 4 unused bench, 1 not in squad."""
    rows = matrix[matrix["season_id"] == SEASON].set_index(["player_id", "matchday"])
    # p8 came on for 20 minutes on matchday 5: played, did not start.
    assert rows.loc[("p8", 5), ["started", "played", "in_squad"]].tolist() == [0.0, 1.0, 1.0]
    # p4 on matchday 3 (md % 4 == 3): unused bench.
    assert rows.loc[("p4", 3), ["started", "played", "in_squad"]].tolist() == [0.0, 0.0, 1.0]
    # p4 on matchday 5 (md % 4 == 1): not in the squad.
    assert rows.loc[("p4", 5), ["started", "played", "in_squad"]].tolist() == [0.0, 0.0, 0.0]
    assert rows.loc[("p1", 5), ["started", "played", "in_squad"]].tolist() == [1.0, 1.0, 1.0]


def test_an_early_substitution_still_counts_as_a_start(conn):
    """The old `minutes >= 60` proxy called this starter a non-starter."""
    conn.execute(
        "UPDATE performances SET minutes = 30 WHERE player_id = 'p2' AND season_id = ? "
        "AND matchday = 4",
        (SEASON,),
    )
    conn.commit()
    m = features.build_matrix(conn, now=NOW, max_seasons=None)
    row = m[(m["player_id"] == "p2") & (m["season_id"] == SEASON) & (m["matchday"] == 4)]
    assert row["started"].iloc[0] == 1.0


def test_a_kicked_off_fixture_without_a_result_is_unknown_not_zero(conn):
    """A stale history: kickoff passed, but the feed still says `st == 0`."""
    later = NOW + timedelta(days=5)  # matchday 11 has kicked off by then
    m = features.build_matrix(conn, now=later, max_seasons=None)
    md11 = m[m["matchday"] == SCHEDULED_MATCHDAY]
    assert md11["completed"].all()
    assert md11["played"].isna().all()
    assert md11["points"].isna().all()
    # ...and therefore never a training row.
    assert not (features.training_rows(m)["matchday"] == SCHEDULED_MATCHDAY).any()


def test_last_match_role_and_start_streak(matrix):
    p4 = matrix[(matrix["player_id"] == "p4") & (matrix["season_id"] == SEASON)]
    p4 = p4.sort_values("matchday").set_index("matchday")
    assert p4.loc[7, "started_last"] == 1.0  # started matchday 6
    assert p4.loc[7, "start_streak"] == 1.0
    assert p4.loc[6, "started_last"] == 0.0
    assert p4.loc[6, "squad_last"] == 0.0  # matchday 5: out of the squad
    assert p4.loc[6, "start_streak"] == 0.0
    p1 = matrix[(matrix["player_id"] == "p1")].sort_values("kickoff")
    # p1 starts everything; the streak carries across the season break.
    p1_now = p1[p1["season_id"] == SEASON].set_index("matchday")
    assert p1_now.loc[1, "start_streak"] == PLAYED_MATCHDAYS
    assert p1_now.loc[3, "start_streak"] == PLAYED_MATCHDAYS + 2


def test_team_context_counts_starters_of_earlier_games_only(matrix):
    md = matrix[(matrix["season_id"] == SEASON)].set_index(["player_id", "matchday"])
    # Team 1 fielded its forward p4 on even matchdays only. Looking back from
    # matchday 7 over 2..6: forwards started on 2, 4, 6.
    assert md.loc[("p4", 7), "team_pos_starters_5"] == pytest.approx(3 / 5)
    # Every position has a single player in the synthetic squads.
    assert (matrix["start_rank_pos"].dropna() == 1.0).all()
    upcoming = matrix[matrix["matchday"] == SCHEDULED_MATCHDAY]
    assert upcoming["team_pos_starters_5"].notna().all()
    assert upcoming["team_rotation_5"].notna().all()


def test_start_rank_orders_teammates_at_one_position(conn):
    conn.execute(
        "INSERT INTO players (player_id, last_name, team_id, position, status) "
        "VALUES ('p9', 'Lastp9', '1', 2, 0)"
    )
    conn.execute(
        """INSERT INTO performances (player_id, season_id, matchday, match_id, competition,
               points, minutes, kickoff, home_team_id, away_team_id, player_team_id,
               perf_status)
           SELECT 'p9', season_id, matchday, match_id, competition, NULL, 0, kickoff,
                  home_team_id, away_team_id, player_team_id, 4
           FROM performances WHERE player_id = 'p2'"""
    )
    conn.commit()
    m = features.build_matrix(conn, now=NOW, max_seasons=None)
    up = m[m["matchday"] == SCHEDULED_MATCHDAY].set_index("player_id")
    assert up.loc["p2", "start_rank_pos"] == 1.0
    assert up.loc["p9", "start_rank_pos"] == 2.0
    # One defender started each game, so the bench defender sits one outside.
    assert up.loc["p9", "start_rank_vs_slots"] == pytest.approx(1.0)


def test_rewriting_the_last_lineup_leaves_earlier_features_untouched(conn):
    """Leak probe for the lineup labels and the team tallies built on them."""
    before = features.build_matrix(conn, now=NOW, max_seasons=None)
    conn.execute(
        "UPDATE performances SET perf_status = 1, minutes = 0 "
        "WHERE season_id = ? AND matchday = ?",
        (SEASON, PLAYED_MATCHDAYS),
    )
    conn.commit()
    after = features.build_matrix(conn, now=NOW, max_seasons=None)
    cols = ["player_id", "season_id", "matchday", *features.FEATURE_COLUMNS]
    earlier = before["matchday"] < PLAYED_MATCHDAYS
    pd.testing.assert_frame_equal(
        before[earlier][cols].reset_index(drop=True),
        after[after["matchday"] < PLAYED_MATCHDAYS][cols].reset_index(drop=True),
    )


def test_form_over_appearances_skips_matchdays_off_the_pitch(matrix):
    p4 = matrix[(matrix["player_id"] == "p4") & (matrix["season_id"] == SEASON)]
    p4 = p4.sort_values("matchday").set_index("matchday")
    appearances = p4[p4["played"] == 1.0]["points"]
    # At matchday 7 the last three appearances were matchdays 2, 4 and 6.
    expected = appearances.loc[[2, 4, 6]].mean()
    assert p4.loc[7, "pts_app_mean_3"] == pytest.approx(expected)


def test_home_flag_matches_the_fixture(matrix):
    md1 = matrix[(matrix["matchday"] == 1) & (matrix["season_id"] == SEASON)]
    home = md1[md1["team_id"] == "1"]
    away = md1[md1["team_id"] == "2"]
    assert (home["is_home"] == 1).all()
    assert (away["is_home"] == 0).all()


def test_opponent_strength_is_populated_for_the_upcoming_fixture(matrix):
    upcoming = matrix[matrix["matchday"] == SCHEDULED_MATCHDAY]
    for col in ("opp_allowed_pos", "opp_allowed_all", "team_scored_all"):
        assert upcoming[col].notna().all(), col


def test_market_value_uses_the_day_before_kickoff(matrix):
    rows = matrix[matrix["log_mv"].notna()]
    assert len(rows) > 0
    assert np.isfinite(rows["log_mv"]).all()
    # The synthetic series rises monotonically, so the trend must be positive.
    trend = matrix["mv_trend_7"].dropna()
    assert len(trend) > 0
    assert (trend > 0).all()


def test_max_seasons_keeps_only_recent_seasons(conn):
    everything = features.build_matrix(conn, now=NOW, max_seasons=None)
    recent = features.build_matrix(conn, now=NOW, max_seasons=1)
    assert everything["season_id"].nunique() == 2
    assert recent["season_id"].nunique() == 1
    assert set(recent["season_id"]) == {SEASON}


def test_prediction_rows_selects_one_matchday(matrix):
    rows = features.prediction_rows(matrix, SEASON, SCHEDULED_MATCHDAY)
    assert len(rows) == 8
    assert (rows["matchday"] == SCHEDULED_MATCHDAY).all()


def test_training_rows_are_all_completed(matrix):
    train = features.training_rows(matrix)
    assert train["completed"].all()
    assert train["points"].notna().all()
