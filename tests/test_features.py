"""Feature-matrix behaviour, above all: no information from the future."""

from __future__ import annotations

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
