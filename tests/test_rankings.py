"""Actual-points rankings, and the shape they are published in."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from kickbase_xp import rankings as rk
from kickbase_xp.publish import API_VERSION, publish
from kickbase_xp.train import run_training

from .conftest import NOW, PLAYED_MATCHDAYS, SCHEDULED_MATCHDAY, SEASON, _kickoff


def _points(pos: int, md: int) -> int:
    """What conftest awards: 40 + 12 * position + 7 * matchday."""
    return 40 + 12 * pos + 7 * md


@pytest.fixture
def season(conn):
    return rk.build_rankings(conn, SEASON, now=NOW)


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ compute


def test_only_kicked_off_matchdays_are_ranked(season):
    assert [s.matchday for s in season.matchdays] == list(range(1, PLAYED_MATCHDAYS + 1))
    assert all(s.complete for s in season.matchdays)
    assert season.latest.matchday == PLAYED_MATCHDAYS


def test_ongoing_matchday_is_ranked_but_flagged_incomplete(conn):
    live = _kickoff(SCHEDULED_MATCHDAY) + timedelta(minutes=30)
    result = rk.build_rankings(conn, SEASON, now=live)
    status = result.latest
    assert status.matchday == SCHEDULED_MATCHDAY
    assert not status.complete
    assert status.matches_played == status.matches_total == 1


def test_cutoff_defaults_to_the_last_fetch(conn):
    # Fetched just after matchday 3 kicked off: later matchdays are unknown,
    # however late the publish runs.
    fetched = _kickoff(3) + timedelta(hours=4)
    conn.execute(
        "INSERT INTO meta (key, value) VALUES ('last_fetch_at', ?)", (fetched.isoformat(),)
    )
    result = rk.build_rankings(conn, SEASON)
    assert [s.matchday for s in result.matchdays] == [1, 2, 3]


def test_previous_season_does_not_leak_in(season):
    # Both seasons award identical points, so a leak would double every total.
    totals = season.cumulative[PLAYED_MATCHDAYS].set_index("player_id")["points"]
    expected = sum(_points(1, md) for md in range(1, PLAYED_MATCHDAYS + 1))
    assert totals["p1"] == expected


def test_matchday_table_skips_players_who_did_not_take_part(season):
    # p4 only appears on even matchdays.
    assert "p4" not in set(season.per_matchday[1]["player_id"])
    assert "p4" in set(season.per_matchday[2]["player_id"])


def test_season_totals_are_cumulative(season):
    p4 = season.cumulative[4].set_index("player_id").loc["p4"]
    assert p4["appearances"] == 2
    assert p4["points"] == _points(4, 2) + _points(4, 4)


def test_ties_share_a_rank_and_order_by_player_id(season):
    table = rk.ranked(season.per_matchday[2])
    # p4/p8 are both forwards and score the same; then the midfielders, etc.
    assert list(table["player_id"][:4]) == ["p4", "p8", "p3", "p7"]
    assert list(table["rank"][:4]) == [1, 1, 3, 3]


def test_position_ranking_restarts_at_one(season):
    gk = rk.ranked(season.per_matchday[2], position="GK")
    assert set(gk["player_id"]) == {"p1", "p5"}
    assert list(gk["rank"]) == [1, 1]


def test_cut_keeps_ties_at_the_boundary(season):
    table = rk.ranked(season.per_matchday[2], top=1)
    assert list(table["player_id"]) == ["p4", "p8"]


# ------------------------------------------------------------------ publish


@pytest.fixture
def published(conn, tmp_path, season):
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    out = tmp_path / "site"
    publish(run, out, feature_rows=run.feature_rows, rankings=season)
    return out / API_VERSION / "rankings"


def test_ranking_files_exist_for_every_matchday(published):
    for scope in ("matchday", "season"):
        for md in range(1, PLAYED_MATCHDAYS + 1):
            assert (published / scope / f"{md}.json").exists()
        current = _load(published / scope / "current.json")
        assert current["matchday"] == PLAYED_MATCHDAYS
        assert current == _load(published / scope / f"{PLAYED_MATCHDAYS}.json")


def test_rankings_index_lists_matchdays_and_endpoints(published):
    index = _load(published / "index.json")
    assert index["seasonId"] == SEASON
    assert index["latestMatchday"] == PLAYED_MATCHDAYS
    assert index["positions"] == ["GK", "DEF", "MID", "FWD"]
    assert [m["matchday"] for m in index["matchdays"]] == list(range(1, PLAYED_MATCHDAYS + 1))
    assert {"matchday", "complete", "matchesPlayed", "matchesTotal"} <= set(index["matchdays"][0])
    for template in index["endpoints"].values():
        assert template.startswith(f"/{API_VERSION}/rankings/")


def test_matchday_payload_shape(published):
    payload = _load(published / "matchday" / "2.json")
    assert payload["scope"] == "matchday"
    assert payload["complete"] is True
    assert set(payload["byPosition"]) == {"GK", "DEF", "MID", "FWD"}
    top = payload["overall"][0]
    assert top == {
        "rank": 1,
        "playerId": "p4",
        "name": "First Lastp4",
        "teamId": "1",
        "teamName": "Alpha",
        "position": "FWD",
        "points": _points(4, 2),
        "minutes": 90,
    }
    assert [e["playerId"] for e in payload["byPosition"]["GK"]] == ["p1", "p5"]


def test_season_payload_carries_appearances(published):
    payload = _load(published / "season" / "4.json")
    assert payload["scope"] == "season"
    p4 = next(e for e in payload["overall"] if e["playerId"] == "p4")
    assert p4["appearances"] == 2
    assert p4["pointsPerAppearance"] == (_points(4, 2) + _points(4, 4)) / 2
    points = [e["points"] for e in payload["overall"]]
    assert points == sorted(points, reverse=True)


def test_main_index_advertises_rankings(published):
    index = _load(published.parent / "index.json")
    assert index["endpoints"]["rankings"] == f"/{API_VERSION}/rankings/index.json"


def test_publishing_without_rankings_writes_none(conn, tmp_path):
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    out = tmp_path / "site"
    publish(run, out, feature_rows=run.feature_rows)
    assert not (out / API_VERSION / "rankings").exists()
