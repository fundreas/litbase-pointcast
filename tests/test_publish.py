"""The published tree is a public contract -- these tests pin its shape."""

from __future__ import annotations

import json

import pytest

from kickbase_xp import features
from kickbase_xp.publish import API_VERSION, publish
from kickbase_xp.train import run_training

from .conftest import NOW, SCHEDULED_MATCHDAY


@pytest.fixture
def published(conn, tmp_path):
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    out = tmp_path / "site"
    publish(run, out, feature_rows=run.feature_rows)
    return run, out


def _load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_target_matchday_is_the_next_unplayed_one(published):
    run, _ = published
    assert run.matchday == SCHEDULED_MATCHDAY


def test_expected_files_exist(published):
    run, out = published
    assert (out / ".nojekyll").exists()
    assert (out / "index.html").exists()
    assert (out / API_VERSION / "index.json").exists()
    assert (out / API_VERSION / "matchday" / f"{run.matchday}.json").exists()
    assert (out / API_VERSION / "matchday" / "current.json").exists()
    assert (out / API_VERSION / "players" / "index.json").exists()
    for pid in run.predictions["player_id"]:
        assert (out / API_VERSION / "players" / f"{pid}.json").exists()


def test_index_declares_its_version_and_endpoints(published):
    _, out = published
    index = _load(out / API_VERSION / "index.json")
    assert index["apiVersion"] == API_VERSION
    # Every advertised endpoint is versioned -- consumers pin /v1/ from day one.
    for template in index["endpoints"].values():
        assert template.startswith(f"/{API_VERSION}/")
    assert index["model"]["library"] == "lightgbm"
    assert index["playerCount"] > 0


def test_matchday_payload_matches_the_documented_schema(published):
    run, out = published
    payload = _load(out / API_VERSION / "matchday" / "current.json")
    assert payload["matchday"] == run.matchday
    assert payload["count"] == len(payload["players"])
    required = {
        "playerId", "name", "teamId", "position", "matchday",
        "xP", "p20", "p80", "pStart", "status", "generatedAt",
    }
    for entry in payload["players"]:
        assert required <= set(entry)
        assert isinstance(entry["playerId"], str)
    # Sorted by xP, best first.
    xps = [e["xP"] for e in payload["players"]]
    assert xps == sorted(xps, reverse=True)


def test_raw_play_probability_is_published_alongside_the_override(published):
    _, out = published
    payload = _load(out / API_VERSION / "matchday" / "current.json")
    for entry in payload["players"]:
        assert entry["pPlayRaw"] is not None
        # The override can only ever lower it.
        assert entry["pPlay"] <= entry["pPlayRaw"] + 1e-9


def test_home_away_is_published(published):
    _, out = published
    payload = _load(out / API_VERSION / "matchday" / "current.json")
    flags = {e["isHome"] for e in payload["players"]}
    assert flags == {True, False}


def test_no_negative_zero_reaches_the_output(published):
    """round() happily produces -0.0, which reads like a bug in a feed."""
    _, out = published
    for path in (out / API_VERSION).rglob("*.json"):
        assert "-0.0" not in path.read_text(encoding="utf-8"), path


def test_player_files_carry_the_feature_digest(published):
    run, out = published
    pid = run.predictions.iloc[0]["player_id"]
    entry = _load(out / API_VERSION / "players" / f"{pid}.json")
    assert entry["playerId"] == pid
    assert entry["features"], "expected a feature digest per player"
    assert "start_share_5" in entry["features"]


def test_json_is_finite_everywhere(published):
    """NaN would produce technically-invalid JSON that many clients reject."""
    _, out = published
    for path in (out / API_VERSION).rglob("*.json"):
        raw = path.read_text(encoding="utf-8")
        assert "NaN" not in raw and "Infinity" not in raw, path
        json.loads(raw)  # strict parse


def test_republishing_replaces_rather_than_accumulates(conn, tmp_path):
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    out = tmp_path / "site"
    publish(run, out, feature_rows=run.feature_rows)
    stale = out / API_VERSION / "players" / "999999.json"
    stale.write_text("{}", encoding="utf-8")
    publish(run, out, feature_rows=run.feature_rows)
    assert not stale.exists()


def test_fallback_predictor_is_recorded_in_the_output(conn, tmp_path):
    run = run_training(
        conn, max_seasons=None, auto_select=False, force_predictor="form_x_startshare", now=NOW
    )
    out = tmp_path / "site"
    publish(run, out, feature_rows=run.feature_rows)
    index = _load(out / API_VERSION / "index.json")
    assert index["predictor"] == "form_x_startshare"
    payload = _load(out / API_VERSION / "matchday" / "current.json")
    # The heuristic has no quantiles to offer; the field stays present but null.
    assert all(e["p20"] is None for e in payload["players"])
    assert all(e["xP"] is not None for e in payload["players"])


def test_feature_matrix_and_predictions_line_up(conn):
    run = run_training(conn, max_seasons=None, auto_select=False, now=NOW)
    matrix = features.build_matrix(conn, now=NOW, max_seasons=None)
    expected = features.prediction_rows(matrix, run.season_id, run.matchday)
    assert len(run.predictions) == len(expected)
    assert set(run.predictions["player_id"]) == set(expected["player_id"])
