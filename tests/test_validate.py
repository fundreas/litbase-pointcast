"""Walk-forward harness, baselines, and the ship-the-heuristic guard."""

from __future__ import annotations

import numpy as np
import pytest

from kickbase_xp import features
from kickbase_xp.baselines import PositionAverageBaseline, build_baselines
from kickbase_xp.train import FALLBACK_NAME, MODEL_NAME, choose_predictor, run_training
from kickbase_xp.validate import score, walk_forward

from .conftest import NOW, SEASON


@pytest.fixture
def matrix(conn):
    return features.build_matrix(conn, now=NOW, max_seasons=None)


def test_score_is_exact_on_a_known_case():
    got = score(np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0]))
    assert got["n"] == 3
    assert got["mae"] == pytest.approx(7 / 3)
    assert got["rmse"] == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
    assert got["spearman"] == pytest.approx(1.0)


def test_score_survives_missing_values():
    got = score(np.array([1.0, np.nan, 3.0]), np.array([1.0, 2.0, np.nan]))
    assert got["n"] == 1
    assert got["mae"] == 0.0


def test_baselines_produce_one_number_per_row(matrix):
    train = features.training_rows(matrix)
    target = matrix[matrix["matchday"] == 5]
    for bl in build_baselines():
        bl.fit(train)
        pred = bl.predict(target)
        assert len(pred) == len(target)
        assert np.isfinite(pred).all(), bl.name


def test_position_average_uses_the_training_split_only(matrix):
    train = features.training_rows(matrix)
    bl = PositionAverageBaseline().fit(train)
    for pos, mean in bl.by_position.items():
        expected = train[train["position"] == pos]["points"].mean()
        assert mean == pytest.approx(expected)


def test_walk_forward_scores_every_predictor(matrix):
    result = walk_forward(matrix, season_id=SEASON, first_matchday=6, min_train_rows=50)
    predictors = set(result.summary["predictor"])
    assert MODEL_NAME in predictors
    assert FALLBACK_NAME in predictors
    assert (result.summary["mae"] >= 0).all()
    assert (result.per_fold["matchday"] >= 6).all()
    assert "Lowest MAE" in result.report()


def test_walk_forward_never_trains_on_the_fold_it_scores(matrix, monkeypatch):
    """Guard the harness itself: each fold's training cutoff must precede it."""
    seen: list[tuple] = []
    import kickbase_xp.validate as validate_mod

    original = validate_mod.TwoStageModel.fit

    def spy(self, train, *, reference=None):
        seen.append((train["kickoff"].max(), reference))
        return original(self, train, reference=reference)

    monkeypatch.setattr(validate_mod.TwoStageModel, "fit", spy)
    walk_forward(matrix, season_id=SEASON, first_matchday=6, min_train_rows=50)
    assert seen
    for latest_train_kickoff, cutoff in seen:
        assert latest_train_kickoff < cutoff


def test_choose_predictor_falls_back_when_the_model_loses(matrix, monkeypatch):
    import kickbase_xp.train as train_mod
    import pandas as pd

    def losing_summary(*_args, **_kwargs):
        summary = pd.DataFrame(
            {"predictor": [MODEL_NAME, FALLBACK_NAME], "mae": [90.0, 40.0], "n": [10, 10]}
        )
        return type("R", (), {"summary": summary})()

    monkeypatch.setattr(train_mod, "walk_forward", losing_summary)
    winner, _ = choose_predictor(matrix, SEASON, folds=2)
    assert winner == FALLBACK_NAME


def test_choose_predictor_keeps_the_model_when_it_wins(matrix, monkeypatch):
    import kickbase_xp.train as train_mod
    import pandas as pd

    def winning_summary(*_args, **_kwargs):
        summary = pd.DataFrame(
            {"predictor": [MODEL_NAME, FALLBACK_NAME], "mae": [40.0, 90.0], "n": [10, 10]}
        )
        return type("R", (), {"summary": summary})()

    monkeypatch.setattr(train_mod, "walk_forward", winning_summary)
    winner, summary = choose_predictor(matrix, SEASON, folds=2)
    assert winner == MODEL_NAME
    assert summary is not None


def test_run_training_records_the_selection_in_metadata(conn):
    run = run_training(conn, max_seasons=None, auto_select=True, now=NOW)
    assert run.metadata["predictor"] in {MODEL_NAME, FALLBACK_NAME}
    assert run.metadata["model"]["train_rows"] > 0
    assert run.metadata["generated_at"].endswith("Z")
    assert len(run.metadata["top_features"]) > 0
