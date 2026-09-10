"""Two-stage model: the composition rule, the status override, the quantiles."""

from __future__ import annotations

import numpy as np
import pytest

from kickbase_xp import features
from kickbase_xp.model import (
    QUANTILES,
    STATUS_PLAY_CAP,
    TwoStageModel,
    blend_unconditional_quantile,
    status_cap,
)

from .conftest import NOW, SCHEDULED_MATCHDAY, SEASON

LEVELS = np.array(QUANTILES)


def _cond(values: list[float]) -> np.ndarray:
    """One row of conditional quantiles, evaluated at every level in QUANTILES."""
    return np.array([np.interp(LEVELS, np.linspace(0.1, 0.9, len(values)), values)])


def test_certain_starter_keeps_the_conditional_quantile():
    cond = _cond([0.0, 50.0, 100.0, 150.0, 200.0])
    got = blend_unconditional_quantile(np.array([1.0]), cond, LEVELS, 0.5)
    assert got[0] == pytest.approx(np.interp(0.5, LEVELS, cond[0]))


def test_unlikely_starter_has_a_floor_of_zero():
    """With a 30% chance of playing, the 20th percentile is "did not play"."""
    cond = _cond([10.0, 60.0, 110.0, 160.0, 210.0])
    floor = blend_unconditional_quantile(np.array([0.3]), cond, LEVELS, 0.2)
    assert floor[0] == 0.0
    # ...while the ceiling still reflects a real performance.
    ceiling = blend_unconditional_quantile(np.array([0.3]), cond, LEVELS, 0.8)
    assert ceiling[0] > 0.0


def test_blended_quantiles_are_monotone_in_alpha():
    cond = _cond([-20.0, 30.0, 90.0, 140.0, 250.0])
    for p in (0.05, 0.25, 0.5, 0.75, 1.0):
        q = [
            blend_unconditional_quantile(np.array([p]), cond, LEVELS, a)[0]
            for a in (0.1, 0.2, 0.5, 0.8, 0.9)
        ]
        assert q == sorted(q), f"non-monotone at p={p}: {q}"


def test_lower_play_probability_never_raises_a_quantile():
    cond = _cond([0.0, 40.0, 80.0, 120.0, 200.0])
    previous = np.inf
    for p in (1.0, 0.8, 0.6, 0.4, 0.2):
        q = blend_unconditional_quantile(np.array([p]), cond, LEVELS, 0.5)[0]
        assert q <= previous + 1e-9
        previous = q


def test_status_cap_table():
    assert status_cap(0) == 1.0
    assert status_cap(4) == 0.0  # suspended: cannot play
    assert status_cap(1) < 0.05  # injured
    assert status_cap(None) == 1.0
    assert 0.0 < status_cap(999) < 1.0  # unknown code: cautious, not certain


@pytest.fixture
def fitted(conn):
    matrix = features.build_matrix(conn, now=NOW, max_seasons=None)
    train = features.training_rows(matrix)
    model = TwoStageModel(params={"n_estimators": 40}).fit(train)
    target = features.prediction_rows(matrix, SEASON, SCHEDULED_MATCHDAY)
    return model, target


def test_prediction_shape_and_composition(fitted):
    model, target = fitted
    out = model.predict(target)
    assert len(out) == len(target)
    assert set(["xP", "p20", "p50", "p80", "p_play", "p_start"]) <= set(out.columns)
    # xP is exactly P(plays) x E[points | plays] -- the plan's headline formula.
    np.testing.assert_allclose(out["xP"], out["p_play"] * out["points_given_play"])
    assert ((out["p_play"] >= 0) & (out["p_play"] <= 1)).all()
    # Starting implies appearing.
    assert (out["p_start"] <= out["p_play"] + 1e-9).all()


def test_status_override_zeroes_a_suspended_player(fitted):
    import pandas as pd

    model, target = fitted
    fit = model.predict(target, status=pd.Series([0] * len(target), index=target.index))
    suspended = model.predict(target, status=pd.Series([4] * len(target), index=target.index))
    assert (suspended["p_play"] == 0.0).all()
    assert (suspended["xP"] == 0.0).all()
    # The override is applied after the model, which is otherwise unchanged.
    np.testing.assert_allclose(fit["p_play_raw"], suspended["p_play_raw"])


def test_injury_status_slashes_but_does_not_zero(fitted):
    import pandas as pd

    model, target = fitted
    injured = model.predict(target, status=pd.Series([1] * len(target), index=target.index))
    assert (injured["p_play"] <= STATUS_PLAY_CAP[1] + 1e-12).all()
    assert (injured["xP"].abs() < 20).all()


def test_feature_importance_covers_every_stage(fitted):
    model, _ = fitted
    imp = model.feature_importance()
    assert set(imp["model"]) == {"p_play", "p_start", "points"}
    assert set(imp["feature"]) == set(features.FEATURE_COLUMNS)
