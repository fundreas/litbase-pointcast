"""Milestone 4 -- the two-stage LightGBM model.

Kickbase points prediction is two problems wearing one coat:

1. will the player be on the pitch at all (and in the squad, and in the XI), and
2. what does he score once he is.

Modelling them jointly forces one tree ensemble to spend its capacity
separating benchwarmers from performers. Splitting them keeps the
playing-time error -- the part we genuinely cannot see without lineup news --
isolated in stage 1, where it belongs.

    xP = P(plays) x E[points | plays]

Stage 2 is fitted at a grid of quantiles as well as the mean, so the published
feed can carry a floor and a ceiling per player rather than a bare point
estimate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import STATUS_FIT
from .features import CATEGORICAL_COLUMNS, FEATURE_COLUMNS

log = logging.getLogger(__name__)

# Conditional quantiles fitted in stage 2. Dense enough that interpolating
# between them to an arbitrary level stays honest.
QUANTILES: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Published floor/ceiling levels (unconditional, i.e. including the chance of
# not playing at all).
FLOOR_QUANTILE = 0.2
CEILING_QUANTILE = 0.8

# Hard override from the plan: a status other than "fit" caps P(plays),
# whatever the model believes. Status is a *current* flag with no history, so
# it cannot be trained on without leaking -- it is applied afterwards instead.
STATUS_PLAY_CAP: dict[int, float] = {
    0: 1.00,  # fit
    1: 0.02,  # injured
    2: 0.50,  # questionable
    4: 0.00,  # suspended
    8: 0.05,  # absent from squad
    16: 0.02,  # rehab / build-up training
}
UNKNOWN_STATUS_CAP = 0.50

# Stage-1 targets, each a binary classifier of its own. Independent binaries
# rather than one multiclass head: `pStart`/`pPlay` keep their meaning, the
# status cap stays a plain `min`, and the noisy "came on as sub" class cannot
# blur the start/bench boundary. Monotonicity is restored after the fact.
LABEL_COLUMNS = ("in_squad", "played", "started")

# Recency weighting: an observation this old counts half as much.
RECENCY_HALFLIFE_DAYS = 450.0

BASE_PARAMS: dict = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.85,
    "reg_lambda": 1.0,
    "verbose": -1,
    "n_jobs": -1,
}


def status_cap(status: float | int | None) -> float:
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return STATUS_PLAY_CAP[STATUS_FIT]
    return STATUS_PLAY_CAP.get(int(status), UNKNOWN_STATUS_CAP)


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Feature frame in a shape LightGBM accepts, with stable column order."""
    X = df.reindex(columns=FEATURE_COLUMNS).copy()
    for col in CATEGORICAL_COLUMNS:
        X[col] = pd.Categorical(X[col].astype("Int64"), categories=[1, 2, 3, 4])
    for col in X.columns:
        if col not in CATEGORICAL_COLUMNS:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float64")
    return X


def recency_weights(kickoff: pd.Series, reference: pd.Timestamp) -> np.ndarray:
    age_days = (reference - pd.to_datetime(kickoff, utc=True)).dt.total_seconds() / 86400.0
    age_days = age_days.clip(lower=0.0).to_numpy()
    return np.exp(-np.log(2.0) * age_days / RECENCY_HALFLIFE_DAYS)


def blend_unconditional_quantile(
    p_play: np.ndarray, cond_q: np.ndarray, levels: np.ndarray, alpha: float
) -> np.ndarray:
    """Quantile of the *unconditional* points distribution.

    Points are a mixture: with probability ``1 - p_play`` the player does not
    appear and scores nothing; otherwise points follow the conditional
    distribution described by ``cond_q`` at ``levels``. The unconditional
    alpha-quantile therefore reads off the conditional distribution at the
    rescaled level ``(alpha - (1 - p)) / p``, and collapses to zero once the
    not-playing mass alone already covers alpha.
    """
    p = np.clip(np.asarray(p_play, dtype="float64"), 1e-9, 1.0)
    shifted = (alpha - (1.0 - p)) / p
    out = np.zeros_like(p)
    active = shifted > 0
    if active.any():
        # cond_q rows are already sorted ascending across `levels`.
        interp = np.array(
            [np.interp(s, levels, row) for s, row in zip(shifted[active], cond_q[active])]
        )
        out[active] = interp
    return out


class _ConstantClassifier:
    """Stands in for a classifier whose training labels had one class only.

    Happens on tiny or old data (seasons before 2021/22 carry almost no
    "not in squad" rows). LightGBM would fit a degenerate model; this says
    the same thing honestly and has no feature importance to report.
    """

    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = np.full(len(X), self.value)
        return np.column_stack([1.0 - p, p])


def _fit_classifier(params: dict, X: pd.DataFrame, y: pd.Series, w: np.ndarray | None):
    labels = y.astype(int)
    if labels.nunique() < 2:
        return _ConstantClassifier(labels.iloc[0] if len(labels) else 0.0)
    m = lgb.LGBMClassifier(**params)
    m.fit(X, labels, sample_weight=w)
    return m


def _positive_proba(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X)[:, 1]


@dataclass
class TwoStagePrediction:
    frame: pd.DataFrame
    feature_importance: pd.DataFrame = field(default_factory=pd.DataFrame)


class TwoStageModel:
    """Fit-from-scratch model. No artifacts, no drift handling -- see plan 2.2."""

    def __init__(
        self,
        *,
        params: dict | None = None,
        quantiles: tuple[float, ...] = QUANTILES,
        use_recency_weights: bool = True,
        fit_quantiles: bool = True,
    ) -> None:
        self.params = {**BASE_PARAMS, **(params or {})}
        self.quantiles = tuple(sorted(quantiles))
        self.use_recency_weights = use_recency_weights
        self.fit_quantiles = fit_quantiles
        self.squad_model: lgb.LGBMClassifier | None = None
        self.play_model: lgb.LGBMClassifier | None = None
        self.start_model: lgb.LGBMClassifier | None = None
        self.points_model: lgb.LGBMRegressor | None = None
        self.quantile_models: dict[float, lgb.LGBMRegressor] = {}
        self.n_train_rows = 0
        self.n_points_rows = 0

    # ------------------------------------------------------------------ fit

    def fit(self, train: pd.DataFrame, *, reference: pd.Timestamp | None = None) -> "TwoStageModel":
        train = train[train["completed"]].dropna(subset=list(LABEL_COLUMNS))
        if train.empty:
            raise ValueError("no completed rows to train on")
        reference = reference or pd.to_datetime(train["kickoff"], utc=True).max()

        X = design_matrix(train)
        w = recency_weights(train["kickoff"], reference) if self.use_recency_weights else None
        self.n_train_rows = len(train)

        self.squad_model = _fit_classifier(self.params, X, train["in_squad"], w)
        self.play_model = _fit_classifier(self.params, X, train["played"], w)
        self.start_model = _fit_classifier(self.params, X, train["started"], w)

        played = train[train["played"] == 1.0]
        if played.empty:
            raise ValueError("no appearances to train the points model on")
        Xp = design_matrix(played)
        yp = played["points"].astype("float64")
        wp = recency_weights(played["kickoff"], reference) if self.use_recency_weights else None
        self.n_points_rows = len(played)

        self.points_model = lgb.LGBMRegressor(objective="regression", **self.params)
        self.points_model.fit(Xp, yp, sample_weight=wp)

        self.quantile_models = {}
        if self.fit_quantiles:
            for alpha in self.quantiles:
                m = lgb.LGBMRegressor(objective="quantile", alpha=alpha, **self.params)
                m.fit(Xp, yp, sample_weight=wp)
                self.quantile_models[alpha] = m

        log.info(
            "trained on %d rows (%d appearances), %d quantile models",
            self.n_train_rows,
            self.n_points_rows,
            len(self.quantile_models),
        )
        return self

    # -------------------------------------------------------------- predict

    def predict(self, rows: pd.DataFrame, *, status: pd.Series | None = None) -> pd.DataFrame:
        if self.play_model is None or self.points_model is None:
            raise RuntimeError("model is not fitted")
        X = design_matrix(rows)

        p_squad_raw = _positive_proba(self.squad_model, X)
        p_play_raw = _positive_proba(self.play_model, X)
        p_start_raw = _positive_proba(self.start_model, X)

        if status is None:
            caps = np.ones(len(rows))
        else:
            caps = np.array([status_cap(s) for s in status.to_numpy()], dtype="float64")
        # Capped, then chained: a start implies an appearance, which implies a
        # squad place. The chain only ever lowers a probability.
        p_squad = np.minimum(p_squad_raw, caps)
        p_play = np.minimum(np.minimum(p_play_raw, caps), p_squad)
        p_start = np.minimum(np.minimum(p_start_raw, caps), p_play)

        points_given_play = self.points_model.predict(X)
        xp = p_play * points_given_play

        out = pd.DataFrame(
            {
                "player_id": rows["player_id"].to_numpy(),
                "matchday": rows["matchday"].to_numpy(),
                "p_squad_raw": p_squad_raw,
                "p_squad": p_squad,
                "p_play_raw": p_play_raw,
                "p_play": p_play,
                "p_start_raw": p_start_raw,
                "p_start": p_start,
                "status_cap": caps,
                "points_given_play": points_given_play,
                "xP": xp,
            },
            index=rows.index,
        )

        if self.quantile_models:
            levels = np.array(self.quantiles)
            cond = np.column_stack(
                [self.quantile_models[a].predict(X) for a in self.quantiles]
            )
            # Quantile models are fitted independently and can cross; a
            # per-row sort restores monotonicity without changing the levels.
            cond = np.sort(cond, axis=1)
            out["p20"] = blend_unconditional_quantile(p_play, cond, levels, FLOOR_QUANTILE)
            out["p50"] = blend_unconditional_quantile(p_play, cond, levels, 0.5)
            out["p80"] = blend_unconditional_quantile(p_play, cond, levels, CEILING_QUANTILE)
        else:
            out["p20"] = np.nan
            out["p50"] = np.nan
            out["p80"] = np.nan
        return out

    # ----------------------------------------------------------- inspection

    def feature_importance(self) -> pd.DataFrame:
        frames = []
        for name, m in (
            ("p_squad", self.squad_model),
            ("p_play", self.play_model),
            ("p_start", self.start_model),
            ("points", self.points_model),
        ):
            if m is None or not hasattr(m, "booster_"):
                continue
            frames.append(
                pd.DataFrame(
                    {
                        "model": name,
                        "feature": FEATURE_COLUMNS,
                        "gain": m.booster_.feature_importance(importance_type="gain"),
                    }
                )
            )
        if not frames:
            return pd.DataFrame(columns=["model", "feature", "gain"])
        out = pd.concat(frames, ignore_index=True)
        return out.sort_values(["model", "gain"], ascending=[True, False]).reset_index(drop=True)
