"""Milestone 3 -- walk-forward validation.

For each held-out matchday: train on everything that kicked off before it,
predict it, score it. This is the only honest way to evaluate a time series --
a random split would let the model read matchday 30 while predicting
matchday 5.

Run locally (`kickbase-xp validate`), not in the nightly Action: it retrains
once per matchday and there is nothing to publish from it.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import features
from .baselines import build_baselines
from .model import TwoStageModel

log = logging.getLogger(__name__)


def score(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype="float64")
    y_pred = np.asarray(y_pred, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "spearman": float("nan")}
    err = y_true - y_pred
    rho = pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")
    return {
        "n": int(len(y_true)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "spearman": float(rho) if rho == rho else float("nan"),
    }


@dataclass
class ValidationResult:
    per_fold: pd.DataFrame
    summary: pd.DataFrame

    def report(self) -> str:
        lines = ["Walk-forward validation", "=" * 60]
        lines.append(self.summary.to_string(index=False))
        best = self.summary.sort_values("mae").iloc[0]
        lines.append("")
        lines.append(f"Lowest MAE: {best['predictor']} ({best['mae']:.2f})")
        return "\n".join(lines)


def walk_forward(
    matrix: pd.DataFrame,
    *,
    season_id: str | None = None,
    first_matchday: int = 6,
    last_matchday: int | None = None,
    min_train_rows: int = 2000,
    fit_quantiles: bool = False,
) -> ValidationResult:
    """Retrain-and-predict over each matchday of one season.

    `first_matchday` defaults to 6 because the rolling-5 features -- and the
    baseline built on them -- are not defined before that.
    """
    done = matrix[matrix["completed"]].copy()
    if done.empty:
        raise ValueError("nothing completed to validate on")
    done["season_order"] = pd.to_numeric(done["season_id"], errors="coerce")

    if season_id is None:
        season_id = str(done.loc[done["season_order"].idxmax(), "season_id"])
    season = done[done["season_id"].astype(str) == str(season_id)]
    if season.empty:
        raise ValueError(f"season {season_id} has no completed rows")

    matchdays = sorted(int(m) for m in season["matchday"].dropna().unique())
    matchdays = [m for m in matchdays if m >= first_matchday]
    if last_matchday is not None:
        matchdays = [m for m in matchdays if m <= last_matchday]

    rows: list[dict] = []
    for md in matchdays:
        target = season[season["matchday"] == md]
        if target.empty:
            continue
        cutoff = pd.to_datetime(target["kickoff"], utc=True).min()
        train = done[pd.to_datetime(done["kickoff"], utc=True) < cutoff]
        if len(train) < min_train_rows:
            log.info("matchday %s: only %d training rows, skipping", md, len(train))
            continue

        y_true = target["points"].to_numpy()

        model = TwoStageModel(fit_quantiles=fit_quantiles)
        model.fit(train, reference=cutoff)
        # No status override here: the flag is only known for *today*, so a
        # historical fold cannot use it without inventing information.
        pred = model.predict(target)
        rows.append({"matchday": md, "predictor": "two_stage_lgbm", **score(y_true, pred["xP"])})

        for bl in build_baselines():
            bl.fit(train)
            rows.append(
                {"matchday": md, "predictor": bl.name, **score(y_true, bl.predict(target))}
            )
        log.info("matchday %s validated on %d rows (train %d)", md, len(target), len(train))

    per_fold = pd.DataFrame(rows)
    if per_fold.empty:
        raise ValueError("no fold produced results -- widen the matchday range")

    summary = (
        per_fold.groupby("predictor")
        .apply(
            lambda g: pd.Series(
                {
                    "folds": len(g),
                    "n": int(g["n"].sum()),
                    # Weight by fold size so a short matchday does not count
                    # the same as a full one.
                    "mae": float(np.average(g["mae"], weights=g["n"])),
                    "rmse": float(np.average(g["rmse"], weights=g["n"])),
                    "spearman": float(np.nanmean(g["spearman"])),
                }
            ),
            include_groups=False,
        )
        .reset_index()
        .sort_values("mae")
    )
    return ValidationResult(per_fold=per_fold, summary=summary)


def run_validation(conn: sqlite3.Connection, **kwargs) -> ValidationResult:
    matrix = features.build_matrix(
        conn,
        min_season=kwargs.pop("min_season", None),
        max_seasons=kwargs.pop("max_seasons", features.DEFAULT_MAX_SEASONS),
    )
    return walk_forward(matrix, **kwargs)
