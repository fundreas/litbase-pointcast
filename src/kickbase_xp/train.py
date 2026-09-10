"""Milestone 4 -- retrain from scratch and predict the next matchday.

There are no model artifacts. The dataset is small enough that a full retrain
is cheaper than versioning weights, and it guarantees the model always sees
everything up to last night (plan 2.2).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from . import features
from .baselines import FormBaseline
from .config import POSITIONS, STATUS_LABELS
from .model import TwoStageModel
from .validate import walk_forward

log = logging.getLogger(__name__)

MODEL_NAME = "two_stage_lgbm"
FALLBACK_NAME = "form_x_startshare"


@dataclass
class PredictionRun:
    matchday: int
    season_id: str
    predictions: pd.DataFrame
    # Feature rows behind the predictions, carried so publish.py can expose
    # them without rebuilding the matrix.
    feature_rows: pd.DataFrame = field(default_factory=pd.DataFrame)
    metadata: dict[str, Any] = field(default_factory=dict)


def current_season_id(matrix: pd.DataFrame) -> str:
    order = pd.to_numeric(matrix["season_id"], errors="coerce")
    return str(matrix.loc[order.idxmax(), "season_id"])


def next_matchday(matrix: pd.DataFrame, season_id: str) -> int:
    """First matchday of the season that has not kicked off yet."""
    season = matrix[matrix["season_id"].astype(str) == str(season_id)]
    upcoming = season[~season["completed"]]
    if upcoming.empty:
        raise ValueError(f"season {season_id} has no upcoming matchday left")
    return int(upcoming["matchday"].min())


def choose_predictor(
    matrix: pd.DataFrame, season_id: str, *, folds: int = 3
) -> tuple[str, pd.DataFrame | None]:
    """Plan 6.3 -- ship the heuristic if the model does not beat it.

    Checked against the most recent completed matchdays on every run, so a
    model that degrades mid-season demotes itself instead of quietly shipping
    worse numbers.
    """
    done = matrix[(matrix["completed"]) & (matrix["season_id"].astype(str) == str(season_id))]
    played_mds = sorted(int(m) for m in done["matchday"].dropna().unique())
    if len(played_mds) < folds + 1:
        log.info("only %d completed matchdays this season, skipping model selection", len(played_mds))
        return MODEL_NAME, None

    try:
        result = walk_forward(
            matrix,
            season_id=season_id,
            first_matchday=played_mds[-folds],
            last_matchday=played_mds[-1],
            fit_quantiles=False,
        )
    except ValueError as exc:
        log.warning("model selection skipped: %s", exc)
        return MODEL_NAME, None

    summary = result.summary.set_index("predictor")
    if MODEL_NAME not in summary.index or FALLBACK_NAME not in summary.index:
        return MODEL_NAME, result.summary
    model_mae = summary.loc[MODEL_NAME, "mae"]
    base_mae = summary.loc[FALLBACK_NAME, "mae"]
    winner = MODEL_NAME if model_mae <= base_mae else FALLBACK_NAME
    log.info(
        "recent-matchday MAE: model %.2f vs baseline %.2f -> shipping %s",
        model_mae,
        base_mae,
        winner,
    )
    return winner, result.summary


def _player_meta(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT p.player_id, p.first_name, p.last_name, p.team_id, t.name AS team_name,
               p.position, p.status, p.market_value, p.lineup_prob
        FROM players p LEFT JOIN teams t ON t.team_id = p.team_id
        """,
        conn,
    )


def run_training(
    conn: sqlite3.Connection,
    *,
    min_season: int | None = None,
    max_seasons: int | None = features.DEFAULT_MAX_SEASONS,
    matchday: int | None = None,
    auto_select: bool = True,
    selection_folds: int = 3,
    force_predictor: str | None = None,
    now: datetime | None = None,
) -> PredictionRun:
    now = now or datetime.now(timezone.utc)
    matrix = features.build_matrix(
        conn, now=now, min_season=min_season, max_seasons=max_seasons
    )
    if matrix.empty:
        raise ValueError("feature matrix is empty -- run `kickbase-xp fetch` first")

    season_id = current_season_id(matrix)
    target_md = matchday if matchday is not None else next_matchday(matrix, season_id)
    target = features.prediction_rows(matrix, season_id, target_md)
    if target.empty:
        raise ValueError(f"no rows for season {season_id} matchday {target_md}")

    train = features.training_rows(matrix)
    log.info(
        "season %s, target matchday %s: %d players, %d training rows",
        season_id,
        target_md,
        len(target),
        len(train),
    )

    predictor = force_predictor
    selection: pd.DataFrame | None = None
    if predictor is None:
        if auto_select:
            predictor, selection = choose_predictor(matrix, season_id, folds=selection_folds)
        else:
            predictor = MODEL_NAME

    meta_df = _player_meta(conn)
    status_by_player = meta_df.set_index("player_id")["status"]
    status = target["player_id"].map(status_by_player)

    model = TwoStageModel()
    model.fit(train, reference=pd.to_datetime(target["kickoff"], utc=True).min())
    preds = model.predict(target, status=status)
    importance = model.feature_importance()

    if predictor == FALLBACK_NAME:
        # Keep the model's playing-time stage (that is where its value is) and
        # swap only the points estimate for the heuristic.
        heuristic = FormBaseline().fit(train).predict(target)
        preds["xP_model"] = preds["xP"]
        preds["xP"] = heuristic
        preds["p20"] = np.nan
        preds["p50"] = np.nan
        preds["p80"] = np.nan

    out = target[
        [
            "player_id",
            "season_id",
            "matchday",
            "match_id",
            "kickoff",
            "team_id",
            "opponent_team_id",
            "is_home",
        ]
    ].copy()
    out = out.join(preds.drop(columns=["player_id", "matchday"]))
    out = out.merge(meta_df, on="player_id", how="left", suffixes=("", "_meta"))
    out["team_id"] = out["team_id"].fillna(out["team_id_meta"])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_meta")])
    out["position_label"] = out["position"].map(POSITIONS)
    out["status_label"] = out["status"].map(STATUS_LABELS).fillna("unknown")
    out = out.sort_values("xP", ascending=False).reset_index(drop=True)

    metadata = {
        "predictor": predictor,
        "model": {
            "name": MODEL_NAME,
            "library": "lightgbm",
            "stages": ["p_play", "p_start", "points_given_play", "quantiles"],
            "quantiles": list(model.quantiles),
            "train_rows": int(model.n_train_rows),
            "appearance_rows": int(model.n_points_rows),
            "features": features.FEATURE_COLUMNS,
        },
        "top_features": (
            importance[importance["model"] == "points"].head(10)[["feature", "gain"]]
            .to_dict("records")
        ),
        "selection": (selection.to_dict("records") if selection is not None else None),
        "generated_at": now.isoformat(timespec="seconds").replace("+00:00", "Z"),
    }
    return PredictionRun(
        matchday=target_md,
        season_id=season_id,
        predictions=out,
        feature_rows=target,
        metadata=metadata,
    )
