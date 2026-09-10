"""Milestone 3 -- the bars the model has to clear.

Plan 6.3: if the model cannot beat naive form, ship the heuristic instead.
`FormBaseline` is therefore a first-class predictor, not just a yardstick --
`train.py` will fall back to it when validation says so.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class Baseline:
    name = "baseline"

    def fit(self, train: pd.DataFrame) -> "Baseline":
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError


class FormBaseline(Baseline):
    """Plan 6.3 #1 -- last-5 scoring form scaled by how often he starts."""

    name = "form_x_startshare"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        form = pd.to_numeric(rows["pts_app_mean_5"], errors="coerce")
        form = form.fillna(pd.to_numeric(rows["pts_mean_5"], errors="coerce"))
        share = pd.to_numeric(rows["start_share_5"], errors="coerce").fillna(0.0)
        return (form.fillna(0.0) * share).to_numpy()


class RollingMeanBaseline(Baseline):
    """The dumbest thing that could work: last five matchdays, zeros included."""

    name = "rolling_mean_5"

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return pd.to_numeric(rows["pts_mean_5"], errors="coerce").fillna(0.0).to_numpy()


class PositionAverageBaseline(Baseline):
    """Plan 6.3 #2 -- one number per position."""

    name = "position_average"

    def __init__(self) -> None:
        self.by_position: dict[int, float] = {}
        self.overall = 0.0

    def fit(self, train: pd.DataFrame) -> "PositionAverageBaseline":
        done = train[train["completed"]]
        self.overall = float(done["points"].mean()) if len(done) else 0.0
        self.by_position = {
            int(k): float(v) for k, v in done.groupby("position")["points"].mean().items()
        }
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        pos = pd.to_numeric(rows["position"], errors="coerce")
        return pos.map(self.by_position).fillna(self.overall).to_numpy()


ALL_BASELINES: tuple[type[Baseline], ...] = (
    FormBaseline,
    RollingMeanBaseline,
    PositionAverageBaseline,
)


def build_baselines() -> list[Baseline]:
    return [cls() for cls in ALL_BASELINES]
