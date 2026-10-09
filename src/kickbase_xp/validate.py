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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import features, lineups
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


PROBABILITY_EPS = 1e-6

# Stage-1 targets: (name, label column, model output column).
CLASSIFICATION_TARGETS = (
    ("squad", "in_squad", "p_squad"),
    ("play", "played", "p_play"),
    ("start", "started", "p_start"),
)

# Probability baselines per target: the obvious heuristics a lineup model
# has to beat. Missing values (no history yet) fall back to the training rate.
PROBABILITY_BASELINES = {
    "squad": ("squad_share_5", "squad_last"),
    "play": ("play_share_5", "played_last"),
    "start": ("start_share_5", "started_last"),
}


def score_proba(y_true: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Log loss and Brier score of a probability forecast for a 0/1 label."""
    y_true = np.asarray(y_true, dtype="float64")
    p = np.asarray(p, dtype="float64")
    mask = np.isfinite(y_true) & np.isfinite(p)
    y_true, p = y_true[mask], np.clip(p[mask], PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    if len(y_true) == 0:
        return {"n": 0, "log_loss": float("nan"), "brier": float("nan"), "rate": float("nan")}
    log_loss = -np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p))
    return {
        "n": int(len(y_true)),
        "log_loss": float(log_loss),
        "brier": float(np.mean((p - y_true) ** 2)),
        "rate": float(np.mean(y_true)),
    }


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


@dataclass
class ValidationResult:
    per_fold: pd.DataFrame
    summary: pd.DataFrame
    # Stage 1 as classifiers: log loss / Brier per target and predictor.
    classification: pd.DataFrame = field(default_factory=lambda: _empty(
        ["target", "predictor", "folds", "n", "log_loss", "brier"]))
    # Expected-lineup tiers against what actually happened.
    tiers: pd.DataFrame = field(default_factory=lambda: _empty(
        ["tier", "n", "start_rate", "play_rate", "squad_rate"]))
    # Predicted XI against the real one, model vs "same XI as last time".
    xi: pd.DataFrame = field(default_factory=lambda: _empty(
        ["predictor", "teams", "hits", "formation_hit"]))

    def report(self) -> str:
        lines = ["Walk-forward validation", "=" * 60]
        lines.append(self.summary.to_string(index=False))
        best = self.summary.sort_values("mae").iloc[0]
        lines.append("")
        lines.append(f"Lowest MAE: {best['predictor']} ({best['mae']:.2f})")
        if not self.classification.empty:
            lines += ["", "Playing time as classifiers (lower is better)", "-" * 60]
            lines.append(self.classification.to_string(index=False, float_format="{:.4f}".format))
        if not self.tiers.empty:
            lines += ["", "Lineup tiers: what actually happened", "-" * 60]
            lines.append(self.tiers.to_string(index=False, float_format="{:.3f}".format))
        if not self.xi.empty:
            lines += ["", "Expected XI: correct starters out of 11", "-" * 60]
            lines.append(self.xi.to_string(index=False, float_format="{:.3f}".format))
        return "\n".join(lines)


def walk_forward(
    matrix: pd.DataFrame,
    *,
    season_id: str | None = None,
    first_matchday: int = 6,
    last_matchday: int | None = None,
    min_train_rows: int = 2000,
    fit_quantiles: bool = False,
    evaluate_lineups: bool = True,
    status: pd.Series | None = None,
) -> ValidationResult:
    """Retrain-and-predict over each matchday of one season.

    `first_matchday` defaults to 6 because the rolling-5 features -- and the
    baseline built on them -- are not defined before that.

    `status`, aligned to `matrix`'s index, is each row's player status *as of
    the day before kickoff* (see `historical_status`). Without it no status
    cap is applied: the flag is only known for today, and a historical fold
    must not invent it.

    `evaluate_lineups` assembles every team's expected XI per fold and scores
    tiers and starters against the real lineup. The nightly predictor choice
    skips it; it has no bearing on xP.
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

    games = features.team_games(matrix) if evaluate_lineups else None

    rows: list[dict] = []
    cls_rows: list[dict] = []
    tier_frames: list[pd.DataFrame] = []
    xi_rows: list[dict] = []
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
        fold_status = status.reindex(target.index) if status is not None else None
        pred = model.predict(target, status=fold_status)
        rows.append({"matchday": md, "predictor": "two_stage_lgbm", **score(y_true, pred["xP"])})

        for bl in build_baselines():
            bl.fit(train)
            rows.append(
                {"matchday": md, "predictor": bl.name, **score(y_true, bl.predict(target))}
            )

        cls_rows += _classification_rows(md, target, train, pred)

        if evaluate_lineups:
            try:
                tiers, xi = _lineup_rows(md, target, pred, games, str(season_id))
            except Exception as exc:  # a degenerate fold must not end the run
                log.warning("matchday %s: lineup evaluation skipped (%s)", md, exc)
            else:
                tier_frames.append(tiers)
                xi_rows += xi
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
    result = ValidationResult(per_fold=per_fold, summary=summary)
    if cls_rows:
        result.classification = _summarise_classification(pd.DataFrame(cls_rows))
    if tier_frames:
        result.tiers = _summarise_tiers(pd.concat(tier_frames, ignore_index=True))
    if xi_rows:
        result.xi = _summarise_xi(pd.DataFrame(xi_rows))
    return result


# ------------------------------------------------------------ classification


def _classification_rows(
    md: int, target: pd.DataFrame, train: pd.DataFrame, pred: pd.DataFrame
) -> list[dict]:
    rows = []
    for name, label, col in CLASSIFICATION_TARGETS:
        y = target[label].to_numpy()
        rows.append({"matchday": md, "target": name, "predictor": "two_stage_lgbm",
                     **score_proba(y, pred[col].to_numpy())})
        fallback = float(train[label].mean())
        for feature in PROBABILITY_BASELINES[name]:
            p = target[feature].astype(float).fillna(fallback).to_numpy()
            rows.append({"matchday": md, "target": name, "predictor": feature,
                         **score_proba(y, p)})
    return rows


def _summarise_classification(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["n"] > 0]
    out = (
        df.groupby(["target", "predictor"])
        .apply(
            lambda g: pd.Series(
                {
                    "folds": len(g),
                    "n": int(g["n"].sum()),
                    "log_loss": float(np.average(g["log_loss"], weights=g["n"])),
                    "brier": float(np.average(g["brier"], weights=g["n"])),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    order = {name: i for i, (name, _, _) in enumerate(CLASSIFICATION_TARGETS)}
    out["_t"] = out["target"].map(order)
    return out.sort_values(["_t", "log_loss"]).drop(columns="_t").reset_index(drop=True)


# ------------------------------------------------------------------- lineups


def _lineup_rows(
    md: int, target: pd.DataFrame, pred: pd.DataFrame, games: pd.DataFrame, season_id: str
) -> tuple[pd.DataFrame, list[dict]]:
    """Assemble the fold's lineups and score them against the real ones."""
    cols = ["player_id", "team_id", "position", "match_id", "opponent_team_id", "is_home",
            "kickoff"]
    frame = target[cols].join(pred[["p_squad", "p_play", "p_start", "xP"]])
    result = lineups.build_lineups(frame, games, season_id, md)

    actual = target.set_index("player_id")[["started", "played", "in_squad"]]
    tier_rows = []
    xi_rows = []
    season_games = games[games["season_id"].astype(str) == season_id]
    for team in result.teams:
        players = team.players.set_index("player_id")[["tier"]].join(actual)
        tier_rows.append(players.reset_index())

        this = season_games[(season_games["team_id"].astype(str) == team.team_id)
                            & (season_games["matchday"] == md)]
        this = lineups._full_games(this)
        if this.empty:
            continue
        real = this.iloc[0]
        real_xi, real_shape = set(real["xi"]), lineups._shape(real)
        xi_rows.append({"matchday": md, "team_id": team.team_id, "predictor": "two_stage_lgbm",
                        "hits": len(set(team.xi["player_id"]) & real_xi),
                        "formation_hit": float(team.shape == real_shape)})

        before = season_games[(season_games["team_id"].astype(str) == team.team_id)
                              & (season_games["kickoff"] < real["kickoff"])]
        before = lineups._full_games(before).sort_values("kickoff")
        if not before.empty:
            prev = before.iloc[-1]
            xi_rows.append({"matchday": md, "team_id": team.team_id, "predictor": "last_xi",
                            "hits": len(set(prev["xi"]) & real_xi),
                            "formation_hit": float(lineups._shape(prev) == real_shape)})
    tiers = pd.concat(tier_rows, ignore_index=True) if tier_rows else pd.DataFrame()
    return tiers, xi_rows


def _summarise_tiers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["started"])
    rows = []
    for tier in lineups.TIERS:
        g = df[df["tier"] == tier]
        rows.append({
            "tier": tier,
            "n": int(len(g)),
            "start_rate": float(g["started"].mean()) if len(g) else float("nan"),
            "play_rate": float(g["played"].mean()) if len(g) else float("nan"),
            "squad_rate": float(g["in_squad"].mean()) if len(g) else float("nan"),
        })
    return pd.DataFrame(rows)


def _summarise_xi(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("predictor")
        .agg(teams=("hits", "size"), hits=("hits", "mean"), formation_hit=("formation_hit", "mean"))
        .reset_index()
        .sort_values("hits", ascending=False)
        .reset_index(drop=True)
    )


def historical_status(conn: sqlite3.Connection, matrix: pd.DataFrame) -> pd.Series:
    """Each row's player status as of the day before kickoff, where archived.

    Status is only ever served as "right now", so this exists only from the
    first nightly snapshot on (2026-09-10). Rows without a snapshot get NaN,
    which `model.status_cap` treats as fit.
    """
    snaps = pd.read_sql_query(
        "SELECT snapshot_date, player_id, status FROM status_snapshots", conn
    )
    if snaps.empty:
        return pd.Series(np.nan, index=matrix.index)
    snaps["day"] = pd.to_datetime(snaps["snapshot_date"], utc=True).astype("datetime64[ns, UTC]")
    snaps = snaps.dropna(subset=["day"]).sort_values("day")
    left = pd.DataFrame({
        "player_id": matrix["player_id"].astype(str),
        "day": (pd.to_datetime(matrix["kickoff"], utc=True).dt.normalize()
                - pd.Timedelta(days=1)).astype("datetime64[ns, UTC]"),
        "_row": np.arange(len(matrix)),
    }).sort_values("day", kind="mergesort")
    snaps["player_id"] = snaps["player_id"].astype(str)
    merged = pd.merge_asof(
        left, snaps[["day", "player_id", "status"]], on="day", by="player_id",
        direction="backward", tolerance=pd.Timedelta(days=3),
    )
    return merged.set_index("_row")["status"].sort_index().set_axis(matrix.index)


def run_validation(
    conn: sqlite3.Connection, *, use_historical_status: bool = False, **kwargs
) -> ValidationResult:
    matrix = features.build_matrix(
        conn,
        min_season=kwargs.pop("min_season", None),
        max_seasons=kwargs.pop("max_seasons", features.DEFAULT_MAX_SEASONS),
    )
    if use_historical_status:
        kwargs["status"] = historical_status(conn, matrix)
    return walk_forward(matrix, **kwargs)
