"""Command line entry point.

    kickbase-xp fetch      # API -> data/history.sqlite
    kickbase-xp predict    # features -> two-stage model -> next matchday
    kickbase-xp publish    # predictions -> site/v1/*.json
    kickbase-xp run        # all three, i.e. what the nightly Action does
    kickbase-xp validate   # walk-forward MAE report (local, one-time)
    kickbase-xp features   # dump the feature matrix for inspection
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from . import archive, db, features
from .api import KickbaseClient
from .config import (
    COMPETITION_BUNDESLIGA,
    DEFAULT_DB_PATH,
    DEFAULT_OUT_DIR,
    DEFAULT_SNAPSHOT_DIR,
    load_credentials,
    load_dotenv,
)
from .fetch import run_fetch
from .publish import publish
from .train import run_training
from .validate import walk_forward


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def _connect(args: argparse.Namespace):
    """Open the working database and replay the committed daily archive.

    The SQLite file is a cache that a fresh CI runner does not have; the
    snapshots in `data/snapshots/` are the part that is actually committed.
    """
    conn = db.connect(args.db)
    archive.load_snapshots(conn, args.snapshots)
    return conn


def _client(args: argparse.Namespace) -> KickbaseClient:
    return KickbaseClient(
        load_credentials(), delay=args.delay, competition_id=args.competition
    )


def cmd_fetch(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    try:
        result = run_fetch(
            _client(args),
            conn,
            snapshot_dir=args.snapshots,
            skip_performances=args.skip_performances,
            skip_market_values=args.skip_market_values,
            player_limit=args.limit,
        )
    finally:
        conn.close()
    print(
        f"season {result['season_id']}, current matchday {result['current_matchday']}\n"
        + "\n".join(f"  {k:<18}{v:>8}" for k, v in result["counts"].items())
    )
    return 0


def _predict(args: argparse.Namespace):
    conn = _connect(args)
    try:
        run = run_training(
            conn,
            max_seasons=args.seasons,
            matchday=args.matchday,
            auto_select=not args.no_auto_select,
            force_predictor=args.predictor,
        )
    finally:
        conn.close()
    return run


def cmd_predict(args: argparse.Namespace) -> int:
    run = _predict(args)
    cols = ["last_name", "team_name", "position_label", "status_label", "xP", "p20", "p80",
            "p_start"]
    top = run.predictions[cols].head(args.top)
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(f"\nMatchday {run.matchday} — predictor: {run.metadata['predictor']}\n")
        print(top.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if args.out_csv:
        run.predictions.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")
    return 0


def cmd_publish(args: argparse.Namespace) -> int:
    run = _predict(args)
    result = publish(run, Path(args.out), feature_rows=run.feature_rows)
    print(
        f"matchday {result['matchday']}: {result['files']} files -> {result['out_dir']} "
        f"(predictor: {run.metadata['predictor']})"
    )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    conn = db.connect(args.db)
    try:
        run_fetch(
            _client(args),
            conn,
            snapshot_dir=args.snapshots,
            skip_performances=args.skip_performances,
            skip_market_values=args.skip_market_values,
        )
    finally:
        conn.close()
    return cmd_publish(args)


def cmd_validate(args: argparse.Namespace) -> int:
    conn = _connect(args)
    try:
        matrix = features.build_matrix(conn, max_seasons=args.seasons)
        result = walk_forward(
            matrix,
            season_id=args.season,
            first_matchday=args.first_matchday,
            last_matchday=args.last_matchday,
            fit_quantiles=args.quantiles,
        )
    finally:
        conn.close()
    print("\n" + result.report() + "\n")
    if args.out_csv:
        result.per_fold.to_csv(args.out_csv, index=False)
        print(f"per-fold results -> {args.out_csv}")
    return 0


def cmd_features(args: argparse.Namespace) -> int:
    conn = _connect(args)
    try:
        matrix = features.build_matrix(conn, max_seasons=args.seasons)
    finally:
        conn.close()
    if args.out_csv:
        matrix.to_csv(args.out_csv, index=False)
        print(f"{len(matrix)} rows -> {args.out_csv}")
    else:
        with pd.option_context("display.width", 200, "display.max_columns", 60):
            print(matrix.head(args.top).to_string(index=False))
    print(f"\n{len(matrix)} rows, {int(matrix['completed'].sum())} completed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="kickbase-xp", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--db", default=str(DEFAULT_DB_PATH),
                   help="working SQLite cache (rebuilt from the API; not committed)")
    p.add_argument("--snapshots", default=str(DEFAULT_SNAPSHOT_DIR),
                   help="committed daily status/market-value archive")
    sub = p.add_subparsers(dest="command", required=True)

    def add_api_args(sp):
        sp.add_argument("--delay", type=float, default=0.25,
                        help="seconds between API requests (courtesy throttle)")
        sp.add_argument("--competition", default=COMPETITION_BUNDESLIGA)
        sp.add_argument("--skip-performances", action="store_true")
        sp.add_argument("--skip-market-values", action="store_true")

    def add_model_args(sp):
        sp.add_argument("--seasons", type=int, default=features.DEFAULT_MAX_SEASONS,
                        help="how many recent seasons to train on (0 = all)")
        sp.add_argument("--matchday", type=int, default=None,
                        help="target matchday (default: the next one)")
        sp.add_argument("--predictor", choices=["two_stage_lgbm", "form_x_startshare"],
                        default=None, help="force a predictor instead of auto-selecting")
        sp.add_argument("--no-auto-select", action="store_true",
                        help="skip the recent-matchday model-vs-baseline check")

    sp = sub.add_parser("fetch", help="refresh the SQLite history from the Kickbase API")
    add_api_args(sp)
    sp.add_argument("--limit", type=int, default=None, help="only the first N players (debug)")
    sp.set_defaults(func=cmd_fetch)

    sp = sub.add_parser("predict", help="train and print the next matchday")
    add_model_args(sp)
    sp.add_argument("--top", type=int, default=25)
    sp.add_argument("--out-csv", default=None)
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("publish", help="train and write the static JSON API")
    add_model_args(sp)
    sp.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    sp.set_defaults(func=cmd_publish)

    sp = sub.add_parser("run", help="fetch + train + publish (the nightly job)")
    add_api_args(sp)
    add_model_args(sp)
    sp.add_argument("--out", default=str(DEFAULT_OUT_DIR))
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("validate", help="walk-forward MAE against the baselines")
    sp.add_argument("--seasons", type=int, default=features.DEFAULT_MAX_SEASONS)
    sp.add_argument("--season", default=None, help="Kickbase season id (default: latest)")
    sp.add_argument("--first-matchday", type=int, default=6)
    sp.add_argument("--last-matchday", type=int, default=None)
    sp.add_argument("--quantiles", action="store_true", help="also fit quantile models (slower)")
    sp.add_argument("--out-csv", default=None)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("features", help="build and inspect the feature matrix")
    sp.add_argument("--seasons", type=int, default=features.DEFAULT_MAX_SEASONS)
    sp.add_argument("--top", type=int, default=10)
    sp.add_argument("--out-csv", default=None)
    sp.set_defaults(func=cmd_features)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
