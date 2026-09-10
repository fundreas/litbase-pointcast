"""Milestone 5 -- write the static JSON API.

The published tree is the product. It is versioned under `/v1/` from the very
first run: consumers (the Kickbase web app first) pin a path, and a schema
change later becomes `/v2/` rather than a silent breakage.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .train import PredictionRun

log = logging.getLogger(__name__)

API_VERSION = "v1"


def _num(value: Any, digits: int = 1) -> float | None:
    """JSON-safe rounding. NaN/Inf become null rather than invalid JSON."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f):
        return None
    # `+ 0.0` folds -0.0 into 0.0; a published "-0.0" reads like a bug.
    return round(f, digits) + 0.0


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return int(f) if math.isfinite(f) else None


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=False),
        encoding="utf-8",
    )


def _player_name(row: pd.Series) -> str:
    first = (row.get("first_name") or "").strip()
    last = (row.get("last_name") or "").strip()
    return f"{first} {last}".strip() or last or str(row["player_id"])


def player_entry(row: pd.Series, generated_at: str) -> dict[str, Any]:
    """One player as published. Field names are part of the v1 contract."""
    return {
        "playerId": str(row["player_id"]),
        "name": _player_name(row),
        "teamId": str(row["team_id"]) if row.get("team_id") else None,
        "teamName": row.get("team_name"),
        "opponentTeamId": str(row["opponent_team_id"]) if row.get("opponent_team_id") else None,
        "isHome": bool(row.get("is_home")) if row.get("is_home") is not None else None,
        "position": row.get("position_label"),
        "matchday": _int(row.get("matchday")),
        "xP": _num(row.get("xP")),
        "p20": _num(row.get("p20")),
        "p50": _num(row.get("p50")),
        "p80": _num(row.get("p80")),
        "pStart": _num(row.get("p_start"), 3),
        "pPlay": _num(row.get("p_play"), 3),
        # Before the status override, so a consumer can tell "the model thinks
        # he is a starter but he is injured" from "the model benched him".
        "pPlayRaw": _num(row.get("p_play_raw"), 3),
        "pointsGivenPlay": _num(row.get("points_given_play")),
        "status": row.get("status_label"),
        "marketValue": _int(row.get("market_value")),
        "kickoff": _kickoff(row.get("kickoff")),
        "generatedAt": generated_at,
    }


def _kickoff(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    if ts is pd.NaT or pd.isna(ts):
        return None
    return ts.strftime("%Y-%m-%dT%H:%M:%SZ")


def _player_detail(row: pd.Series, entry: dict[str, Any], features: dict[str, Any]) -> dict:
    return {**entry, "features": features}


FEATURE_DIGEST = [
    "pts_mean_5",
    "pts_app_mean_5",
    "min_mean_5",
    "start_share_5",
    "play_share_5",
    "pts_std_5",
    "season_pts_mean",
    "mv_trend_7",
    "opp_allowed_pos",
    "team_scored_all",
]


def publish(
    run: PredictionRun,
    out_dir: Path,
    *,
    feature_rows: pd.DataFrame | None = None,
    clean: bool = True,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    version_dir = out_dir / API_VERSION
    if clean and version_dir.exists():
        shutil.rmtree(version_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # GitHub Pages must serve the tree verbatim, no Jekyll processing.
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")

    generated_at = run.metadata.get("generated_at") or datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    df = run.predictions
    entries = [player_entry(row, generated_at) for _, row in df.iterrows()]

    matchday_payload = {
        "apiVersion": API_VERSION,
        "seasonId": str(run.season_id),
        "matchday": run.matchday,
        "generatedAt": generated_at,
        "predictor": run.metadata.get("predictor"),
        "count": len(entries),
        "players": entries,
    }
    _write(version_dir / "matchday" / f"{run.matchday}.json", matchday_payload)
    _write(version_dir / "matchday" / "current.json", matchday_payload)

    feature_lookup: dict[str, dict[str, Any]] = {}
    if feature_rows is not None and not feature_rows.empty:
        for _, frow in feature_rows.iterrows():
            feature_lookup[str(frow["player_id"])] = {
                k: _num(frow.get(k), 3) for k in FEATURE_DIGEST if k in frow
            }

    for entry in entries:
        pid = entry["playerId"]
        _write(
            version_dir / "players" / f"{pid}.json",
            {
                "apiVersion": API_VERSION,
                **entry,
                "features": feature_lookup.get(pid, {}),
            },
        )

    _write(
        version_dir / "players" / "index.json",
        {
            "apiVersion": API_VERSION,
            "generatedAt": generated_at,
            "matchday": run.matchday,
            "players": [
                {
                    "playerId": e["playerId"],
                    "name": e["name"],
                    "teamId": e["teamId"],
                    "position": e["position"],
                    "xP": e["xP"],
                }
                for e in entries
            ],
        },
    )

    index_payload = {
        "apiVersion": API_VERSION,
        "seasonId": str(run.season_id),
        "matchday": run.matchday,
        "generatedAt": generated_at,
        "playerCount": len(entries),
        "predictor": run.metadata.get("predictor"),
        "model": run.metadata.get("model"),
        "topFeatures": run.metadata.get("top_features"),
        "recentValidation": run.metadata.get("selection"),
        "endpoints": {
            "index": f"/{API_VERSION}/index.json",
            "currentMatchday": f"/{API_VERSION}/matchday/current.json",
            "matchday": f"/{API_VERSION}/matchday/{{matchday}}.json",
            "playerIndex": f"/{API_VERSION}/players/index.json",
            "player": f"/{API_VERSION}/players/{{playerId}}.json",
        },
        "notes": {
            "pointsScale": "Kickbase points, roughly -100..600 per matchday.",
            "xP": "P(plays) x E[points | plays].",
            "p20/p80": "Unconditional floor/ceiling: includes the chance of not playing.",
            "source": "Kickbase v4 API only -- no odds, xG or external lineup feeds.",
        },
    }
    _write(version_dir / "index.json", index_payload)
    _write(out_dir / "index.json", index_payload)
    (out_dir / "index.html").write_text(_landing_page(index_payload), encoding="utf-8")

    written = sum(1 for _ in version_dir.rglob("*.json"))
    log.info("published %d JSON files to %s", written, out_dir)
    return {"files": written, "out_dir": str(out_dir), "matchday": run.matchday}


def _landing_page(index_payload: dict[str, Any]) -> str:
    md = index_payload["matchday"]
    generated = index_payload["generatedAt"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>kickbase-xp — expected points API</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.6 ui-sans-serif, system-ui, sans-serif; max-width: 46rem;
         margin: 0 auto; padding: 2.5rem 1.25rem; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, monospace; font-size: .9em; }}
  pre {{ padding: .85rem 1rem; border-radius: .5rem; overflow-x: auto;
         background: color-mix(in srgb, currentColor 8%, transparent); }}
  h1 {{ font-size: 1.6rem; margin-bottom: .25rem; }}
  .sub {{ opacity: .7; margin-top: 0; }}
  table {{ border-collapse: collapse; width: 100%; }}
  td, th {{ text-align: left; padding: .35rem .6rem .35rem 0;
            border-bottom: 1px solid color-mix(in srgb, currentColor 15%, transparent); }}
</style></head><body>
<h1>kickbase-xp</h1>
<p class="sub">Expected Kickbase points for the next Bundesliga matchday.
Currently matchday <strong>{md}</strong>, generated {generated}.</p>
<p>Static JSON, rebuilt nightly, versioned under <code>/v1/</code>. Everything is
derived from the Kickbase v4 API alone — no betting odds, xG or external
lineup predictors.</p>
<h2>Endpoints</h2>
<table>
<tr><th>Path</th><th>Contents</th></tr>
<tr><td><code>/v1/index.json</code></td><td>matchday, model metadata, recent validation</td></tr>
<tr><td><code>/v1/matchday/current.json</code></td><td>all players, sorted by xP</td></tr>
<tr><td><code>/v1/matchday/{{md}}.json</code></td><td>a specific matchday</td></tr>
<tr><td><code>/v1/players/index.json</code></td><td>compact player list</td></tr>
<tr><td><code>/v1/players/{{playerId}}.json</code></td><td>one player incl. key features</td></tr>
</table>
<h2>Reading the numbers</h2>
<pre>xP   = P(plays) × E[points | plays]
p20  = pessimistic case, includes the chance of not playing at all
p80  = ceiling
pStart = probability of starting (≥60 minutes)</pre>
<p>Kickbase points run roughly −100…600 per matchday, so an xP of 140 is a solid
performance, not a typo.</p>
</body></html>
"""
