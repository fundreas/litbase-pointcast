"""OpenAPI 3.1 description of the published tree, served at `/v1/openapi.json`.

Built from the same constants the publisher uses, so enums cannot drift from
the output. The tests validate every published file against these schemas.

Schemas list their fields but do not forbid unknown ones: v1 only ever grows
by adding fields, and a consumer validating against this spec must not break
when it does.
"""

from __future__ import annotations

from typing import Any

from . import __version__
from .config import POSITIONS, STATUS_LABELS
from .lineups import FORMATION_SOURCES, FORMATION_WINDOW, TIER_DESCRIPTIONS, TIERS
from .rankings import TOP_N

OPENAPI_VERSION = "3.1.0"
SPEC_MEDIA_TYPE = "application/vnd.oai.openapi+json"


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    out = dict(schema)
    out["type"] = [schema["type"], "null"]
    if "enum" in out:
        out["enum"] = [*out["enum"], None]
    return out


def _ref(name: str) -> dict[str, str]:
    return {"$ref": f"#/components/schemas/{name}"}


def _obj(properties: dict[str, Any], *, description: str | None = None) -> dict[str, Any]:
    """An object whose listed fields are all always present (possibly null)."""
    schema: dict[str, Any] = {
        "type": "object",
        "required": list(properties),
        "properties": properties,
    }
    if description:
        schema["description"] = description
    return schema


STRING = {"type": "string"}
INTEGER = {"type": "integer"}
NUMBER = {"type": "number"}
BOOLEAN = {"type": "boolean"}
TIMESTAMP = {"type": "string", "format": "date-time", "examples": ["2026-10-03T22:24:10Z"]}
PROBABILITY = {"type": "number", "minimum": 0, "maximum": 1}
POSITION = {"type": "string", "enum": list(POSITIONS.values())}
API_VERSION_FIELD = {"type": "string", "const": "v1"}
SEASON_ID = {"type": "string", "description": "Kickbase season id. Global, not per competition."}


def _player_prediction() -> dict[str, Any]:
    return {
        "playerId": {**STRING, "examples": ["1685"]},
        "name": {**STRING, "examples": ["Joshua Kimmich"]},
        "teamId": _nullable(STRING),
        "teamName": _nullable(STRING),
        "opponentTeamId": _nullable(STRING),
        "isHome": _nullable(BOOLEAN),
        "position": _nullable(POSITION),
        "matchday": _nullable(INTEGER),
        "xP": _nullable(
            {**NUMBER, "description": "Expected points: P(plays) x E[points | plays]."}
        ),
        "p20": _nullable(
            {
                **NUMBER,
                "description": "20th percentile, including the chance of not playing. "
                "Null when the heuristic fallback predictor is active.",
            }
        ),
        "p50": _nullable({**NUMBER, "description": "Median. Null under the fallback predictor."}),
        "p80": _nullable(
            {**NUMBER, "description": "80th percentile (ceiling). Null under the fallback predictor."}
        ),
        **_probabilities(),
        "pointsGivenPlay": _nullable(
            {**NUMBER, "description": "Expected points if the player plays."}
        ),
        "status": _nullable(STATUS),
        "marketValue": _nullable({**INTEGER, "description": "Euros, as of the last fetch."}),
        "kickoff": _nullable(TIMESTAMP),
        "generatedAt": TIMESTAMP,
    }


STATUS = {**STRING, "enum": sorted({*STATUS_LABELS.values(), "unknown"})}
TIER = {
    **STRING,
    "enum": list(TIERS),
    "description": " ".join(f"`{t}`: {d}" for t, d in TIER_DESCRIPTIONS.items()),
}


def _probabilities() -> dict[str, Any]:
    """The three stage-1 probabilities, capped and raw. Shared by every file."""
    return {
        "pStart": _nullable(
            {
                **PROBABILITY,
                "description": "P(in the starting XI), capped by the player's status. "
                "Always <= pPlay.",
            }
        ),
        "pPlay": _nullable(
            {
                **PROBABILITY,
                "description": "P(plays at all), capped by the player's status. Always <= pSquad.",
            }
        ),
        "pPlayRaw": _nullable(
            {**PROBABILITY, "description": "P(plays) before the status cap. Always >= pPlay."}
        ),
        "pStartRaw": _nullable(
            {**PROBABILITY, "description": "P(starts) before the status cap. Always >= pStart."}
        ),
        "pSquad": _nullable(
            {
                **PROBABILITY,
                "description": "P(in the matchday squad), capped by the player's status.",
            }
        ),
        "pSquadRaw": _nullable(
            {**PROBABILITY, "description": "P(in the squad) before the status cap."}
        ),
    }


def _player_ref() -> dict[str, Any]:
    return _nullable(_obj({"playerId": STRING, "name": STRING}))


def _lineup_entry() -> dict[str, Any]:
    return _obj(
        {
            "playerId": {**STRING, "examples": ["1685"]},
            "name": STRING,
            "position": POSITION,
            "status": _nullable(STATUS),
            "tier": TIER,
            "depthRank": {
                **INTEGER,
                "minimum": 1,
                "description": "Place in the pecking order at this position: the expected "
                "starters first, then the bench, then absentees, each by pStart.",
            },
            "inLineup": {**BOOLEAN, "description": "In the expected starting XI."},
            **_probabilities(),
            "replaces": {
                **_player_ref(),
                "description": "For a bench player: the expected starter at the same "
                "position he would most likely replace (the one with the lowest pStart). "
                "Null for starters and absentees.",
            },
            "replacedBy": {
                **_player_ref(),
                "description": "For an expected starter: the next man up at the same "
                "position. Null when nobody fit is left behind him.",
            },
            "xP": _nullable(NUMBER),
            "p20": _nullable(NUMBER),
            "p80": _nullable(NUMBER),
            "marketValue": _nullable(INTEGER),
        },
        description="One squad player on a team sheet.",
    )


def _team_lineup() -> dict[str, Any]:
    entries = {"type": "array", "items": _ref("LineupEntry")}
    counts = _obj({label: {**INTEGER, "minimum": 0} for label in POSITIONS.values()})
    return _obj(
        {
            "apiVersion": API_VERSION_FIELD,
            "seasonId": SEASON_ID,
            "matchday": {**INTEGER, "minimum": 1},
            "generatedAt": TIMESTAMP,
            "predictor": _nullable(STRING),
            "teamId": STRING,
            "teamName": _nullable(STRING),
            "opponentTeamId": _nullable(STRING),
            "opponentTeamName": _nullable(STRING),
            "isHome": _nullable(BOOLEAN),
            "kickoff": _nullable(TIMESTAMP),
            "summary": _obj(
                {
                    "formation": {
                        **STRING,
                        "pattern": r"^\d+-\d+-\d+$",
                        "examples": ["4-4-2"],
                        "description": "Expected starters as DEF-MID-FWD, in Kickbase "
                        "positions (the goalkeeper is implied).",
                    },
                    "counts": counts,
                    "formationSource": {
                        **STRING,
                        "enum": list(FORMATION_SOURCES),
                        "description": "Where the usual shape came from: the team's own "
                        f"last {FORMATION_WINDOW} games, the league's, or a default.",
                    },
                    "usualFormation": {
                        **STRING,
                        "pattern": r"^\d+-\d+-\d+$",
                        "description": "The team's most common recent shape. `formation` "
                        "only deviates from it on clear evidence.",
                    },
                    "formationHistory": {
                        "type": "array",
                        "items": STRING,
                        "description": "Shapes of the team's recent complete lineups, most "
                        "recent first.",
                    },
                    "squadSize": {**INTEGER, "minimum": 0},
                    "confidence": _nullable(
                        {**PROBABILITY, "description": "Mean pStart over the expected XI."}
                    ),
                    "tiers": _obj({t: {**INTEGER, "minimum": 0} for t in TIERS}),
                }
            ),
            "lineup": {
                **_obj({label: entries for label in POSITIONS.values()}),
                "description": "The expected starting XI by position, in depth order.",
            },
            "bench": {
                **entries,
                "description": "Expected in the matchday squad but not in the XI, best first.",
            },
            "out": {
                **entries,
                "description": "Not expected in the matchday squad, best first.",
            },
        },
        description="One team's expected lineup for the predicted matchday.",
    )


def _ranking_entry(*, season: bool) -> dict[str, Any]:
    props: dict[str, Any] = {
        "rank": {
            **INTEGER,
            "minimum": 1,
            "description": "Competition rank within this list: ties share a rank (1, 1, 3).",
        },
        "playerId": STRING,
        "name": STRING,
        "teamId": _nullable(STRING),
        "teamName": _nullable(STRING),
        "position": _nullable(POSITION),
        "points": INTEGER,
        "minutes": _nullable(INTEGER),
    }
    if season:
        props["appearances"] = {
            **INTEGER,
            "minimum": 1,
            "description": "Matchdays the player scored points on.",
        }
        props["pointsPerAppearance"] = _nullable(NUMBER)
        description = (
            "Season totals through the file's matchday. `teamId` is the player's current club."
        )
    else:
        description = "Points on one matchday. `teamId` is the club the player played for."
    return _obj(props, description=description)


def _ranking_file(scope: str, entry: str) -> dict[str, Any]:
    ranked_list = {
        "type": "array",
        "items": _ref(entry),
        "description": f"Top {TOP_N}, best first. A tie at place {TOP_N} is kept whole, "
        "so the list can run a few entries long.",
    }
    return _obj(
        {
            "apiVersion": API_VERSION_FIELD,
            "seasonId": SEASON_ID,
            "scope": {"type": "string", "const": scope},
            **_matchday_status_props(),
            "top": {**INTEGER, "const": TOP_N},
            "generatedAt": TIMESTAMP,
            "overall": ranked_list,
            "byPosition": _obj({label: ranked_list for label in POSITIONS.values()}),
        }
    )


def _matchday_status_props() -> dict[str, Any]:
    return {
        "matchday": {**INTEGER, "minimum": 1},
        "complete": {
            **BOOLEAN,
            "description": "False while matches of this matchday are scheduled or live. "
            "Points of an incomplete matchday are partial.",
        },
        "matchesPlayed": {**INTEGER, "minimum": 0, "description": "Matches kicked off."},
        "matchesTotal": {**INTEGER, "minimum": 1},
    }


def _schemas() -> dict[str, Any]:
    prediction = _player_prediction()
    string_map = {"type": "object", "additionalProperties": {"type": "string"}}
    return {
        "PlayerPrediction": _obj(
            prediction, description="One player's forecast for the target matchday."
        ),
        "PlayerDetail": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                **prediction,
                "features": {
                    "type": "object",
                    "description": "Digest of the model inputs, e.g. `pts_mean_5` "
                    "(mean points over the last five matchdays).",
                    "additionalProperties": _nullable(NUMBER),
                },
            }
        ),
        "MatchdayPredictions": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                "seasonId": SEASON_ID,
                "matchday": INTEGER,
                "generatedAt": TIMESTAMP,
                "predictor": _nullable(STRING),
                "count": INTEGER,
                "players": {
                    "type": "array",
                    "items": _ref("PlayerPrediction"),
                    "description": "Every player with a fixture, sorted by xP, best first.",
                },
            }
        ),
        "PlayerIndex": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                "generatedAt": TIMESTAMP,
                "matchday": INTEGER,
                "players": {
                    "type": "array",
                    "items": _obj(
                        {
                            "playerId": STRING,
                            "name": STRING,
                            "teamId": _nullable(STRING),
                            "position": _nullable(POSITION),
                            "xP": _nullable(NUMBER),
                        }
                    ),
                },
            }
        ),
        "ApiIndex": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                "seasonId": SEASON_ID,
                "matchday": {**INTEGER, "description": "The matchday being predicted."},
                "generatedAt": TIMESTAMP,
                "playerCount": INTEGER,
                "predictor": _nullable(
                    {
                        **STRING,
                        "description": "`two_stage_lgbm`, or `form_x_startshare` when the "
                        "model lost to the heuristic on recent matchdays.",
                    }
                ),
                "model": _nullable({"type": "object"}),
                "topFeatures": _nullable({"type": "array", "items": {"type": "object"}}),
                "recentValidation": _nullable({"type": "array", "items": {"type": "object"}}),
                "endpoints": {
                    **string_map,
                    "description": "Path templates relative to the site root.",
                },
                "notes": string_map,
            }
        ),
        "MatchdayStatus": _obj(_matchday_status_props()),
        "RankingsIndex": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                "seasonId": SEASON_ID,
                "generatedAt": TIMESTAMP,
                "top": {**INTEGER, "const": TOP_N},
                "positions": {"type": "array", "items": POSITION},
                "latestMatchday": _nullable(
                    {**INTEGER, "description": "Null before the season's first kickoff."}
                ),
                "matchdays": {
                    "type": "array",
                    "items": _ref("MatchdayStatus"),
                    "description": "Every matchday with a ranking, oldest first.",
                },
                "endpoints": string_map,
            }
        ),
        "MatchdayRankingEntry": _ranking_entry(season=False),
        "SeasonRankingEntry": _ranking_entry(season=True),
        "MatchdayRanking": _ranking_file("matchday", "MatchdayRankingEntry"),
        "SeasonRanking": _ranking_file("season", "SeasonRankingEntry"),
        "LineupEntry": _lineup_entry(),
        "TeamLineup": _team_lineup(),
        "LineupsIndex": _obj(
            {
                "apiVersion": API_VERSION_FIELD,
                "seasonId": SEASON_ID,
                "matchday": {**INTEGER, "minimum": 1},
                "generatedAt": TIMESTAMP,
                "predictor": _nullable(STRING),
                "tiers": {"type": "array", "items": TIER},
                "positions": {"type": "array", "items": POSITION},
                "formationWindow": {**INTEGER, "const": FORMATION_WINDOW},
                "teams": {
                    "type": "array",
                    "description": "Every team with a fixture, by kickoff.",
                    "items": _obj(
                        {
                            "teamId": STRING,
                            "teamName": _nullable(STRING),
                            "opponentTeamId": _nullable(STRING),
                            "isHome": _nullable(BOOLEAN),
                            "kickoff": _nullable(TIMESTAMP),
                            "formation": STRING,
                            "formationSource": {**STRING, "enum": list(FORMATION_SOURCES)},
                            "confidence": _nullable(PROBABILITY),
                            "path": {**STRING, "description": "The team's lineup file."},
                        }
                    ),
                },
                "endpoints": string_map,
            }
        ),
    }


TEAM_PARAM = {
    "name": "teamId",
    "in": "path",
    "required": True,
    "schema": {"type": "string"},
    "example": "2",
}

MATCHDAY_PARAM = {
    "name": "matchday",
    "in": "path",
    "required": True,
    "schema": {"type": "integer", "minimum": 1},
}


def _get(
    operation_id: str,
    summary: str,
    schema: str,
    tag: str,
    *,
    description: str | None = None,
    params: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    responses: dict[str, Any] = {
        "200": {
            "description": "OK",
            "content": {"application/json": {"schema": _ref(schema)}},
        }
    }
    if params:
        responses["404"] = {"description": "No such file has been published."}
    op: dict[str, Any] = {
        "operationId": operation_id,
        "summary": summary,
        "tags": [tag],
        "responses": responses,
    }
    if description:
        op["description"] = description
    if params:
        op["parameters"] = params
    return {"get": op}


def _paths(v: str) -> dict[str, Any]:
    return {
        f"/{v}/index.json": _get(
            "getApiIndex", "API metadata, model and endpoints", "ApiIndex", "meta"
        ),
        f"/{v}/matchday/current.json": _get(
            "getCurrentPredictions",
            "Predictions for the next matchday",
            "MatchdayPredictions",
            "predictions",
        ),
        f"/{v}/matchday/{{matchday}}.json": _get(
            "getMatchdayPredictions",
            "Predictions by matchday number",
            "MatchdayPredictions",
            "predictions",
            description="Only the matchday currently being predicted is published; "
            "past predictions are not kept.",
            params=[MATCHDAY_PARAM],
        ),
        f"/{v}/players/index.json": _get(
            "getPlayerIndex", "Compact list of all predicted players", "PlayerIndex", "players"
        ),
        f"/{v}/players/{{playerId}}.json": _get(
            "getPlayer",
            "One player's prediction with a feature digest",
            "PlayerDetail",
            "players",
            params=[
                {
                    "name": "playerId",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "example": "1685",
                }
            ],
        ),
        f"/{v}/rankings/index.json": _get(
            "getRankingsIndex",
            "Matchdays that have rankings",
            "RankingsIndex",
            "rankings",
        ),
        f"/{v}/rankings/matchday/current.json": _get(
            "getCurrentMatchdayRanking",
            "Points ranking of the latest matchday that has kicked off",
            "MatchdayRanking",
            "rankings",
        ),
        f"/{v}/rankings/matchday/{{matchday}}.json": _get(
            "getMatchdayRanking",
            "Points ranking of one matchday",
            "MatchdayRanking",
            "rankings",
            params=[MATCHDAY_PARAM],
        ),
        f"/{v}/rankings/season/current.json": _get(
            "getCurrentSeasonRanking",
            "Season-to-date points ranking through the latest matchday",
            "SeasonRanking",
            "rankings",
        ),
        f"/{v}/rankings/season/{{matchday}}.json": _get(
            "getSeasonRanking",
            "Season-to-date points ranking through a matchday",
            "SeasonRanking",
            "rankings",
            description="Totals over matchdays 1 through `matchday`.",
            params=[MATCHDAY_PARAM],
        ),
        f"/{v}/lineups/index.json": _get(
            "getLineupsIndex",
            "Teams with an expected lineup for the predicted matchday",
            "LineupsIndex",
            "lineups",
        ),
        # Listed before the `{matchday}` template it would otherwise also match.
        f"/{v}/lineups/current/{{teamId}}.json": _get(
            "getCurrentTeamLineup",
            "One team's expected lineup for the next matchday",
            "TeamLineup",
            "lineups",
            params=[TEAM_PARAM],
        ),
        f"/{v}/lineups/{{matchday}}/{{teamId}}.json": _get(
            "getTeamLineup",
            "One team's expected lineup by matchday number",
            "TeamLineup",
            "lineups",
            description="Only the matchday currently being predicted is published.",
            params=[MATCHDAY_PARAM, TEAM_PARAM],
        ),
    }


def build_spec(api_version: str, *, base_url: str | None = None) -> dict[str, Any]:
    """The OpenAPI document for one API version.

    `base_url` is the site root as deployed (on a GitHub project page that
    includes the repository path). Without it the server is given relative
    to the document, which resolves to the site root too.
    """
    servers = []
    if base_url:
        servers.append({"url": base_url.rstrip("/"), "description": "GitHub Pages"})
    servers.append({"url": "..", "description": "Relative to this document"})
    return {
        "openapi": OPENAPI_VERSION,
        "info": {
            "title": "kickbase-xp",
            "version": __version__,
            "summary": "Expected and actual Kickbase points for the Bundesliga.",
            "description": (
                "Static JSON, rebuilt nightly around 22:30 UTC. Read-only and "
                "unauthenticated; every operation is a plain GET of a file.\n\n"
                "Kickbase points run roughly -100 to 600 per matchday. Fields are only "
                f"ever added within `/{api_version}/`; a breaking change moves to a new "
                "version prefix. Ignore fields you do not know."
            ),
        },
        "servers": servers,
        "tags": [
            {"name": "meta", "description": "API metadata."},
            {"name": "predictions", "description": "Expected points for the next matchday."},
            {"name": "players", "description": "Per-player predictions."},
            {
                "name": "rankings",
                "description": "Actual points of the running season, top "
                f"{TOP_N} overall and per position.",
            },
            {
                "name": "lineups",
                "description": "Expected starting XI, bench and absentees per team, with "
                "start/play/squad probabilities, tiers and rival links.",
            },
        ],
        "paths": _paths(api_version),
        "components": {"schemas": _schemas()},
    }
