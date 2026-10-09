"""Expected lineup per team: XI, bench, out -- with tiers and rival links.

The classifiers in `model.py` answer per player and independently, so their
start probabilities do not add up to eleven and say nothing about who
competes with whom. This module turns them into one coherent team sheet:

1. **Shape.** Kickbase only knows GK/DEF/MID/FWD, so a "formation" here is
   the number of starters per line (``4-4-2`` = 4 DEF, 4 MID, 2 FWD). Read
   off the team's own last few games of the *current* season -- older seasons
   are thinned by survivorship and show 8-10 starters -- falling back to the
   league's, then to a static default.
2. **Selection.** One goalkeeper, then the outfield shape within the observed
   range that maximises summed ``p_start``, minus a small penalty per player
   away from the usual shape so it only flexes on clear evidence.
3. **Tiers** from probability *and* rank: an XI member is never "bench", a
   player the shape has no room for is never better than "coin flip".
4. **Rivals.** Within a position the next man up replaces the weakest
   starter. Kickbase positions are the only grouping there is, so links never
   cross positions.

Compute only; `publish.py` serialises and `openapi.py` documents, both from
the constants defined here.
"""

from __future__ import annotations

import itertools
import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .config import POSITIONS

log = logging.getLogger(__name__)

TIERS: tuple[str, ...] = ("sure", "likely", "coin_flip", "bench", "out")
TIER_DESCRIPTIONS = {
    "sure": "In the expected XI and very likely to start.",
    "likely": "In the expected XI, a start is likely but not certain.",
    "coin_flip": "On the boundary: in the XI with a modest pStart, or outside it with a real chance.",
    "bench": "Expected in the matchday squad, but not in the starting XI.",
    "out": "Not expected in the matchday squad (status, or model).",
}

# Tier thresholds on pStart / pPlay / pSquad. Calibrated in the walk-forward
# (`validate.py`, docs/validation.md): "sure" should start ~9 times in 10,
# "out" should almost never play.
TIER_SURE = 0.85
TIER_LIKELY = 0.60
TIER_COIN = 0.35
OUT_PLAY = 0.05
OUT_SQUAD = 0.15

XI_SIZE = 11
GK, DEF, MID, FWD = 1, 2, 3, 4
OUTFIELD = (DEF, MID, FWD)

# Games of the team's own history that set its usual shape.
FORMATION_WINDOW = 5
# Summed pStart a shape must gain per player of deviation from the usual one.
FORMATION_PRIOR = 0.10
# Starters per line ever considered; also the default bounds on matchday 1.
LINE_BOUNDS: dict[int, tuple[int, int]] = {DEF: (3, 5), MID: (2, 6), FWD: (1, 3)}
DEFAULT_SHAPE: tuple[int, int, int] = (4, 4, 2)

FORMATION_SOURCES = ("team", "league", "default")


def formation_label(shape: tuple[int, int, int]) -> str:
    return "-".join(str(n) for n in shape)


# --------------------------------------------------------------------- shape


@dataclass(frozen=True)
class ShapeBounds:
    mode: tuple[int, int, int]
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]
    source: str
    history: tuple[str, ...] = ()


def _full_games(games: pd.DataFrame) -> pd.DataFrame:
    """Games with a complete XI: one GK and ten outfield starters on record."""
    if games.empty:
        return games
    total = games[[GK, DEF, MID, FWD]].sum(axis=1)
    return games[(total == XI_SIZE) & (games[GK] == 1)]


def _shape(row: pd.Series) -> tuple[int, int, int]:
    return (int(row.loc[DEF]), int(row.loc[MID]), int(row.loc[FWD]))


def _bounds_from(shapes: list[tuple[int, int, int]], source: str, history) -> ShapeBounds:
    # Most common shape; on a tie the most recent of the tied ones. `shapes`
    # runs oldest to newest.
    counts = Counter(shapes)
    top = max(counts.values())
    mode = next(s for s in reversed(shapes) if counts[s] == top)
    lo, hi = [], []
    for i, line in enumerate(OUTFIELD):
        seen = [s[i] for s in shapes]
        lo_l = min(min(seen), mode[i] - 1)
        hi_l = max(max(seen), mode[i] + 1)
        lo.append(max(lo_l, LINE_BOUNDS[line][0]))
        hi.append(min(hi_l, LINE_BOUNDS[line][1]))
    return ShapeBounds(mode, tuple(lo), tuple(hi), source, tuple(history))


def formation_bounds(
    games: pd.DataFrame,
    team_id: str,
    season_id: str,
    before: pd.Timestamp | None = None,
) -> ShapeBounds:
    """The team's usual shape and the range it may flex within.

    `games` is `features.team_games`. Only the current season counts, and
    only games that kicked off before `before`.
    """
    if games is None or games.empty or "season_id" not in games:
        return _default_bounds(())
    season = games[games["season_id"].astype(str) == str(season_id)]
    if before is not None and not pd.isna(before):
        season = season[pd.to_datetime(season["kickoff"], utc=True) < before]
    full = _full_games(season)

    own = full[full["team_id"].astype(str) == str(team_id)].sort_values("kickoff")
    own = own.tail(FORMATION_WINDOW)
    history = tuple(formation_label(_shape(r)) for _, r in own.iloc[::-1].iterrows())
    if len(own) >= 2:
        return _bounds_from([_shape(r) for _, r in own.iterrows()], "team", history)

    if not full.empty:
        recent_mds = sorted(full["matchday"].unique())[-FORMATION_WINDOW:]
        league = full[full["matchday"].isin(recent_mds)].sort_values("kickoff")
        if len(league) >= 2:
            return _bounds_from([_shape(r) for _, r in league.iterrows()], "league", history)

    return _default_bounds(history)


def _default_bounds(history: tuple[str, ...]) -> ShapeBounds:
    return ShapeBounds(
        DEFAULT_SHAPE,
        tuple(LINE_BOUNDS[line][0] for line in OUTFIELD),
        tuple(LINE_BOUNDS[line][1] for line in OUTFIELD),
        "default",
        history,
    )


# ----------------------------------------------------------------- selection


def ordered(players: pd.DataFrame) -> pd.DataFrame:
    """Best candidate first: pStart, then pPlay, then xP, then id -- stable."""
    key = players.assign(
        _s=-players["p_start"].fillna(0.0),
        _p=-players["p_play"].fillna(0.0),
        _x=-players["xP"].astype(float).fillna(-np.inf) if "xP" in players else 0.0,
        _id=players["player_id"].astype(str),
    )
    key = key.sort_values(["_s", "_p", "_x", "_id"], kind="mergesort")
    return players.loc[key.index]


def is_out(players: pd.DataFrame) -> pd.Series:
    p_squad = players["p_squad"] if "p_squad" in players else players["p_play"]
    return (players["p_play"].fillna(0.0) <= OUT_PLAY) | (p_squad.fillna(0.0) < OUT_SQUAD)


@dataclass
class Selection:
    xi: list[str]
    shape: tuple[int, int, int]
    gk: int


def assemble_team(
    players: pd.DataFrame, bounds: ShapeBounds, *, prior: float = FORMATION_PRIOR
) -> Selection:
    """Pick the XI. `players` needs player_id, position, p_start, p_play."""
    out = is_out(players)
    eligible = ordered(players[~out])
    by_line = {
        pos: eligible[eligible["position"] == pos]["player_id"].astype(str).tolist()
        for pos in (GK, *OUTFIELD)
    }
    p_start = dict(zip(players["player_id"].astype(str), players["p_start"].fillna(0.0)))

    gks = by_line[GK]
    if not gks:
        # Nobody fit in goal: still name the most likely keeper over an empty slot.
        gks = ordered(players[players["position"] == GK])["player_id"].astype(str).tolist()
        if not gks:
            # Routine in old seasons (survivorship), alarming in the live one.
            log.info("team without a goalkeeper on record: %s", players["team_id"].iloc[0])
    gk = gks[:1]

    avail = [len(by_line[line]) for line in OUTFIELD]
    slots = min(XI_SIZE - len(gk), sum(avail))

    def feasible(lo, hi):
        ranges = [range(max(0, lo[i]), max(0, hi[i]) + 1) for i in range(3)]
        return [t for t in itertools.product(*ranges) if sum(t) == slots]

    lo = tuple(min(bounds.lo[i], avail[i]) for i in range(3))
    hi = tuple(min(bounds.hi[i], avail[i]) for i in range(3))
    shapes = feasible(lo, hi) or feasible((0, 0, 0), tuple(avail))

    def score(t):
        total = sum(p_start[pid] for i, line in enumerate(OUTFIELD) for pid in by_line[line][: t[i]])
        deviation = sum(abs(t[i] - bounds.mode[i]) for i in range(3))
        return (-round(total - prior * deviation, 12), deviation, t)

    best = min(shapes, key=score) if shapes else (0, 0, 0)
    xi = gk + [pid for i, line in enumerate(OUTFIELD) for pid in by_line[line][: best[i]]]
    return Selection(xi=xi, shape=best, gk=len(gk))


# ------------------------------------------------------------- tiers, rivals


def assign_tiers(players: pd.DataFrame, xi: set[str]) -> pd.Series:
    """First matching rule wins; an XI member is floored at `coin_flip`."""
    ids = players["player_id"].astype(str)
    in_xi = ids.isin(xi)
    p_start = players["p_start"].fillna(0.0)
    tiers = np.select(
        [
            in_xi & (p_start >= TIER_SURE),
            in_xi & (p_start >= TIER_LIKELY),
            in_xi,
            is_out(players),
            p_start >= TIER_COIN,
        ],
        ["sure", "likely", "coin_flip", "out", "coin_flip"],
        default="bench",
    )
    return pd.Series(tiers, index=players.index)


def depth_order(players: pd.DataFrame) -> pd.DataFrame:
    """Per position: XI first, then the bench, then out -- each by `ordered`."""
    ranked = ordered(players)
    group = np.where(ranked["in_lineup"], 0, np.where(ranked["tier"] == "out", 2, 1))
    ranked = ranked.assign(_g=group).sort_values("_g", kind="mergesort").drop(columns="_g")
    ranked["depth_rank"] = ranked.groupby("position", sort=False).cumcount() + 1
    return ranked


def rival_links(players: pd.DataFrame) -> pd.DataFrame:
    """`replaces` / `replaced_by` ids, from a frame already in `depth_order`."""
    players = players.copy()
    players["replaces"] = None
    players["replaced_by"] = None
    for _, grp in players.groupby("position", sort=False):
        starters = grp[grp["in_lineup"]]
        reserves = grp[~grp["in_lineup"] & (grp["tier"] != "out")]
        next_up = str(reserves["player_id"].iloc[0]) if len(reserves) else None
        weakest = str(starters["player_id"].iloc[-1]) if len(starters) else None
        players.loc[starters.index, "replaced_by"] = next_up
        players.loc[reserves.index, "replaces"] = weakest
    return players


# ---------------------------------------------------------------- team sheet


@dataclass
class TeamLineup:
    team_id: str
    team_name: str | None
    match_id: str | None
    opponent_team_id: str | None
    opponent_team_name: str | None
    is_home: bool | None
    kickoff: Any
    shape: tuple[int, int, int]
    gk: int
    bounds: ShapeBounds
    # One row per squad player in depth order, with `in_lineup`, `tier`,
    # `depth_rank`, `replaces`, `replaced_by` added to the prediction columns.
    players: pd.DataFrame = field(repr=False)

    @property
    def formation(self) -> str:
        return formation_label(self.shape)

    @property
    def counts(self) -> dict[str, int]:
        return {
            POSITIONS[GK]: self.gk,
            **{POSITIONS[line]: self.shape[i] for i, line in enumerate(OUTFIELD)},
        }

    @property
    def xi(self) -> pd.DataFrame:
        return self.players[self.players["in_lineup"]]

    @property
    def confidence(self) -> float | None:
        xi = self.xi
        return float(xi["p_start"].mean()) if len(xi) else None

    @property
    def tier_counts(self) -> dict[str, int]:
        counts = self.players["tier"].value_counts()
        return {t: int(counts.get(t, 0)) for t in TIERS}


@dataclass
class MatchdayLineups:
    season_id: str
    matchday: int
    teams: list[TeamLineup]


def _mode(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    counts = s.astype(str).value_counts()
    top = counts[counts == counts.max()].index
    return sorted(top)[0]


def build_team(
    players: pd.DataFrame,
    games: pd.DataFrame,
    season_id: str,
    *,
    team_names: dict[str, str] | None = None,
) -> TeamLineup:
    """One team's sheet from its prediction rows (one per squad player)."""
    team_id = str(players["team_id"].iloc[0])
    players = players.copy()
    players["position"] = pd.to_numeric(players["position"], errors="coerce")
    players = players[players["position"].isin([GK, *OUTFIELD])]

    # The team's fixture is the mode over its rows: a fresh transfer can still
    # carry his old club's fixture for a night.
    match_id = _mode(players["match_id"]) if "match_id" in players else None
    fixture = players[players["match_id"].astype(str) == match_id] if match_id else players
    first = fixture.sort_values("player_id").iloc[0]
    kickoff = pd.to_datetime(fixture["kickoff"], utc=True).min() if "kickoff" in fixture else None
    opponent = _mode(fixture["opponent_team_id"]) if "opponent_team_id" in fixture else None
    names = team_names or {}

    bounds = formation_bounds(games, team_id, season_id, before=kickoff)
    selection = assemble_team(players, bounds)
    xi = set(selection.xi)

    players["in_lineup"] = players["player_id"].astype(str).isin(xi)
    players["tier"] = assign_tiers(players, xi)
    players = rival_links(depth_order(players))

    team_name = _mode(players["team_name"]) if "team_name" in players else None
    return TeamLineup(
        team_id=team_id,
        team_name=team_name or names.get(team_id),
        match_id=match_id,
        opponent_team_id=opponent,
        opponent_team_name=names.get(opponent) if opponent else None,
        is_home=bool(first["is_home"]) if pd.notna(first.get("is_home")) else None,
        kickoff=kickoff,
        shape=selection.shape,
        gk=selection.gk,
        bounds=bounds,
        players=players,
    )


def build_lineups(
    predictions: pd.DataFrame,
    games: pd.DataFrame,
    season_id: str,
    matchday: int,
) -> MatchdayLineups:
    """Every team with a fixture on `matchday`, ordered by kickoff then id."""
    if predictions.empty:
        return MatchdayLineups(str(season_id), int(matchday), [])
    preds = predictions[predictions["team_id"].notna()].copy()
    preds["team_id"] = preds["team_id"].astype(str)
    team_names: dict[str, str] = {}
    if "team_name" in preds:
        for tid, grp in preds.groupby("team_id"):
            name = _mode(grp["team_name"])
            if name:
                team_names[tid] = name

    teams = [
        build_team(grp, games, season_id, team_names=team_names)
        for _, grp in preds.groupby("team_id", sort=False)
    ]
    teams.sort(key=lambda t: (_kickoff_key(t.kickoff), _numeric_id(t.team_id)))
    return MatchdayLineups(str(season_id), int(matchday), teams)


def _kickoff_key(kickoff) -> float:
    if kickoff is None or pd.isna(kickoff):
        return float("inf")
    return pd.Timestamp(kickoff).timestamp()


def _numeric_id(value: str) -> tuple[int, str]:
    return (int(value), value) if str(value).isdigit() else (10**9, str(value))


def lineups_for_run(run) -> MatchdayLineups:
    """`build_lineups` over a `train.PredictionRun`."""
    return build_lineups(run.predictions, run.team_games, run.season_id, run.matchday)
