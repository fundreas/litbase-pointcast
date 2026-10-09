"""Team assembly: shape, selection, tiers and rival links on fabricated squads."""

from __future__ import annotations

import itertools

import pandas as pd
import pytest

from kickbase_xp import lineups as lu
from kickbase_xp.lineups import DEF, FWD, GK, MID

SEASON = "42"
KICKOFF = pd.Timestamp("2026-10-10T13:30:00Z")


def _squad(team: str = "1", spec: dict[int, list[float]] | None = None, **extra) -> pd.DataFrame:
    """One row per player; `spec` maps position -> list of p_start values."""
    spec = spec or {
        GK: [0.97, 0.03, 0.01],
        DEF: [0.95, 0.92, 0.88, 0.70, 0.30, 0.10],
        MID: [0.96, 0.90, 0.65, 0.55, 0.40, 0.12],
        FWD: [0.93, 0.58, 0.33, 0.08],
    }
    rows = []
    for pos, values in spec.items():
        for i, p in enumerate(values):
            rows.append(
                {
                    "player_id": f"{team}-{pos}-{i}",
                    "team_id": team,
                    "team_name": f"Team {team}",
                    "position": pos,
                    "p_start": p,
                    "p_play": min(1.0, p + 0.25),
                    "p_squad": min(1.0, p + 0.5),
                    "xP": 100 * p,
                    "match_id": "m1",
                    "opponent_team_id": "2" if team == "1" else "1",
                    "is_home": team == "1",
                    "kickoff": KICKOFF,
                    **extra,
                }
            )
    return pd.DataFrame(rows)


def _games(team: str, shapes: list[tuple[int, int, int]], season: str = SEASON) -> pd.DataFrame:
    rows = []
    for md, (d, m, f) in enumerate(shapes, 1):
        rows.append(
            {
                "team_id": team,
                "season_id": season,
                "matchday": md,
                "kickoff": KICKOFF - pd.Timedelta(days=7 * (len(shapes) - md + 1)),
                "xi": frozenset(),
                GK: 1, DEF: d, MID: m, FWD: f,
            }
        )
    return pd.DataFrame(rows)


NO_GAMES = _games("1", [])


# --------------------------------------------------------------------- shape


def test_shape_comes_from_the_teams_own_recent_games():
    games = _games("1", [(4, 4, 2), (4, 3, 3), (4, 4, 2)])
    b = lu.formation_bounds(games, "1", SEASON)
    assert b.source == "team"
    assert b.mode == (4, 4, 2)
    assert b.history == ("4-4-2", "4-3-3", "4-4-2")  # most recent first
    assert b.lo == (3, 3, 1) and b.hi == (5, 5, 3)


def test_shape_tie_goes_to_the_most_recent():
    games = _games("1", [(4, 4, 2), (5, 3, 2)])
    assert lu.formation_bounds(games, "1", SEASON).mode == (5, 3, 2)


def test_shape_falls_back_to_the_league_then_to_the_default():
    other = _games("2", [(5, 4, 1), (5, 4, 1), (4, 4, 2)])
    b = lu.formation_bounds(other, "1", SEASON)
    assert b.source == "league" and b.mode == (5, 4, 1)
    d = lu.formation_bounds(NO_GAMES, "1", SEASON)
    assert d.source == "default" and d.mode == lu.DEFAULT_SHAPE


def test_older_seasons_and_incomplete_xis_do_not_set_the_shape():
    old = _games("1", [(3, 5, 2), (3, 5, 2)], season="34")
    short = _games("1", [(3, 3, 2), (3, 3, 2)])  # 9 starters: survivorship gaps
    b = lu.formation_bounds(pd.concat([old, short]), "1", SEASON)
    assert b.source == "default"


# ----------------------------------------------------------------- selection


def test_a_full_squad_yields_eleven_with_one_keeper():
    team = lu.build_team(_squad(), _games("1", [(4, 4, 2)] * 3), SEASON)
    xi = team.xi
    assert len(xi) == 11
    assert (xi["position"] == GK).sum() == 1
    assert team.formation == "4-4-2"
    assert team.counts == {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}


def test_selection_matches_a_brute_force_over_shapes():
    squad = _squad()
    bounds = lu.formation_bounds(_games("1", [(4, 4, 2), (4, 5, 1), (5, 3, 2)]), "1", SEASON)
    got = lu.assemble_team(squad, bounds)

    def total(t):
        s = 0.0
        for i, pos in enumerate((DEF, MID, FWD)):
            s += squad[squad["position"] == pos]["p_start"].nlargest(t[i]).sum()
        return s - lu.FORMATION_PRIOR * sum(abs(t[i] - bounds.mode[i]) for i in range(3))

    shapes = [
        t
        for t in itertools.product(*(range(bounds.lo[i], bounds.hi[i] + 1) for i in range(3)))
        if sum(t) == 10
    ]
    best = max(total(t) for t in shapes)
    assert total(got.shape) == pytest.approx(best)


@pytest.mark.parametrize(("fourth_mid", "expected"), [(0.80, (4, 4, 2)), (0.60, (5, 3, 2))])
def test_the_shape_flexes_only_on_clear_evidence(fourth_mid, expected):
    spec = {
        GK: [0.97],
        DEF: [0.99, 0.98, 0.97, 0.96, 0.95],
        MID: [0.99, 0.98, 0.97, fourth_mid],
        FWD: [0.99, 0.98, 0.10],
    }
    bounds = lu.formation_bounds(_games("1", [(4, 4, 2)] * 3), "1", SEASON)
    assert lu.assemble_team(_squad(spec=spec), bounds).shape == expected


def test_selection_is_deterministic_under_row_order():
    squad = _squad()
    games = _games("1", [(4, 4, 2)] * 3)
    a = lu.build_team(squad, games, SEASON)
    b = lu.build_team(squad.sample(frac=1.0, random_state=7), games, SEASON)
    assert list(a.players["player_id"]) == list(b.players["player_id"])
    assert list(a.players["tier"]) == list(b.players["tier"])


def test_ties_break_by_player_id():
    spec = {GK: [0.9, 0.9], DEF: [0.5] * 5, MID: [0.5] * 5, FWD: [0.5] * 3}
    team = lu.build_team(_squad(spec=spec), NO_GAMES, SEASON)
    gk = team.xi[team.xi["position"] == GK]["player_id"].tolist()
    assert gk == ["1-1-0"]


def test_unavailable_players_are_never_picked_and_the_shape_adapts():
    squad = _squad()
    # Suspend all but three defenders: a back four is no longer possible.
    defs = squad["position"] == DEF
    suspended = defs & squad["player_id"].isin(["1-2-0", "1-2-1", "1-2-2"])
    squad.loc[suspended, ["p_start", "p_play", "p_squad"]] = 0.0
    team = lu.build_team(squad, _games("1", [(4, 4, 2)] * 3), SEASON)
    assert not set(team.xi["player_id"]) & set(squad.loc[suspended, "player_id"])
    assert team.counts["DEF"] == 3
    assert len(team.xi) == 11
    assert (team.players.loc[team.players["player_id"].isin(squad.loc[suspended, "player_id"]), "tier"] == "out").all()


def test_a_tiny_squad_yields_a_tiny_xi():
    spec = {GK: [0.9], DEF: [0.9], MID: [0.9], FWD: [0.9]}
    team = lu.build_team(_squad(spec=spec), NO_GAMES, SEASON)
    assert len(team.xi) == 4
    assert team.counts == {"GK": 1, "DEF": 1, "MID": 1, "FWD": 1}


# --------------------------------------------------------------------- tiers


def _tiers(team: lu.TeamLineup) -> dict[str, str]:
    return dict(zip(team.players["player_id"], team.players["tier"]))


def test_tier_rules():
    team = lu.build_team(_squad(), _games("1", [(4, 4, 2)] * 3), SEASON)
    t = _tiers(team)
    assert t["1-1-0"] == "sure"  # GK 0.97
    assert t["1-2-3"] == "likely"  # 4th DEF 0.70, in the XI
    assert t["1-3-3"] == "coin_flip"  # 4th MID 0.55, in the XI: floored
    assert t["1-3-4"] == "coin_flip"  # 5th MID 0.40, outside: capped
    assert t["1-2-4"] == "bench"  # 5th DEF 0.30
    assert t["1-2-5"] == "bench"  # 0.10, but still likely in the squad


def test_out_tier_follows_the_squad_and_play_probabilities():
    squad = _squad()
    squad.loc[squad["player_id"] == "1-4-3", ["p_play", "p_squad"]] = [0.5, 0.10]
    squad.loc[squad["player_id"] == "1-2-5", ["p_play"]] = 0.02
    t = _tiers(lu.build_team(squad, NO_GAMES, SEASON))
    assert t["1-4-3"] == "out"  # p_squad below OUT_SQUAD
    assert t["1-2-5"] == "out"  # p_play below OUT_PLAY


def test_tiers_and_depth_are_monotone_within_a_position():
    team = lu.build_team(_squad(), _games("1", [(4, 4, 2)] * 3), SEASON)
    order = {t: i for i, t in enumerate(lu.TIERS)}
    for _, grp in team.players.groupby("position"):
        grp = grp.sort_values("depth_rank")
        assert list(grp["depth_rank"]) == list(range(1, len(grp) + 1))
        ranks = [order[t] for t in grp["tier"]]
        assert ranks == sorted(ranks)
        assert list(grp["in_lineup"]) == sorted(grp["in_lineup"], reverse=True)


def test_tier_counts_cover_every_player():
    team = lu.build_team(_squad(), NO_GAMES, SEASON)
    assert sum(team.tier_counts.values()) == len(team.players)
    assert set(team.tier_counts) == set(lu.TIERS)


# -------------------------------------------------------------------- rivals


def test_rival_links_point_at_the_next_man_up_and_the_weakest_starter():
    team = lu.build_team(_squad(), _games("1", [(4, 4, 2)] * 3), SEASON)
    p = team.players.set_index("player_id")
    # Defence: XI is DEF 0..3; next man up is DEF 4.
    for pid in ("1-2-0", "1-2-1", "1-2-2", "1-2-3"):
        assert p.loc[pid, "replaced_by"] == "1-2-4"
        assert p.loc[pid, "replaces"] is None
    assert p.loc["1-2-4", "replaces"] == "1-2-3"
    # Keepers: the number two backs up the number one.
    assert p.loc["1-1-0", "replaced_by"] == "1-1-1"
    assert p.loc["1-1-1", "replaces"] == "1-1-0"


def test_out_players_are_never_rivals():
    squad = _squad()
    squad.loc[squad["position"] == FWD, "p_squad"] = [1.0, 1.0, 0.05, 0.05]
    team = lu.build_team(squad, _games("1", [(4, 4, 2)] * 3), SEASON)
    p = team.players.set_index("player_id")
    for pid in ("1-4-2", "1-4-3"):
        assert p.loc[pid, "tier"] == "out"
        assert p.loc[pid, "replaces"] is None
    # Both forwards start, nobody fit is left: no next man up.
    assert p.loc["1-4-0", "replaced_by"] is None


# --------------------------------------------------------------------- build


def test_build_lineups_covers_every_team_in_kickoff_order():
    late = _squad("2")
    late["kickoff"] = KICKOFF + pd.Timedelta(hours=2)
    preds = pd.concat([late, _squad("10"), _squad("1")], ignore_index=True)
    result = lu.build_lineups(preds, NO_GAMES, SEASON, 5)
    assert [t.team_id for t in result.teams] == ["1", "10", "2"]
    first = result.teams[0]
    assert first.opponent_team_id == "2"
    assert first.opponent_team_name == "Team 2"
    assert first.is_home is True
    assert first.confidence == pytest.approx(first.xi["p_start"].mean())


def test_no_predictions_means_no_teams():
    empty = lu.build_lineups(pd.DataFrame(), NO_GAMES, SEASON, 5)
    assert empty.teams == []
