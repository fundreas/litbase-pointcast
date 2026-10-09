# Expected lineups per team: `/v1/lineups/…`

## Context

The feed already publishes `pStart`/`pPlay` per player, but nothing team-level: the probabilities are independent per player and do not add up to eleven, so a consumer cannot read an expected XI off them. The user wants, for the upcoming matchday, one JSON per team with the expected lineup and bench, start/play probabilities, a five-level tier (pretty sure / quite sure / 50:50 / benched / not in squad), and substitution links ("B would replace A"), plus the OpenAPI spec updated.

Facts established in this session that the plan builds on:

- `performances.perf_status` (the feed's per-match `st`) is a real lineup label back to 2013/14: **5 = started** (exactly 11 per team-matchday in season 42), **3 = came on as sub**, **4 = in matchday squad, unused**, **1 = not in squad**, **0 = not yet played**. Never NULL. Today `started` is derived as `minutes >= 60` ([features.py:166-169](src/kickbase_xp/features.py#L166-L169)), which mislabels ~5% of starters; `config.PERF_STATUS_PLAYED = 5` is documented as "took part" (wrong, it means started) and is unused. `load_raw` does not even select `perf_status`.
- Season 14 (2021/22) has ~400 rows with `st=4` but real minutes, so `played` must be `st in (3,5) OR minutes > 0`. Seasons ≤ 13 emit almost no `st=1/4` rows (not-in-squad matchdays simply have no row). Both are outside the default 4-season window.
- Earlier seasons suffer survivorship bias (only players still in the league are replayed): last season's teams show 8-10 starters per matchday. **Formation counts must come from the current season**, with league-wide and static fallbacks.
- Kickbase positions are only GK/DEF/MID/FWD. No tactical formation is derivable; "formation" means DEF-MID-FWD counts in Kickbase terms. Season 42 starter tuples so far: 1-4-4-2 ×14, 1-4-5-1 ×9, 1-5-3-2 ×5, 1-5-4-1 ×3, 1-4-3-3 ×2, others once. Ranges: DEF 3-5, MID 2-6, FWD 1-3.
- P(start | started last match) ≈ 0.78-0.80 vs ≈ 0.20 otherwise; from the bench 0.09, from a sub appearance 0.28, from out-of-squad 0.00. XI changes between consecutive matchdays average ~1.8, so "repeat last XI" is the bar to clear.
- Kickbase's own `lineup_prob` tier (LigaInsider) was strongly predictive but vanished from the API on 2026-09-17 (team profile and player detail both return null now). Snapshots 2026-09-10..16 still hold it; usable only as a one-off benchmark for matchday 3. Not a design input.
- Status (injured/suspended/questionable) is only known for today and is applied as a post-hoc cap in [model.py:46-53](src/kickbase_xp/model.py#L46-L53). Keep that mechanism.
- Squads on the scheduled matchday are 23-29 players with 2-3 GK. A few players have `player_team_id != players.team_id` (transfers); scheduled rows already use the current club, so group by `team_id` and take the team's fixture as the mode over its rows.
- Local `data/history.sqlite` was last fetched 2026-09-10; matchdays 3-4 sit there as completed with `perf_status=0`. Treat `perf_status == 0` on a completed row as *unknown label* (NaN), never as "did not play".

User decisions: five tiers; **upcoming matchday only** (no actual-lineup files for finished matchdays); per-position depth ordering **with explicit rival links**.

## Design summary

```
labels (perf_status) -> 3 classifiers (squad / play / start) -> per-team assembly
   -> tiers + depth ranks + rival links -> /v1/lineups/{md}/{teamId}.json
```

Pattern: compute in a new `src/kickbase_xp/lineups.py` (like `rankings.py`), serialise in `publish.py`, constants imported by `openapi.py` so spec and output cannot drift.

### 1. Labels (features.py, config.py)

- `config.py`: replace `PERF_STATUS_PLAYED` with `PERF_STATUS_STARTED = 5`, `PERF_STATUS_SUB = 3`, `PERF_STATUS_BENCH = 4`, `PERF_STATUS_OUT = 1`, corrected comment.
- `features.load_raw`: select `perf_status`.
- `features._prepare_base`: for completed rows with `perf_status != 0`:
  `started = st == 5`; `played = st in (3,5) or minutes > 0`; `in_squad = st in (3,4,5) or minutes > 0`. For completed rows with `st == 0` and minutes present, fall back to the minutes proxy; with neither, all three labels NaN (unknown). `points_app` unchanged. Add `in_squad` and `perf_status` to `ID_COLUMNS`.
- `model.TwoStageModel.fit`: `dropna` on all three label columns.
- Effect: `p_play`/`xP` effectively unchanged inside the default window; `p_start` moves up for early-subbed starters.

### 2. New features (features.py, all leak-free via shift(1) or `merge_asof(allow_exact_matches=False)`)

Per player (`_add_player_history`): `started_last`, `played_last`, `squad_last`, `start_streak` (consecutive starts strictly before the row; NaN breaks it), `squad_share_5`.

Team context (new `_add_team_context`):
- `start_rank_pos`: rank (desc, method `min`) of the already-shifted `start_share_5` within `(season_id, matchday, team_id, position)`. Cross-player but only consumes shifted columns, so leak-free.
- `team_pos_starters_5`: mean starters the team fielded at this position over its last 5 completed matchdays before this kickoff. Built from a per-`(season_id, matchday, team_id, position)` tally of `started == 1`, rolled by 5, attached with `pd.merge_asof(on="kickoff", by=["team_id","position"], direction="backward", allow_exact_matches=False)`.
- `start_rank_vs_slots = start_rank_pos - team_pos_starters_5`.
- `team_rotation_5`: mean over the last 5 team-matchdays of `1 - |XI_t ∩ XI_{t-1}| / |XI_t|`, same merge_asof.

All added to `FEATURE_COLUMNS`. `publish.FEATURE_DIGEST` gains `start_rank_pos`, `team_pos_starters_5`.

### 3. Third classifier (model.py, train.py)

Keep independent binaries (chosen over 4-class multiclass: `pStart`/`pPlay` keep their meaning, the status cap stays a one-liner, `xP` consumers untouched, noisy `sub` class isolated).

- `squad_model` next to `play_model`/`start_model`; outputs `p_squad_raw`, `p_squad`, `p_start_raw` (currently not exported).
- Cap then chain: `p_squad = min(p_squad_raw, cap)`, `p_play = min(p_play_raw, cap, p_squad)`, `p_start = min(p_start_raw, cap, p_play)`.
- `feature_importance` covers `p_squad`; `train` metadata `stages` adds `p_squad`.
- `PredictionRun` gains `formation_history: pd.DataFrame` (one row per completed team-matchday of the current season before the target: `team_id, matchday, kickoff, gk, def, mid, fwd, xi` (set of player ids)), built in `run_training` from `matrix[started == 1]` via a `lineups.formation_history(matrix, season_id, before=)` helper that the walk-forward reuses per fold.

### 4. Assembly, tiers, rivals (new `src/kickbase_xp/lineups.py`)

Constants (imported by openapi.py): `TIERS = ("sure", "likely", "coin_flip", "bench", "out")`, `FORMATION_WINDOW = 5`, `FORMATION_PRIOR = 0.10`, `LINE_BOUNDS = {"DEF": (3,5), "MID": (2,6), "FWD": (1,3)}`, `DEFAULT_SHAPE = (4,4,2)`, thresholds `TIER_SURE = 0.85`, `TIER_LIKELY = 0.60`, `TIER_COIN = 0.35`, `OUT_SQUAD = 0.15`, `OUT_PLAY = 0.05`.

Sort key everywhere: `(-p_start, -p_play, -xP, player_id)` so output is byte-stable.

**Shape bounds per team** (`formation_bounds`):
1. ≥ 2 team rows this season → mode of each line over the last 5 (tie → most recent), bounds = observed min/max widened to at least mode ± 1, clamped to `LINE_BOUNDS`; `formationSource = "team"`.
2. else ≥ 2 league rows this season → same over the league's last 5 matchdays; `"league"`.
3. else `DEFAULT_SHAPE` with `LINE_BOUNDS`; `"default"`.
Then clamp each lower bound to the players available at that line; `xi_size = min(11, squad size)` so tiny test squads work.

**Selection** (`assemble_team`): enumerate every `(d, m, f)` within bounds with `d+m+f = xi_size-1`; XI = top GK + top-d DEF + top-m MID + top-f FWD by sort key; score = `Σ p_start − FORMATION_PRIOR · (|d−mode_d|+|m−mode_m|+|f−mode_f|)`; argmax, ties → smaller deviation, then mode order. At most ~15 triples, plain loop. Exactly one GK (top by key, even at p_start 0); zero GK rows → no GK slot + warning. Within a position the XI is exactly the top-k, so tiers and depth ranks are monotone within `(team, position)`.

**Tiers** (`assign_tiers`, first rule wins):
1. `out`: `p_play ≤ OUT_PLAY` or `p_squad < OUT_SQUAD`
2. in XI, `p_start ≥ 0.85` → `sure`
3. in XI, `p_start ≥ 0.60` → `likely`
4. in XI → `coin_flip` (floor)
5. not in XI, `p_start ≥ 0.35` → `coin_flip` (cap)
6. else `bench`
`questionable` (cap 0.5) tops out at `coin_flip`. Thresholds are tuned once against the calibration table (§6).

**Rival links** (`rival_links`), within position only:
- non-XI, non-`out` player B: `replaces` = XI member at B's position with the lowest sort key (null if none).
- XI member A: `replacedBy` = highest-key non-XI, non-`out` player at A's position, i.e. depth rank k+1 (null if none).
- `out` players: both null, never a target. Several bench players may point at the same weakest starter; that is honest "next man up" semantics.
- GK: GK2 is `replacedBy` of GK1 even at tiny p_start.

`depthRank` = within-`(team, position)` rank over the whole squad (`out` last). `build_lineups(run) -> SeasonLineups` returns per-team dataclasses (team meta, fixture as mode over rows, shape, summary, ranked entries).

### 5. JSON shape

Additive fields on the existing `PlayerPrediction` (matchday and player files): `pSquad`, `pSquadRaw`, `pStartRaw`. `pStart` description changes from "≥60 minutes" to "in the starting XI" in [openapi.py](src/kickbase_xp/openapi.py), the landing page ([publish.py:363](src/kickbase_xp/publish.py#L363)) and README.

`/v1/lineups/{matchday}/{teamId}.json` and `/v1/lineups/current/{teamId}.json` (`TeamLineup`):

```json
{
  "apiVersion": "v1", "seasonId": "42", "matchday": 5, "generatedAt": "…", "predictor": "two_stage_lgbm",
  "teamId": "2", "teamName": "Bayern", "opponentTeamId": "10", "opponentTeamName": "…",
  "isHome": true, "kickoff": "2026-10-10T13:30:00Z",
  "summary": {
    "formation": "4-4-2", "counts": {"GK":1,"DEF":4,"MID":4,"FWD":2},
    "formationSource": "team", "formationHistory": ["4-4-2","4-5-1"],
    "squadSize": 25, "confidence": 0.81,
    "tiers": {"sure":8,"likely":2,"coin_flip":3,"bench":9,"out":3}
  },
  "lineup": {"GK":[…],"DEF":[…],"MID":[…],"FWD":[…]},
  "bench": [...], "out": [...]
}
```

`LineupEntry`: `playerId, name, position, status, pStart, pStartRaw, pPlay, pPlayRaw, pSquad, pSquadRaw, tier, depthRank, inLineup, replaces, replacedBy, xP, p20, p80, marketValue`. `replaces`/`replacedBy` are `{"playerId","name"}` or null. `lineup` lists sorted by `depthRank`; `bench`/`out` by sort key. `confidence` = mean `pStart` over the XI. `opponentTeamName` from a `team_id -> team_name` map over all prediction rows (no DB access in publish).

`/v1/lineups/index.json` (`LineupsIndex`): `apiVersion, seasonId, matchday, generatedAt, predictor, tiers, positions, formationWindow, teams[{teamId, teamName, opponentTeamId, isHome, kickoff, formation, formationSource, confidence, path}], endpoints{team, currentTeam}`. Teams sorted by kickoff then numeric team id.

Main `index.json` `endpoints` gains `lineups`, `teamLineup`, `currentTeamLineup`; `notes` gains a `lineups` sentence; landing page table gains three rows. Spec: new `lineups` tag, schemas `LineupEntry`, `TeamLineup`, `LineupsIndex`; paths listed with `/lineups/current/{teamId}.json` **before** `/lineups/{matchday}/{teamId}.json` so `tests/test_openapi.py::_spec_path_for` resolves `current` deterministically; a `teamId` path param like the existing `playerId` one.

### 6. Validation (validate.py, cli.py, docs)

Extend `walk_forward` without changing `summary` (so `train.choose_predictor` is untouched). `ValidationResult` gains:
- `classification`: per fold/target (`squad`,`play`,`start`) log loss + Brier (hand-rolled, clip 1e-6) for the model and probability baselines `squad_share_5`, `play_share_5`, `start_share_5`, `started_last`.
- `tiers`: per fold assemble lineups from the fold's predictions and `formation_history(..., before=cutoff)`, assign tiers, join `perf_status`; aggregate `n, start_rate, play_rate, squad_rate` per tier. Targets: `sure ≥ 0.90`, `likely` 0.65-0.85, `coin_flip` 0.35-0.60, `bench ≤ 0.15` start rate, `out ≤ 0.05` play rate.
- `xi`: per team-matchday `|predicted XI ∩ actual XI|` out of 11 and formation hit rate, for the model and a `last_xi` baseline (repeat the previous XI).
`report()` prints all tables. `cmd_validate` gets `--historical-status` (apply the status cap from `status_snapshots` as of the day before kickoff; honest from 2026-09-10 on). Guard the assembly in `choose_predictor`'s fold loop with try/except so it can never demote the predictor.

`docs/validation.md`: new sections after Results: "Playing-time classification", "Expected XI" (hits/11 vs `last_xi`, tier calibration table with the thresholds), "One-off: LigaInsider tiers on matchday 3" (start rate per `lineup_prob` tier from the 2026-09-11 snapshot vs the model's tiers, after a fresh fetch). Update the caveat at the end about calibration. README: API table rows, a `### Lineups` section mirroring Rankings, "three classifiers" in the Model section, "start share" wording, one Results line.

## Implementation steps (each keeps the suite green)

1. **Labels**: `config.py`, `features.py` (`load_raw`, `_prepare_base`, `ID_COLUMNS`), `model.py` (`dropna`), `tests/conftest.py` label variety (keep 8 players: p4's odd matchdays alternate `st=1`/`st=4`, p8 comes on as a sub on matchday 5 with `st=3`, 20 min, so every classifier sees two classes; pinned assertions on `start_share_5`, rankings md 1/2/4 and md 2 payloads stay valid), `tests/test_features.py` label tests.
2. **Features**: `features.py` additions + `_add_team_context`; tests incl. a second leak probe that flips `perf_status`/`minutes` of the last matchday and asserts earlier rows unchanged (the existing probe only perturbs `points`), and a merge_asof exact-match exclusion test.
3. **Third classifier**: `model.py`, `train.py` (`stages`, `formation_history` on `PredictionRun`), `tests/test_model.py` (importance set adds `p_squad`; `p_play ≤ p_squad`).
4. **`lineups.py` + `tests/test_lineups.py`**: pure unit tests on a fabricated 2-team × 25-player frame: exactly `min(11,n)`, one GK, counts within bounds, objective equals brute force, `FORMATION_PRIOR` behaviour (5th DEF at 0.95 vs 4th MID at 0.80 does not flex; vs 0.60 does), determinism under shuffle, tie-break by id, fallbacks team→league→default, tier rules incl. floor/cap/`out` by cap and by `p_squad`, monotone tiers/depth within `(team, position)`, rival links incl. GK and empty bench, `out` never a target, `inLineup` ⇔ membership, formation string.
5. **Validation**: `validate.py`, `cli.py`, `tests/test_validate.py` (classification has model + baselines with finite log loss; tiers ⊆ `TIERS`; xi has `last_xi`; `report()` mentions all tables).
6. **Publish + spec together** (strict test couples them): `publish.py` (`player_entry` new fields, `publish_lineups`, index endpoints/notes, landing rows, `pStart` wording), `openapi.py` (schemas, paths, tag, description), `tests/test_publish.py` (files exist, `current/{teamId}.json == {md}/{teamId}.json`, 4-man XI with one GK for the fixture teams, summary tier counts equal entries, main index advertises endpoints). `tests/test_openapi.py` runs unchanged and must pass.
7. **Docs**: README, `docs/validation.md` after a real `kickbase-xp fetch` + `validate` run.
8. Optional: `predict --team` prints one expected XI; `now = min(wall clock, last_fetch_at)` in `run_training` to close the stale-republish hazard.

## Critical files

- `src/kickbase_xp/features.py` (labels, new features, merge_asof leak guard)
- `src/kickbase_xp/lineups.py` (new: bounds, selection, tiers, rivals, `build_lineups`)
- `src/kickbase_xp/model.py`, `src/kickbase_xp/train.py` (third classifier, raw outputs, chain, `formation_history`)
- `src/kickbase_xp/publish.py`, `src/kickbase_xp/openapi.py` (change together)
- `src/kickbase_xp/validate.py`, `src/kickbase_xp/cli.py`
- `tests/conftest.py`, `tests/test_lineups.py` (new), plus touched tests above
- `README.md`, `docs/validation.md`

Reuse: `publish._num/_int/_write/_player_name/_kickoff`, `openapi._obj/_nullable/_ref/_get/PROBABILITY/POSITION/MATCHDAY_PARAM`, `rankings.ranked` sort-key style, `validate.walk_forward` fold loop, `model.status_cap`.

## Risks / edge cases

- Nightly jitter at the XI boundary shows up in `coin_flip`; `FORMATION_PRIOR` keeps the shape from flapping; `formationHistory` makes shape changes visible.
- Teams without a fixture on the target matchday get no file and no index row.
- `questionable` players can never be `sure`/`likely` (cap 0.5); if too harsh, raise the cap, not the tier rule.
- Walk-forward tiers without the status cap overstate `bench`/`coin_flip` start rates; `--historical-status` fixes that from 2026-09-10 on only.
- Per-team renormalisation of `p_start` to sum 11 is an *optional* experiment: measure Brier with and without before adopting; never overload `pStartRaw` (means "pre-cap").
- `rankings.py` is untouched by the label change (filters on `points.notna()`).

## Verification

1. `uv run pytest` green after every step.
2. `uv run kickbase-xp fetch` (fresh history incl. matchdays 3+), then `uv run kickbase-xp publish --out /tmp/site` and inspect `site/v1/lineups/index.json` and one team file: exactly 11 in `lineup`, one GK, tier counts match, rival links resolve to ids in the same file, `current/{teamId}.json` identical to `{md}/{teamId}.json`.
3. `uv run kickbase-xp validate --first-matchday 6` and read the tier calibration table; adjust thresholds once if `sure` < 0.90 or `out` play rate > 0.05; record in `docs/validation.md`.
4. Spot-check against the 2026-09-11 snapshot's `lineup_prob` for matchday 3 (tier agreement) as a sanity benchmark.
5. Spec: `tests/test_openapi.py` validates every published file strictly and that every documented path is published; also open `site/v1/openapi.json` in a validator once.
