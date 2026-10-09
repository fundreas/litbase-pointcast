# kickbase-xp

Expected Kickbase points (`xP`) for every Bundesliga player, for the next
matchday, rebuilt nightly on free GitHub infrastructure and published as
static JSON on GitHub Pages — a free, open expected-points API for Kickbase.

Everything is derived from the Kickbase v4 API alone. No betting odds, no xG
feed, no LigaInsider scrape.

```
xP = P(plays) × E[points | plays]
```

## The published API

Versioned under `/v1/` from the first run, so a later schema change becomes
`/v2/` rather than a silent breakage for consumers.

| Path | Contents |
|---|---|
| `/v1/openapi.json` | OpenAPI 3.1 description of everything below |
| `/v1/index.json` | matchday, model metadata, recent validation |
| `/v1/matchday/current.json` | all players, sorted by xP |
| `/v1/matchday/{md}.json` | one specific matchday |
| `/v1/players/index.json` | compact player list |
| `/v1/players/{playerId}.json` | one player, including a feature digest |
| `/v1/lineups/index.json` | teams with an expected lineup for the next matchday |
| `/v1/lineups/current/{teamId}.json` | one team's expected XI, bench and absentees |
| `/v1/lineups/{md}/{teamId}.json` | the same, by matchday number |

```json
{
  "playerId": "1685",
  "name": "Joshua Kimmich",
  "teamId": "2",
  "position": "MID",
  "matchday": 3,
  "xP": 230.9,
  "p20": 103.9,
  "p80": 274.7,
  "pStart": 0.64,
  "status": "fit",
  "generatedAt": "2026-09-10T21:04:00Z"
}
```

The OpenAPI document is generated from the same constants as the output, and
the tests validate every published file against it, with unknown fields
rejected, so the two cannot drift apart. The spec itself allows unknown fields,
because v1 only ever grows by adding fields and a client validating against it
should not break when that happens. Its first `servers` entry is the deployed
Pages URL, which the nightly job takes from `actions/configure-pages`. The
landing page links the spec as `rel="service-desc"` (RFC 8631).

`pStart` is P(in the starting XI), `pPlay` P(plays at all) and `pSquad`
P(in the matchday squad), each capped by the player's status; the `…Raw`
fields are the model's values before that cap. `pStart ≤ pPlay ≤ pSquad`
always holds.

`p20`/`p80` are **unconditional** — they already fold in the chance of not
playing at all, so a rotation risk shows up as a floor of `0` rather than as
an optimistic conditional number. That is the difference between a gamble and
a safe pick, which is what you actually want when setting a lineup.

Kickbase points run roughly −100…600 per matchday. An xP of 140 is a good
performance, not a typo.

### Rankings

Actual points of the running season, computed from the same performance
history the model trains on, so they cost no extra API calls.

| Path | Contents |
|---|---|
| `/v1/rankings/index.json` | ranked matchdays, each with `complete`, `matchesPlayed`, `matchesTotal` |
| `/v1/rankings/matchday/{md}.json` | points scored on matchday `md` alone |
| `/v1/rankings/season/{md}.json` | season totals over matchdays 1…`md` |
| `/v1/rankings/{matchday,season}/current.json` | the latest matchday that has kicked off |

Every file holds `overall` (top 100) and `byPosition` with `GK`, `DEF`,
`MID` and `FWD` (top 100 each). Ties share a rank (`1, 1, 3`), and a tie at
place 100 is kept whole, so a list can run a few entries long.

```json
{
  "rank": 1,
  "playerId": "1685",
  "name": "Kimmich",
  "teamId": "2",
  "teamName": "Bayern",
  "position": "MID",
  "points": 557,
  "minutes": 192,
  "appearances": 2,
  "pointsPerAppearance": 278.5
}
```

`appearances` and `pointsPerAppearance` exist in season files only. A
matchday entry lists the club the player played for that day; a season entry
lists the current club.

The ongoing matchday is ranked as it stands, with `complete: false` until
every match is three hours past kickoff. The cutoff is the last fetch, not
the wall clock, so a re-publish from stale history leaves out matchdays it has
no points for instead of presenting them as finished and empty.

### Lineups

The expected team sheet of every club for the predicted matchday, built from
the same probabilities as the matchday file.

```json
{
  "teamId": "2", "teamName": "Bayern", "opponentTeamId": "10", "isHome": true,
  "summary": {
    "formation": "4-4-2", "counts": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
    "formationSource": "team", "usualFormation": "4-4-2",
    "formationHistory": ["4-4-2", "4-5-1"], "squadSize": 25, "confidence": 0.81,
    "tiers": {"sure": 8, "likely": 2, "coin_flip": 3, "bench": 9, "out": 3}
  },
  "lineup": {"GK": [...], "DEF": [...], "MID": [...], "FWD": [...]},
  "bench": [...],
  "out": [...]
}
```

Each player carries `pStart`, `pPlay`, `pSquad` (and their raw values), a
`tier`, a `depthRank` within his position and two rival links:

| Tier | Meaning |
|---|---|
| `sure` | in the expected XI, pStart ≥ 0.85 |
| `likely` | in the expected XI, pStart ≥ 0.60 |
| `coin_flip` | in the XI with a lower pStart, or outside it with pStart ≥ 0.35 |
| `bench` | expected in the squad, not in the XI |
| `out` | pPlay ≤ 0.05 or pSquad < 0.15 — status or model says he will not be there |

`replaces` on a bench player names the expected starter at his position he
would most likely replace (the one with the lowest pStart); `replacedBy` on a
starter names the next man up. Links never cross positions.

**Formations are in Kickbase positions.** Kickbase knows GK, DEF, MID and FWD
only, so `4-4-2` means four defenders, four midfielders and two forwards as
Kickbase classifies them, not a tactical system. The usual shape is the most
common one in the team's last five complete lineups of the *current* season
(earlier seasons are thinned by survivorship), falling back to the league's,
then to `4-4-2`. The XI is the shape within the team's observed range that
maximises summed pStart, with a penalty of 0.10 per player away from the
usual shape, so it only flexes on clear evidence and does not flap from night
to night.

## How it works

```
kickbase-xp fetch    # Kickbase v4 API  ->  data/history.sqlite
kickbase-xp predict  # features -> two-stage model -> next matchday (stdout)
kickbase-xp publish  # ...and write site/v1/*.json
kickbase-xp run      # fetch + publish, i.e. what the nightly Action does
kickbase-xp validate # walk-forward MAE vs. the baselines (local, one-time)
kickbase-xp features # dump the feature matrix for inspection
```

### 1. Data layer — `fetch.py`, `db.py`

Login, then the league table (which doubles as the team list), one team
profile per club for the player universe, per-player performance history,
market-value history, and the fixture list. Requests are throttled and
retried; this is an unofficial API and the pipeline is a guest on it.

Everything lands in `data/history.sqlite`. The performance endpoint replays
a player's full career on every call, so the first run *is* the backfill.

**What gets committed, and why it is not the database.** The plan called for
committing the SQLite file so history survives ephemeral Action runners. The
intent is right; the mechanism does not scale. The database is ~16 MB (4 MB
gzipped) and git cannot delta a re-VACUUMed binary blob, so a nightly commit
adds the whole thing again every night — over a gigabyte a year.

So the history is split by whether the API can re-serve it:

- **Re-derivable** — performances, fixtures, squads. `data/history.sqlite` is
  a pure cache and is *not* committed.
- **Perishable** — what a player's status and market value were *on a given
  day*. The API only ever answers "right now", and market-value history is
  capped at 365 days. Miss a day and it is gone.

The perishable slice is one row per player per day, so it goes to
`data/snapshots/YYYY-MM-DD.csv.gz`: **4 KB for all 461 players**, written
once, never rewritten. Git stores each file a single time and a year costs
about 1.5 MB. As the archive ages it also reaches further back than the API's
own 365-day window, so the feed's market-value history becomes better than
what Kickbase exposes.

**Two things the feed does that will bite you if you miss them:**

- Season ids are global, not per-competition. A player's 2. Bundesliga, La
  Liga or MLS seasons arrive in the same list, with ids *interleaved* among
  the Bundesliga ones — the current 2. Bundesliga season (43) outranks the
  current Bundesliga one (42). Only the season's `n` field tells them apart,
  so it is stored per row and filtered on everywhere.
- A finished match with no minutes entry means the player was not involved,
  which is a 0, not a missing value. Distinguishing that from a genuinely
  scheduled fixture is what `completed` does. A kicked-off fixture the
  history has not caught up with (`st == 0`, no minutes) is *unknown* and
  never trained on.
- The per-match `st` is the real lineup: `5` started (exactly eleven per
  team), `3` came on, `4` unused bench, `1` not in the squad, `0` not yet
  played. The `started`/`played`/`in_squad` labels come from it; minutes
  only back it up, for the 2021/22 season where some subs are filed as `4`.

### 2. Features — `features.py`

Every feature is computed from information available *strictly before* the
kickoff of the row it describes, enforced by a per-player `shift(1)` over a
chronologically sorted frame. Finished and scheduled matchdays go through the
same code path, so training and inference cannot drift apart.

| Group | Features |
|---|---|
| Form | rolling mean/median points (3/5/10), volatility, last result, form over the last N *appearances*, season and career means |
| Playing time | rolling minutes, play/start/squad share over 5/10, last match's role, start streak, season shares, games played, days since last match |
| Role | position, matchday |
| Competition for places | start rank among teammates at the position, the team's usual starters there, rank minus slots, team rotation rate |
| Market | log market value, 7- and 30-day momentum |
| Fixture | home/away, opponent points allowed to this position, opponent and own team strength |

**Opponent strength without external data.** "Points conceded to forwards"
needs no xG feed: it is literally the average score of the opposing team's
forwards in that team's past fixtures. Same source, better aligned to the
target than a league table would be. Estimates are shrunk toward the team's
value last season, falling back to the league's *previous* seasons — never a
mean over the whole dataset, which would quietly leak the future into
matchday 1.

**Market value as free crowd knowledge.** MV reacts to press conferences and
rotation rumours that are invisible in the API. Reading its 7-day momentum
imports community knowledge without importing a dependency.

### 3. Model — `model.py`, `train.py`

Two stages, because this is two problems:

1. **Playing time** — three classifiers, for `P(in squad)`, `P(plays)` and
   `P(starts)`, trained on the feed's real lineup labels. Most of the
   predictable signal lives here; a bench player predicted at 180 points is
   worse than useless.
2. **Points given playing** — a mean regressor plus nine quantile regressors,
   which is where the floor and ceiling come from.

Retrained from scratch on every run. ~23k rows trains in seconds, so there
are no model artifacts to version, no drift handling, and the model always
sees everything up to last night.

**Status is a hard override, not a feature.** The fit/injured/suspended flag
is only ever known for *today*, so training on it would leak. It is applied
afterwards as a cap on all three probabilities: suspended → 0,
injured → 0.02, questionable → 0.5. The model's own belief is kept in the
`…Raw` fields for debugging.

### 3b. Lineups — `lineups.py`

Turns per-player probabilities into one team sheet per club: the usual shape,
the XI that maximises summed pStart within it, tiers, depth ranks and rival
links. See [Lineups](#lineups) above.

### 4. Validation — `validate.py`, `baselines.py`

Walk-forward: for each matchday, train only on what kicked off before it,
predict it, score it. Run locally, not in the Action.

Baselines to beat, from the plan:

1. `form_x_startshare` — last-5 form × start share
2. `rolling_mean_5` — the dumbest thing that could work
3. `position_average`

The walk-forward also scores stage 1 as classifiers (log loss and Brier
against the rolling share and last-match baselines), calibrates the lineup
tiers against what actually happened, and counts correct starters out of 11
against simply repeating the team's last XI. `--historical-status` applies
the archived status of the day before kickoff, where a snapshot exists.

Plan §6.3 says: if the model cannot beat naive form, ship the heuristic. That
is not a footnote here — it is wired in. Every nightly run re-checks the model
against `form_x_startshare` on the most recent matchdays and demotes itself if
it loses, recording the choice in `/v1/index.json` as `predictor`. A model
that degrades mid-season stops shipping instead of quietly shipping worse
numbers.

### 5. Publishing — `publish.py`

Static JSON to `site/`, deployed to Pages as an artifact (no committed build
output). `NaN` never reaches the output; it becomes `null`, because
`NaN` is invalid JSON that many clients reject outright.

### 6. Automation — `.github/workflows/nightly.yml`

Cron at 22:20 UTC, after the ~22:00 CET market-value update. Tests, fetch,
train, publish, commit the day's snapshot, deploy to Pages.
`workflow_dispatch` takes an optional matchday and a `skip_fetch` toggle for
re-publishing without touching the API.

## Setup

```bash
uv sync --extra dev
printf 'KICK_EMAIL=you@example.com\nKICK_PASS=...\n' > .env
uv run kickbase-xp fetch      # ~4 min for 461 players
uv run kickbase-xp predict
```

`.env` is gitignored and only used locally. In CI the credentials come from
the repository secrets `KICK_EMAIL` and `KICK_PASS`.

GitHub Pages on the free plan requires a **public** repository. That means
the predictions are public — arguably the point.

## Results

Walk-forward on the 2025/26 season, 29 matchdays, 8,224 predictions — full
write-up in [docs/validation.md](docs/validation.md).

| Predictor | MAE | RMSE | Spearman |
|---|---|---|---|
| **two_stage_lgbm** | **37.34** | **55.73** | **0.618** |
| form_x_startshare (baseline 1) | 39.83 | 62.20 | 0.513 |
| rolling_mean_5 | 40.51 | 60.48 | 0.527 |
| position_average (baseline 2) | 52.84 | 66.30 | 0.124 |

The model clears both baselines, winning 25 of 29 matchdays. The MAE gain is
modest (−6%); the rank correlation going from 0.51 to 0.62 is the number that
matters, because nobody picks a squad by absolute points — they pick the best
available player at a position.

The expected lineups get 9.5 of 11 starters right against 8.8 for "same XI
as last week", and players tiered `sure` start 89% of the time. On the one
matchday where LigaInsider's tiers are still on record, the model's top tier
was as reliable as theirs. Details in [docs/validation.md](docs/validation.md).

## Known limitations

- **Survivorship bias.** The feed only replays players who are on a squad
  *today*, so historical matchdays are missing everyone who has since left the
  league. Recent seasons are barely affected; older ones look like a league of
  a few hundred survivors, which is part of why training defaults to the last
  four seasons (`--seasons`). The rankings inherit this: a player who leaves
  the league mid-season drops out of earlier matchdays too.
- **No lineup news.** Surprise rotations are invisible until the market value
  reacts. The accuracy ceiling is below LigaInsider's, and the two-stage split
  is what keeps that error contained in the playing-time model rather than
  smeared across the points estimate.
- **Status has no history.** Only today's flag is available, so it can only be
  an override. The nightly `status_snapshots` table is accumulating the
  history that would make it a real feature in v2.
- **`lineup_prob`** was a LigaInsider-derived tier, stored but deliberately
  unused. Kickbase stopped serving it on 2026-09-17; the snapshots up to
  2026-09-16 keep it as a one-off benchmark for the lineup tiers.
- **Lineups cannot see news.** A late injury, a new signing without history
  or a coach resting players for a European week only shows up once the
  status flag or the market value moves.

## Later (v2+)

- SHAP contributions per player in the JSON
- Multi-matchday horizon for transfer planning
- Status history as a trained feature, once enough snapshots have accumulated
- 2. Bundesliga (the data is already being stored and labelled)
