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
| `/v1/index.json` | matchday, model metadata, recent validation |
| `/v1/matchday/current.json` | all players, sorted by xP |
| `/v1/matchday/{md}.json` | one specific matchday |
| `/v1/players/index.json` | compact player list |
| `/v1/players/{playerId}.json` | one player, including a feature digest |

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

`p20`/`p80` are **unconditional** — they already fold in the chance of not
playing at all, so a rotation risk shows up as a floor of `0` rather than as
an optimistic conditional number. That is the difference between a gamble and
a safe pick, which is what you actually want when setting a lineup.

Kickbase points run roughly −100…600 per matchday. An xP of 140 is a good
performance, not a typo.

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
  scheduled fixture is what `completed` does.

### 2. Features — `features.py`

Every feature is computed from information available *strictly before* the
kickoff of the row it describes, enforced by a per-player `shift(1)` over a
chronologically sorted frame. Finished and scheduled matchdays go through the
same code path, so training and inference cannot drift apart.

| Group | Features |
|---|---|
| Form | rolling mean/median points (3/5/10), volatility, last result, form over the last N *appearances*, season and career means |
| Playing time | rolling minutes, play share, start share (≥60 min) over 5/10, season shares, games played, days since last match |
| Role | position, matchday |
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

1. **Playing time** — classifiers for `P(plays)` and `P(starts)`. Most of the
   predictable signal lives here; a bench player predicted at 180 points is
   worse than useless.
2. **Points given playing** — a mean regressor plus nine quantile regressors,
   which is where the floor and ceiling come from.

Retrained from scratch on every run. ~23k rows trains in seconds, so there
are no model artifacts to version, no drift handling, and the model always
sees everything up to last night.

**Status is a hard override, not a feature.** The fit/injured/suspended flag
is only ever known for *today*, so training on it would leak. It is applied
afterwards as a cap on `P(plays)`: suspended → 0, injured → 0.02,
questionable → 0.5. The model's own belief is kept in `pPlayRaw` for
debugging.

### 4. Validation — `validate.py`, `baselines.py`

Walk-forward: for each matchday, train only on what kicked off before it,
predict it, score it. Run locally, not in the Action.

Baselines to beat, from the plan:

1. `form_x_startshare` — last-5 form × start share
2. `rolling_mean_5` — the dumbest thing that could work
3. `position_average`

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

## Known limitations

- **Survivorship bias.** The feed only replays players who are on a squad
  *today*, so historical matchdays are missing everyone who has since left the
  league. Recent seasons are barely affected; older ones look like a league of
  a few hundred survivors, which is part of why training defaults to the last
  four seasons (`--seasons`).
- **No lineup news.** Surprise rotations are invisible until the market value
  reacts. The accuracy ceiling is below LigaInsider's, and the two-stage split
  is what keeps that error contained in the playing-time model rather than
  smeared across the points estimate.
- **Status has no history.** Only today's flag is available, so it can only be
  an override. The nightly `status_snapshots` table is accumulating the
  history that would make it a real feature in v2.
- **`lineup_prob`** is fetched and stored but deliberately unused: it is a
  LigaInsider-derived tier, and v1 stays free of external predictors.

## Later (v2+)

- SHAP contributions per player in the JSON
- Multi-matchday horizon for transfer planning
- Status history as a trained feature, once enough snapshots have accumulated
- 2. Bundesliga (the data is already being stored and labelled)
