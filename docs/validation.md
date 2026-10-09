# Walk-forward validation

Milestone 3 of the plan: does the model actually beat the naive baselines,
and by enough to be worth the machinery?

Reproduce with:

```bash
uv run kickbase-xp validate --season 34
```

The MAE table predates the switch from the minutes-based starter label to
the feed's real lineup status; re-running gives 37.24 for the model (37.34
before) and leaves the baselines within ±0.3.

## Method

For each matchday of the 2025/26 Bundesliga season, train only on rows that
kicked off **before** its first fixture, predict it, score it. No random
splits — those would let the model read matchday 30 while predicting
matchday 5.

Errors are scored against actual Kickbase points, with a non-appearance
counting as 0 (which is what the consumer of the feed cares about: what did
this player actually deliver). The first five matchdays are skipped because
the rolling-5 features, and the baseline built on them, are undefined there.

The status override is **not** applied in validation. The fit/injured flag is
only ever known for today, so a historical fold cannot use it without
inventing information — which means the live feed should be slightly better
than these numbers, not worse.

## Results — season 2025/26, matchdays 6–34

29 folds, 8,224 predictions.

| Predictor | MAE | RMSE | Spearman |
|---|---|---|---|
| **two_stage_lgbm** | **37.34** | **55.73** | **0.618** |
| form_x_startshare (baseline 1) | 39.83 | 62.20 | 0.513 |
| rolling_mean_5 | 40.51 | 60.48 | 0.527 |
| position_average (baseline 2) | 52.84 | 66.30 | 0.124 |

The model clears both baselines from the plan, so §6.3's "ship the heuristic
instead" clause does not trigger. It wins on 25 of 29 individual matchdays.

**The rank correlation is the more interesting number.** A 6% MAE improvement
is modest; going from 0.51 to 0.62 Spearman is not. Nobody picks a squad by
absolute points — they pick the best available player at a position, and
ordering is what that depends on. RMSE improving faster than MAE (−10.4% vs
−6.2%) says the same thing from the other side: the model's remaining errors
are less catastrophic, mostly because stage 1 catches the players who were
never going to be on the pitch.

Split by point in the season:

| Window | two_stage_lgbm | form_x_startshare |
|---|---|---|
| Matchdays 6–16 | 38.35 | 40.08 |
| Matchdays 17–34 | 36.75 | 39.68 |

The gap widens as the season goes on, which is what you would expect: the
opponent-strength features are shrunk heavily toward their prior early and
only start carrying real information once each team has a few matchdays on
record.

## Playing-time classification

Stage 1 is three binary classifiers trained on the feed's real lineup labels
(`st`: started, came on, unused, not in squad). Scored as probability
forecasts on the same 29 folds, against the obvious heuristics: the rolling
five-match share and "same as last match". Lower is better.

| Target | Predictor | Log loss | Brier |
|---|---|---|---|
| in squad | **two_stage_lgbm** | **0.118** | **0.026** |
| | squad_share_5 | 0.300 | 0.040 |
| | squad_last | 0.402 | 0.029 |
| plays | **two_stage_lgbm** | **0.370** | **0.112** |
| | play_share_5 | 0.993 | 0.144 |
| | played_last | 2.320 | 0.168 |
| starts | **two_stage_lgbm** | **0.405** | **0.130** |
| | start_share_5 | 1.078 | 0.173 |
| | started_last | 3.022 | 0.219 |

The huge log losses of the last-match baselines are what a hard 0/1 forecast
earns when it is wrong; their Brier scores are the fairer comparison, and the
model still wins those by 25–40%.

## Expected lineups

Each fold assembles every team's expected XI exactly as the published
`/v1/lineups/` files do, then checks the tiers against what happened. No
status cap (see Method), so injured players stay in the pool and the upper
tiers are slightly pessimistic.

| Tier | Players | Started | Played | In squad |
|---|---|---|---|---|
| sure | 2,148 | 89.0% | 94.4% | 99.5% |
| likely | 1,423 | 73.1% | 89.0% | 98.9% |
| coin_flip | 1,783 | 37.8% | 67.8% | 97.4% |
| bench | 1,666 | 14.8% | 46.2% | 96.9% |
| out | 1,204 | 1.1% | 3.6% | 73.8% |

Targets set before the run: `sure` ≥ 90% starts, `likely` 65–85%,
`coin_flip` 35–60%, `bench` ≤ 15%, `out` ≤ 5% appearances. Everything lands
inside except `sure`, one point short — with injured players still in the
pool, which the live status cap removes. The thresholds were left as
designed rather than tuned to this season.

Correct starters, counted only where the real XI is complete on record
(survivorship leaves just 37 such team-matchdays in 2025/26):

| Predictor | Starters right, of 11 | Shape right |
|---|---|---|
| **expected XI** | **9.46** | 70% |
| last_xi (repeat the previous XI) | 8.79 | 73% |

Repeating last week's XI is the bar, because XIs change by under two
players a week on average. The model beats it by two-thirds of a player per
team. It guesses the *shape* slightly less often than "same as last time";
the 0.10-per-player penalty keeps it from flexing on thin evidence.

### One-off: against LigaInsider on matchday 3 (2026/27)

Kickbase served a LigaInsider lineup tier (`lineup_prob`, 1–5) until
2026-09-17. The snapshot from the day before matchday 3 still holds it, so
this one matchday allows a direct comparison. The model is trained only on
data before kickoff and capped with that day's archived status
(`--historical-status`).

| Model tier | n | Started | | LigaInsider | n | Started |
|---|---|---|---|---|---|---|
| sure | 93 | 93.5% | | 1 | 110 | 92.7% |
| likely | 77 | 76.6% | | 2 | 50 | 84.0% |
| coin_flip | 47 | 59.6% | | 3 | 81 | 51.9% |
| bench | 181 | 13.3% | | 4 | 48 | 16.7% |
| out | 64 | 0.0% | | 5 | 172 | 2.3% |

The top tiers are as reliable as LigaInsider's, from Kickbase data alone.
LigaInsider names more sure starters (110 vs 93): its journalists know the
training-ground news the model cannot see. The expected XI got 8.94 of 11
starters right that day. One matchday, so read it as a sanity check, not a
result.

## Context for the absolute numbers

A MAE of 37 on a scale that runs roughly −100…600 sounds large, and is. Two
reasons it cannot get much smaller:

1. **Kickbase points are genuinely noisy.** A single goal, card or penalty
   swings a player's score by more than the entire MAE. A large part of the
   residual is irreducible.
2. **No lineup news.** Surprise rotations are invisible in Kickbase data
   until the market value reacts, usually too late. This is the accuracy
   ceiling the plan called out in §6.4, and it is the reason the two-stage
   split exists — the error lands in the playing-time model, where it is at
   least legible, instead of being smeared across the points estimate.

## Caveats

- **Survivorship bias.** Only players on a squad *today* appear in the
  performance feed, so 2025/26 validates on ~280 players per matchday rather
  than the ~460 who actually played. Everyone who left the league is missing.
  This flatters every predictor roughly equally, so the comparison holds, but
  the absolute MAE is optimistic.
- **Single season.** Season 34 is the only complete recent season with good
  roster coverage. Earlier seasons are progressively thinner.
- Quantile models are skipped during validation (`--quantiles` enables them);
  the calibration of `p20`/`p80` is not measured here. The calibration of
  `pStart`/`pPlay`/`pSquad` is, through the tier table above.
- **Lineup accuracy on old seasons is thin.** Complete XIs on record are rare
  before the running season, so the starter count rests on few teams. The
  running season will give a cleaner number as it accumulates.
