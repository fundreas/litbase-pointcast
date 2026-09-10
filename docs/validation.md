# Walk-forward validation

Milestone 3 of the plan: does the model actually beat the naive baselines,
and by enough to be worth the machinery?

Reproduce with:

```bash
uv run kickbase-xp validate --season 34
```

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
  the calibration of `p20`/`p80` is not measured here.
