# Kickbase Expected Points (kickbase-xp) — Project Plan

A nightly pipeline that predicts Kickbase matchday points per player, runs entirely on free GitHub infrastructure, and publishes its predictions as static JSON files via GitHub Pages — effectively a free, open "expected points API" for Kickbase.

---

## 1. Goal & Scope

- **Output:** Expected Kickbase points (`xP`) for every Bundesliga player for the **next matchday**, plus an uncertainty range and start probability.
- **Runtime:** Nightly GitHub Action (free plan), scheduled after the Kickbase market value update (~22:00 CET).
- **Publishing:** Static JSON files on GitHub Pages, versioned under `/v1/`.
- **Data constraint:** All features derived exclusively from Kickbase (v4 API) data. No external lineup predictors, odds, or xG feeds in v1.
- **First consumer:** The existing Kickbase web app (kickbase.my-domain.at).

### Non-goals (v1)

- Live points during matches
- Lineup/rotation news ingestion
- Multi-league support beyond Bundesliga (design should not preclude it)

---

## 2. Key Design Decisions

### 2.1 No model reuse from FPL projects

Trained models (e.g. `solpaul/fpl-prediction`, SmartPlayFPL) are weights over FPL's feature schema and FPL's target distribution (~0–20 points). Kickbase points come from a completely different scoring system (Stats Perform event grading, roughly −100 to +600) with different available features. **We reuse the approach (feature engineering, validation method, two-stage structure), never the model files.**

### 2.2 Retrain from scratch on every nightly run

The dataset is tiny by ML standards: ~550 players × ~34 matchdays × a few seasons ≈ 30–50k tabular rows. LightGBM trains on this in seconds. Full nightly retraining means:

- No model artifacts to version or store
- No drift handling
- Model always uses all data up to yesterday

### 2.3 Two-stage model

Kickbase points prediction is really two different problems:

1. **Playing-time model** — classifier/regressor for P(plays) and expected minutes. Most of the predictable signal lives here; a bench player predicted at 180 points is useless.
2. **Points-given-playing model** — regression of Kickbase points conditional on the player actually playing.

**Final prediction:**

```
expected_points = P(plays) × E[points | plays]
```

Stage 2 optionally as **quantile regression** (p20 / p50 / p80) so the API exposes a floor/ceiling per player — more useful for lineup decisions (gamble vs. safe pick) than a single number.

---

## 3. Features / Indicators (all from Kickbase v4 API)

### 3.1 Player form & role

| Feature | Notes |
|---|---|
| Rolling mean/median points (last 3, last 5 matchdays) | Core form signal |
| Season average points | Stabilizes early-form noise |
| Rolling minutes (last 3, last 5) | Playing-time trend |
| Start share (games with ≥60 min in last 5) | Best available proxy for start probability |
| Points volatility (std dev) | Separates consistent defenders from boom-bust strikers |
| Position (GK / DEF / MID / FWD) | Points distributions differ massively by position |
| Player status flag (fit / questionable / injured) | **Hard override:** status ≠ fit slashes P(plays) regardless of model output |

### 3.2 Market signal

| Feature | Notes |
|---|---|
| Current market value | Crowd-sourced quality estimate |
| 7-day MV trend / momentum | MV reacts to news invisible in the API (press conferences, rotation rumors) — importing community knowledge for free |

### 3.3 Fixture context

| Feature | Notes |
|---|---|
| Home / away | Players score measurably more at home |
| Opponent strength by position | Derived from Kickbase data itself: avg. points the opponent has *conceded to each position* this season ("points allowed vs forwards", etc.) — replaces external xG/table data |
| Own team's average total points | Team strength proxy |

### 3.4 Explicitly skipped in v1

External lineup predictors, betting odds, xG feeds. Costs some accuracy vs. LigaInsider, keeps the pipeline dependency-free and fully automated.

---

## 4. Architecture & Repo Layout

```
kickbase-xp/
├── .github/
│   └── workflows/
│       └── nightly.yml        # cron ~23:00 CET, after 22:00 MV update
├── src/
│   ├── fetch.py               # Kickbase v4 API pull (login, players, history, MV, status, fixtures)
│   ├── features.py            # feature matrix builder
│   ├── train.py               # 2× LightGBM, full retrain per run
│   └── publish.py             # JSON output for gh-pages
├── data/
│   └── history.sqlite         # committed player/matchday history
├── docs/ (or gh-pages branch) # published JSON
└── requirements.txt
```

### 4.1 Nightly job steps

1. **Fetch** — login (credentials from repo secrets), pull player list, per-player performance history, market values, status flags, next fixtures. Small delays between requests (same courtesy the community analysis tools apply — this is an unofficial API).
2. **Persist** — upsert into committed SQLite file. Proven pattern (Kickbase-Trading-Advisor commits `player_data_total.db` the same way); history survives ephemeral Action runners.
3. **Train** — rebuild feature matrix, retrain both LightGBM models from scratch.
4. **Predict** — next matchday, all players.
5. **Publish** — write JSON, push to `gh-pages` branch.

### 4.2 Published JSON layout (static API)

```
/v1/index.json                 # matchday, generated_at, model metadata
/v1/matchday/{md}.json         # all players, sorted by xP
/v1/players/{playerId}.json    # per player: xP, p20/p80, P(start), key features
```

**Version the path from day one** — schema changes later must not break consumers (the Kickbase web app first among them).

Example player entry:

```json
{
  "playerId": "237",
  "name": "…",
  "teamId": "2",
  "position": "FWD",
  "matchday": 12,
  "xP": 142.3,
  "p20": 41.0,
  "p80": 268.5,
  "pStart": 0.87,
  "status": "fit",
  "generatedAt": "2026-09-10T21:04:00Z"
}
```

---

## 5. GitHub Free Plan Constraints

| Concern | Assessment |
|---|---|
| Actions minutes | Public repo: unlimited. Even private: a run of this size (few min API fetch + seconds of training) stays far under the 2,000 min/month cap. |
| GitHub Pages | Requires a **public repo** on the free plan → repo should be public. |
| Secrets | Kickbase credentials strictly in Actions secrets (`KICK_USER`, `KICK_PASS`). Never in code or committed files. |
| Public predictions | Anyone can read the published JSON. Arguably a feature — it would be the first open Kickbase points forecast. |

---

## 6. Bootstrapping & Validation

### 6.1 Bootstrapping

Day one there is no local history, but the per-player performance endpoint returns past matchdays **including previous seasons** → the first run can backfill a full training set immediately.

### 6.2 Validation (local, one-time — not in the Action)

Walk-forward validation, copied from the FPL projects' method:

- For each historical matchday: train only on data *before* it, predict it, measure MAE.
- Aggregate MAE across the season(s).

### 6.3 Baselines to beat

1. **Naive form:** last-5-matchdays average × start share
2. **Position average**

If the model doesn't beat baseline 1, **ship the heuristic as v1** and iterate. A published simple xP feed beats an unpublished clever one.

### 6.4 Expectation setting

Without lineup news, the accuracy ceiling is below LigaInsider's — surprise rotations are invisible in Kickbase data until market value reacts. The two-stage design isolates that error in the playing-time model, where it belongs.

---

## 7. Milestones

| # | Milestone | Content |
|---|---|---|
| 1 | Data layer | `fetch.py` + SQLite schema, backfill of historical matchdays |
| 2 | Features | `features.py`, feature matrix reproducible from SQLite |
| 3 | Baselines & validation | Walk-forward harness, naive baselines, MAE report |
| 4 | Models | Two-stage LightGBM, quantile outputs, beat baseline 1 |
| 5 | Publishing | `publish.py`, `/v1/` JSON schema, gh-pages deploy |
| 6 | Automation | `nightly.yml` with secrets, scheduled cron, end-to-end run |
| 7 | Consumption | Integrate feed into the Kickbase web app |

---

## 8. Later Ideas (v2+)

- Expose per-feature contributions (SHAP) per player in the JSON
- Multi-matchday horizon (next 3 matchdays) for transfer planning
- 2. Bundesliga support
- Confidence-weighted lineup optimizer on top of the xP feed
