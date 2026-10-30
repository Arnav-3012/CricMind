# CricMind

[![CI](https://github.com/Arnav-3012/CricMind/actions/workflows/ci.yml/badge.svg)](https://github.com/Arnav-3012/CricMind/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![uv](https://img.shields.io/badge/packaging-uv-DE5FE9)
![DuckDB](https://img.shields.io/badge/DuckDB-FFF000?logo=duckdb&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-02A04D)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Data: Cricsheet ODC-By 1.0](https://img.shields.io/badge/data-Cricsheet%20ODC--By%201.0-blue)](https://cricsheet.org)

**Player matchups in men's T20 cricket: how a batter does against a bowler, answered from held-out-season evidence.**

CricMind turns Cricsheet ball-by-ball data into player and archetype matchup estimates, then puts a tool-using
LLM agent on top that answers tactical questions with numbers taken only from tool output. Every layer is
scored on seasons the model never saw, and negative results are reported as they came out.

## Research question

Do archetypes built from **learned player embeddings** predict held-out delivery outcomes better than archetypes
built from **hand-made per-phase stats**, and can an agent answer matchup questions from them with verifiable numbers?

**Answer: no to the first, yes to the second.** Stats archetypes beat embedding archetypes, and neither adds
anything once a player's own rates are in the model. The agent works because its numbers come from tools, not
because archetypes are strong.

## Results

Split by match date: train ≤ 2023, validation 2024, test 2025–2026 (478,076 balls, 2,210 matches), touched once at
the phase gate. Metric is multiclass log-loss over 7 outcomes (`dot, 1, 2, 3+, 4, 6, W`), lower is better.
All numbers are written to `reports/` by code.

| Model (test 2025–26) | Log-loss |
|---|---|
| Phase frequencies (baseline) | 1.4824 |
| Stats-archetype matchup table | 1.4809 |
| cric2vec (learned embeddings, MLP) | 1.4511 |
| LightGBM on hand-made stats | **1.4464** |

- **Archetypes are barely better than the baseline.** On the 24% of test balls where both players have an
  archetype label, the stats table beats the phase baseline by 0.0062 (95% CI 0.0053–0.0071) and the embedding
  table by 0.0010 (CI 0.0002–0.0019). Stats win.
- **Clusters are not stable.** Bootstrap ARI is 0.47 / 0.34 (stats, batter / bowler) and about 0.01 for
  embeddings. A role-based re-cluster gives one stable batter clustering (k=3, ARI 0.69) and no stable bowler one.
- **A hierarchical backoff chain shows archetypes add nothing beyond player rates** (exploratory re-score of the
  spent test seasons, labelled as such in `reports/p5b/`).
- **Agent** (`gpt-oss-120b`, free tier): 30 mixed questions score 1.00 numeric accuracy on the 24 scored,
  0.83 tool-choice accuracy and 6/6 correct refusals; a 20-question tool-coverage set scores 0.95 numeric.
  Serving reproduces the reported validation log-loss exactly (`reports/p6/serving.json`).

## How it works

```
Cricsheet JSON ─► parse (registry IDs) ─► DuckDB views ─► features ─► models
                                                                      │
          baselines → cric2vec / LightGBM → archetypes → matchup table → hierarchical chain
                                                                      │
              train-only serving artifact ─► read-only tools ─► LLM agent ─► Streamlit dashboard
```

| Layer | Code | What it does |
|---|---|---|
| Data | `src/cricmind/data/` | Parse 10,804 men's T20 matches (2,472,642 deliveries) keyed by Cricsheet registry IDs |
| Models | `src/cricmind/models/` | Phase, logistic and LightGBM baselines; cric2vec embeddings |
| Archetypes | `src/cricmind/archetypes/` | Stats and embedding vectors, k-means / GMM / HDBSCAN, stability ladder |
| Matchups | `src/cricmind/matchups/` | Empirical-Bayes shrunk archetype table; hierarchical backoff chain |
| Agent | `src/cricmind/agent/` | Tools: profile, top players, matchup, what-if, archetype, next ball, SELECT-only SQL |
| Eval | `src/cricmind/eval/` | Metrics with bootstrap CIs; 30 + 20 generated agent questions |

Guards that matter: scalers, embeddings, clusters and shrinkage priors are fitted on train only; test seasons are
scored once per phase behind a lock; the agent uses a read-only DuckDB connection and a SELECT-only SQL tool with a
row limit; secrets stay in `.env`.

## Run it

```bash
uv sync
uv run ruff check . && uv run pytest -q          # tests run on small committed fixtures
cp .env.example .env                              # set LLM_BASE_URL, LLM_MODEL, LLM_API_KEY
uv run --env-file .env streamlit run src/cricmind/dashboard.py
```

The dashboard has tabs for each tool; every player is a searchable dropdown keyed by registry ID. All tabs work
without an LLM key except "Ask the agent". The full data and the serving artifact are built locally and are not
in the repo (`data/` and `artifacts/` are gitignored).

## Limits

- Cricsheet has no handedness or bowling style, so archetypes come from performance only.
- About 70% of test balls involve a player pair without both labels and fall back to the phase baseline.
- The agent was evaluated on one free-tier model; the questions are generated from the same data it queries.
- Public demo and web UI are not deployed yet.

## Data attribution

Ball-by-ball data and the player register come from [Cricsheet](https://cricsheet.org), released under the
[Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/).
Any output derived from this data must credit Cricsheet.
