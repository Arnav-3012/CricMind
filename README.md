# CricMind

[![CI](https://github.com/Arnav-3012/CricMind/actions/workflows/ci.yml/badge.svg)](https://github.com/Arnav-3012/CricMind/actions/workflows/ci.yml)

CricMind discovers T20 player archetypes from Cricsheet ball-by-ball data and asks one question: do archetypes
built from learned player embeddings predict held-out delivery outcomes better than archetypes built from
hand-made per-phase stats? Archetypes feed a batter-archetype × bowler-archetype matchup table, and a tool-using
agent answers tactical questions from it with numbers that come from tool output. Every layer is evaluated on
held-out seasons, and a negative result (stats beat embeddings) is reported as-is.

## Results

TBD

## Data attribution

Ball-by-ball data and the player register come from [Cricsheet](https://cricsheet.org), released under the
[Open Data Commons Attribution License v1.0 (ODC-By 1.0)](https://opendatacommons.org/licenses/by/1-0/).
Any output derived from this data must credit Cricsheet.
