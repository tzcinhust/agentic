# TAPM Full Test Run 1

This directory contains the complete available artifacts from the first full
test run of Transition-Aware Process Memory (TAPM) on STATE-Bench.

## Configuration

- Source commit: `0ac6338`
- Agent: `ProcessWorkflowMemoryAgent`
- Agent model: `gpt-5.4`
- Memory mode: `hybrid`
- Verifier mode: `full`
- TAPM mode: `enforce`
- Learned, entity-aware, and value-aware TAPM checks: enabled
- Retrieval top-k: 3
- STATE-Bench protocol: `state_bench_v0.8.1_gpt54`
- Runs: 1
- Workers per domain: 1
- Inline task/state and UX scoring: enabled

## Results

| Domain | Completed trajectories | Task completion | State requirements | Task requirements | Mean UX |
| --- | ---: | ---: | ---: | ---: | ---: |
| Shopping Assistant | 49/50 | 21/50 (42%)* | 45/50 (90%)* | 21/50 (42%)* | 3.684/5** |
| Customer Support | 50/50 | 28/50 (56%) | 41/50 (82%) | 32/50 (64%) | 3.996/5 |
| Travel | 50/50 | 27/50 (54%) | 40/50 (80%) | 29/50 (58%) | 3.514/5 |
| Overall | 149/150 | 76/150 (50.67%)* | 126/150 (84%)* | 82/150 (54.67%)* | — |

\* Shopping task `13-hard_expired_seeded_promo_repair_after_add` failed on
both retry attempts because the agent exceeded the benchmark limit of eight
tool rounds. It produced no trajectory and is conservatively counted as a
failure in the table. Consequently, STATE-Bench's official metrics command
rejects the Shopping run as incomplete.

\** Shopping UX is the mean over the 49 scored trajectories; the missing task
has no UX score.

Official one-run metric files are included for Customer Support and Travel.
The protocol warns that benchmark submissions require five complete runs, so
these results are local experimental evidence rather than a leaderboard or
SOTA claim.

## Historical Run-1 Comparison

The previous workflow-memory run reported 46% Shopping, 66% Customer Support,
and 68% Travel task completion. It was produced with an earlier code state and
a separate stochastic model run, so it is useful context but not a strict
same-code TAPM ablation.

## Directory Layout

- `<domain>/run1/`: scored task trajectories.
- `<domain>/metrics.json`: standardized one-run metrics where the run is complete.
- `<domain>/per_task_metrics/`: standardized per-task metric records where available.
- `<domain>.txt`: captured batch-run log, renamed from `.log` for version control.

No API keys, endpoint credentials, SSH credentials, or local environment files
are included.
