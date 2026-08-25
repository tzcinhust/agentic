# Executable Process Workflow Memory for STATE-Bench

This repository contains an experimental Agent Learning Track method for
[Microsoft STATE-Bench](https://github.com/microsoft/STATE-Bench). It builds
process-conformant workflow cards from fixed training trajectories, retrieves
relevant cards at inference time, and verifies proposed state-changing actions
and final responses before they are returned to the benchmark harness.

## Status

This branch is an archive of the best development snapshot that produced the
11/20 result. It is intentionally separate from later Shopping fixes. The
runtime agent and client were behaviorally reconstructed from the Python 3.10
bytecode left by that run; the bytecode evidence is stored under
`artifacts/statebench_best_055/evidence/`.

For the closest faithful replay of the archived run, use the prebuilt memory artifact at
`artifacts/statebench_best_055/memory/process_workflows.json`. Do not rebuild
it with the current workflow builder and call the result the archived run:
the builder was changed after the run. Remote model output may still vary.

The current code targets STATE-Bench 0.8.1 and was compatibility-checked against
upstream commit `5644b1838d96bc4483da29642d058ecaa6f80f7f`.

The following numbers are development results on one deterministic 80/20 split
of the 100 public Shopping Assistant training trajectories. They are not an
official held-out test result or a leaderboard claim.

| Method | Completion | State | Task | UX |
| --- | ---: | ---: | ---: | ---: |
| Structured workflow memory | 8/20 | 17/20 | 8/20 | 3.691 |
| Executable verifier, run 1 | 11/20 | 19/20 | 11/20 | 4.242 |

The 20-task split was subsequently used for error analysis, so it must now be
treated as a development set. This archive intentionally excludes fixes made
after the run; those changes require a fresh end-to-end benchmark evaluation.

The archived logs and all 20 trajectories are under
`artifacts/statebench_best_055/`. The score is a one-run local analysis only;
the protocol warning says that five runs are required for a compliant result.

## Method

1. Split the public training trajectories into a deterministic build and
   development partition.
2. Group build trajectories by intent and discover frequent process variants.
3. Use public-train requirements as supervision to avoid imitating failed
   trajectories.
4. Induce structured workflow cards containing preconditions, branches,
   disclosures, confirmation gates, refresh obligations, and forbidden actions.
5. Compile those fields into `require_tool`, `require_confirmation`, `disclose`,
   `refresh`, and `forbid` runtime rules.
6. Retrieve up to three diverse cards with lexical, character, process-support,
   and intent-specific scoring.
7. Check deterministic invariants and semantic workflow compliance before
   writes and final responses. Rejected candidates receive concrete feedback
   and are regenerated up to a configured limit.

The agent does not read `runtime_context.task_summary`, hidden test
requirements, or test environment state during retrieval.

## Install

Clone STATE-Bench and copy this repository's `agents`, `clients`, and `scripts`
directories into its repository root. Use the benchmark's Python 3.12+ `uv`
environment:

```bash
git clone https://github.com/microsoft/STATE-Bench.git
cd STATE-Bench
uv sync
```

Set provider configuration through environment variables. Do not commit keys:

```bash
export WORKFLOW_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
export WORKFLOW_LLM_API_KEY="..."
export WORKFLOW_LLM_MODEL="your-model"

export STATE_BENCH_AGENT_BASE_URL="https://your-openai-compatible-endpoint/v1"
export STATE_BENCH_AGENT_API_KEY="..."
export STATE_BENCH_AGENT_MODEL="your-model"
export STATE_BENCH_AGENT_TIMEOUT_SECONDS=120
export STATE_BENCH_AGENT_MAX_RETRIES=6
```

Simulator and judge credentials must be configured according to STATE-Bench's
locked evaluation-client protocol.

## Build A Development Artifact

Create the deterministic split:

```bash
uv run python scripts/split_train_validation.py \
  --source datasets/train_task_trajectories \
  --output outputs/shopping_train_split \
  --domain shopping_assistant
```

Build structured executable workflow memory from only the 80 build
trajectories:

```bash
uv run python scripts/build_process_workflows.py \
  --data-root outputs/shopping_train_split/build \
  --output outputs/shopping_memory/process_workflows.json \
  --domains shopping_assistant \
  --structured \
  --llm-base-url "$WORKFLOW_LLM_BASE_URL" \
  --llm-model "$WORKFLOW_LLM_MODEL" \
  --cache-dir outputs/shopping_memory/workflow_cache
```

The builder may read matching public-train task requirements as supervision.
It must not read held-out test task definitions or environments.

## Run

```bash
export STATE_BENCH_MEMORY_PATH=outputs/shopping_memory/process_workflows.json
export STATE_BENCH_MEMORY_MODE=hybrid
export STATE_BENCH_VERIFIER_MODE=full
export STATE_BENCH_VERIFIER_MAX_REVISIONS=2
export STATE_BENCH_VERIFIER_MIN_CONFIDENCE=0.7

uv run python -m state_bench.scripts.run_batch \
  --domain shopping_assistant \
  --agent-class ProcessWorkflowMemoryAgent \
  --agent-client-class OpenCodeLLMClient \
  --agent-model-name "$STATE_BENCH_AGENT_MODEL" \
  --retrieve-learnings-top-k 3 \
  --num-runs 5 \
  --output-dir outputs/shopping_assistant
```

Use the official test split, locked simulator and judge, five runs, and official
metrics/submission commands before making any leaderboard claim.

## Tests

From the STATE-Bench repository root after copying these files:

```bash
uv run pytest \
  tests/test_build_process_workflows.py \
  tests/test_process_workflow_memory.py \
  tests/test_opencode_client.py -q
```

The runnable values for the archived `.55` snapshot are also listed in
`configs/statebench_best_055.env.example`. Provider URL and API key are left
blank by design.
