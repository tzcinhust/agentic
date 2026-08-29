# Cross-Domain Process Workflow Memory and Task-Closure Memory for STATE-Bench

This branch archives the cross-domain Process Workflow Memory experiment run
on STATE-Bench 0.8.1 at upstream commit
`5644b1838d96bc4483da29642d058ecaa6f80f7f`.

## Task-Closure Memory v2 probe

This branch also contains the independently induced Task-Closure Memory v2
implementation and its first fixed 30-task probe. It scored 14/30 versus 12/30
for the same task IDs in archived PWM run1, but the audit found that 235/239
main generations received a guard while true Final Closure appeared only three
times. The result is diagnostic evidence, not a causal improvement or
leaderboard result.

- [Experiment record](artifacts/task_closure_memory_v2/experiments/task_closure_v2_subset30_99f9bfc_r1a/README.md)
- [Full mechanism audit](artifacts/task_closure_memory_v2/AUDIT.md)
- [Machine-readable paired summary](artifacts/task_closure_memory_v2/experiments/task_closure_v2_subset30_99f9bfc_r1a/paired_summary.json)

## Archived Result

The completed evidence is one 50-task test run per domain:

| Domain | Completion | State | Task | UX |
| --- | ---: | ---: | ---: | ---: |
| Shopping Assistant | 23/50 (0.46) | 45/50 | 23/50 | 3.916 |
| Travel | 34/50 (0.68) | 41/50 | 35/50 | 3.888 |
| Customer Support | 33/50 (0.66) | 43/50 | 36/50 | 4.357 |

These are local one-run results, not protocol-compliant five-run leaderboard
results. The attempted continuation stopped with 114 additional unscored
trajectories: Shopping 54, Travel 4, and Customer Support 56. They remain in
the archive as partial evidence and are excluded from the table.

## Recovery Evidence

The runtime snapshot was recovered from the original Codex execution record:

- The complete initial agent source and its only pre-run patch were replayed.
- The recovered agent is 9,057 bytes, matching the recorded upload size.
- The exact 24,509-byte builder and 1,883,823-byte memory artifact survived on
  the experiment server.
- The exact client and base agent survived in the local pre-run adapter copy.
- The original launch commands, logs, trajectories, per-task metrics, and
  aggregate metrics survived.

No original agent hash or pre-run bytecode survived, so this is a source-chain
recovery rather than the bytecode-equivalence proof used by the `.55` archive.
Remote model output may vary on replay even with identical code and inputs.

## Method

The builder groups fixed training trajectories by process family and induces
148 workflow cards: 54 Travel, 44 Customer Support, and 50 Shopping Assistant.
At inference time the agent retrieves up to three distinct families using
BM25-style lexical matching, character overlap, process support, conformance,
quality, tool overlap, and domain-specific intent matching. The cards are
injected as procedural guidance; current identifiers, state, price, and policy
must still be verified with live tools.

## Replay

Copy `agents`, `clients`, and `scripts` into the root of the pinned STATE-Bench
checkout. Use the archived memory file directly:

```bash
export STATE_BENCH_MEMORY_PATH=artifacts/statebench_cross_domain_pwm/memory/process_workflows.json
export STATE_BENCH_MEMORY_MODE=hybrid
export STATE_BENCH_AGENT_BASE_URL="https://your-openai-compatible-endpoint/v1"
export STATE_BENCH_AGENT_API_KEY="..."
export STATE_BENCH_AGENT_MODEL=gpt-5.4
export STATE_BENCH_AGENT_MAX_TOKENS=4096
export STATE_BENCH_AGENT_TIMEOUT_SECONDS=120
export STATE_BENCH_AGENT_MAX_RETRIES=6
```

The original command for each of `travel`, `customer_support`, and
`shopping_assistant` was:

```bash
python -u -m state_bench.scripts.run_batch \
  --domain "$DOMAIN" \
  --split test \
  --num-runs 5 \
  --num-workers 8 \
  --no-score \
  --agent-class ProcessWorkflowMemoryAgent \
  --agent-client-class OpenCodeLLMClient \
  --retrieve-learnings-top-k 3 \
  --agent-model-name gpt-5.4 \
  --output-dir "outputs/pwm_gpt54_proxy_nova/$DOMAIN"
```

The exact run assets are under `artifacts/statebench_cross_domain_pwm/runs/`.

## Tests

Run inside the pinned STATE-Bench environment:

```bash
python -m pytest tests/test_process_workflow_memory.py tests/test_opencode_client.py -q
```
