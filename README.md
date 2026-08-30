# Cross-Domain Process Workflow Memory for STATE-Bench

This branch archives the cross-domain Process Workflow Memory experiment run
on STATE-Bench 0.8.1 at upstream commit
`5644b1838d96bc4483da29642d058ecaa6f80f7f`.

## Selective Decision-Aware PWM

This branch adds a fail-open retrieval sidecar without changing the archived
`ProcessWorkflowMemoryAgent`, its actor loop, or the v1 workflow-card file. The
new `RiskAwareProcessWorkflowMemoryAgent` supports three cumulative ablations:

- `A`: selective risk-aware reranking with true greedy MMR and adaptive 0--3
  card retrieval; returned cards remain the original v1 text.
- `B`: A plus deterministic typed-card packing. The primary card is rendered as
  `WHEN/READ/DECIDE/WRITE/VERIFY/SAY/NEVER`; secondary cards contribute compact
  constraints unless their write is explicit in the request.
- `C`: B plus a latest-intent/recent-tool state query, neutral bounded State-Q
  prior, and FlowSwitch-Lite active-card reuse.

Only Shopping is promoted in the frozen sidecar. Travel and Customer Support
remain byte-for-byte v1 PWM fallbacks until each passes its own promotion gate.
The training trajectories contain no scored outcomes, so the generated
State-Q prior is deliberately neutral (`0.5`, coverage `0`) rather than using
invented labels. No runtime model call is added.

The sidecar builder reads only
`../STATE-Bench/datasets/train_task_trajectories`, a v1 card file built from the
same allowed partition, and the checked-in train-ID split manifest. Dev and
lockbox evaluation must first build the isolated optimizer-80 pair. This is a
strict GPT-5.4/NovaCode build: malformed output or an exhausted transport
failure aborts the whole build, and fallback is available only through an
explicit `--no-llm` developer invocation.

```powershell
.\scripts\build_optimizer80_artifacts.ps1 -Workers 2
```

That command reads 80 training trajectories per domain, never opens the 10 dev
or 10 lockbox trajectory bodies, and installs
`process_workflows_optimizer80.json` plus
`workflow_router_v2_optimizer80.json`. The final full-100 sidecar is rebuilt
only after the cumulative C candidate has passed the held-out lockbox gate.
The guarded builder re-validates that immutable paired evidence under the
current commit, then rebuilds both v1 memory and its v2 sidecar from all 100
training trajectories per domain:

```powershell
.\scripts\build_full100_artifacts.ps1 `
  -LockboxBaseline outputs/selective_pwm/lockbox/baseline `
  -LockboxCandidate outputs/selective_pwm/lockbox/candidate-C `
  -Workers 2
```

After both artifacts pass validation, it installs
`artifacts/statebench_cross_domain_pwm/memory/process_workflows.json` and
`artifacts/statebench_cross_domain_pwm/memory/workflow_router_v2.json`, which
contains source hashes, the dev/lockbox/optimizer splits, typed contracts,
frozen weights, thresholds, utility provenance, and per-domain promotion
flags. Existing full-100 artifacts are backed up inside the build run
directory before replacement. The JSON Schema is at
`docs/workflow_router_v2.schema.json`.

Run the offline checks before spending evaluation budget:

```powershell
& '..\STATE-Bench\.venv\Scripts\python.exe' -m pytest tests -q
```

The staged evaluator reads the local STATE-Bench `.env` without printing or
copying its API key. It pins GPT-5.4 and hybrid memory, starts its own hashed
45-RPM/burst-5 relay, writes an append-only status/usage ledger with no prompt
or response text, fixes the agent client at 4096 max tokens, 120-second
timeout, zero client retries, and temperature 0. Every run is bound to the
clean repository commit and implementation hashes, and each arm is written to
a fresh directory. Dev, lockbox, and paired-150 keep one-run fresh calls;
official-750 uses exactly one five-run official test call per domain
(`--num-runs 5 --num-runs-idx-start 1`) with a shared immutable batch record.
`-Resume` is evidence-driven: it never touches a
fully scored pass or failure, score-only retries raw trajectories in an
isolated directory, and reruns one Agent task at a time only for a missing
trajectory whose latest immutable session proves an exhausted transport
failure. The Agent, official simulator/judge, and relay retry layers are
recorded separately in every run manifest. The complete
contract is in `docs/auditable_resume_protocol.md`. For example:

```powershell
.\scripts\run_selective_pwm.ps1 -Stage dev -Arm baseline -StartRelay
.\scripts\run_selective_pwm.ps1 -Stage dev -Arm candidate -RouterStage A -StartRelay

& '..\STATE-Bench\.venv\Scripts\python.exe' scripts/evaluate_gate.py `
  --gate dev `
  --baseline outputs/selective_pwm/dev/baseline `
  --candidate outputs/selective_pwm/dev/candidate-A
```

Advance from A to B to C only when the preceding arm passes the configured
gate. The exact dev, lockbox, paired-150, and official-750 thresholds live in
`configs/evaluation_gates.json`. Before claiming an official result, validate
the complete fresh C-stage output (including all protocol/model metadata and
all 750 scored observations):

```powershell
& '..\STATE-Bench\.venv\Scripts\python.exe' scripts/validate_official_submission.py `
  --candidate outputs/selective_pwm/official750/candidate-C
```

## Policy-Guard Extension

This development branch adds the post-archive policy-obligation and verification
experiments without replacing the recovered PWM implementation. The main added
agent is `LoyaltyVerifiedPolicyAgent`, which combines process workflow memory,
retrieved policy obligations, and a narrow pre-commit verifier for loyalty
mutations. The additional agent modules preserve the intermediate ablation arms
used to evaluate late-bound policy injection, post-tool review, obligation
ledgers, and broader write verification.

Policy obligations are mined from the fixed training trajectories with
`scripts/mine_policy_obligations.py` and stored at
`artifacts/statebench_cross_domain_pwm/memory/policy_obligations.json`. The
evaluation relay helper is available at `tools/eval_shim.py`; it changes only
transport routing and does not modify benchmark prompts, task definitions, or
protocol files.

No API credentials are committed. Configure provider credentials through local
environment variables before replaying an experiment.

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
