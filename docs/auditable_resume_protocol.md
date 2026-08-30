# Auditable Resume Protocol

`scripts/run_selective_pwm.ps1` treats each `(domain, run)` as an immutable
evidence unit. Dev, lockbox, and paired-150 keep their existing one-run fresh
calls. Official-750 instead invokes the official CLI exactly once per domain as
`run_batch --split test --num-runs 5 --num-runs-idx-start 1`.

That one call produces a self-hashed `auditable_fresh_batch` record binding the
exact command contract, one immutable batch log, one relay-ledger byte range,
and runs 1 through 5. Five first-session projections reference the same batch
record and add each run's own pre/post trajectory hashes. All five projections
must exist and agree before any chain verifies; a crash during evidence writing
therefore fails closed. Subsequent Resume records remain ordinary per-run hash
chains and never repeat the five-run fresh command.

The records contain only task IDs, file hashes, state labels, exit codes, and
transport counts. They never store task definitions, requirements, user text,
prompts, responses, or exception text.

## Resume decisions

`-Resume` first verifies the complete record chain, log hashes, relay segment
hashes, manifest hash, and current trajectory hashes. It then applies these
rules independently to every expected trajectory:

- `scored`: immutable, including `task_completion_pass=0`; never run or score
  again.
- `unscored`: eligible only when its newest record proves an exhausted
  transport failure during scoring. The official `state_bench.scripts.score`
  CLI runs for that one trajectory in an isolated `_resume_tmp` root. The
  canonical raw trajectory is replaced atomically only after a complete GPT-5.4
  task/state and UX score passes validation.
- `missing`: eligible only when its newest record ties that task to an
  exhausted transport failure during Agent/simulator execution. Each eligible
  ID is passed to its own one-run, one-task `run_batch` invocation and its own
  immutable session record.
- no proof, stale proof, non-transport failure, hash mismatch, or malformed
  evidence: reject without executing or scoring anything.

Every normal Resume invocation appends a new record. A repeated transport
failure can therefore be retried only from the newer failure proof. A
successful scored failure is final, not a reason to resample.

## Locked execution contract

The run manifest distinguishes three retry layers. The custom Agent client is
locked to GPT-5.4, 4096 output tokens, a 120-second timeout, zero client
retries, and temperature 0. In pinned STATE-Bench v0.8.1, the official
simulator/judge constructors leave the OpenAI 2.16 SDK default of two retries
in place and the benchmark wrapper has `Tenacity max_attempts=5`; the runner
cannot disable either through a supported environment option without modifying
the pinned benchmark. These are request retries, not scored-trajectory
resampling, and every request still passes through the attributable ledger.
The relay is independently fixed at 45 RPM, burst 5 per 1 second, and five
attempts for retryable transport/status failures. Official-750 execution
remains fixed at two workers.

Use the same command that created the arm and add `-Resume`; every other
manifest-bound argument must remain identical:

```powershell
.\scripts\run_selective_pwm.ps1 `
  -Stage dev `
  -Arm candidate `
  -RouterStage A `
  -StartRelay `
  -Resume
```

Resume is intentionally conservative. If a process was interrupted before a
session record could be completed, the missing proof is not reconstructed from
task contents; that output root is ineligible for automatic recovery.

## Validator and billing integration

The execution validator can import `verify_session_chain` and `snapshot_run`
from `scripts.resume_protocol`. For every domain and expected run it should:

1. require a non-empty, contiguous chain whose first record is `fresh`; for
   official-750, require all five first records to reference one verified
   `auditable_fresh_batch` with the exact official command contract;
2. require every record's `run_manifest_sha256`, `record_sha256`, previous
   record hash, batch-log hash, relay-prefix hash, and relay-segment hash to
   verify;
3. require the latest recorded post-state for every manifest task to equal the
   current trajectory state and SHA-256, and require every final state to be
   `scored`;
4. enforce the transition allowlist: `fresh` starts from missing,
   `resume_agent` targets only missing, `resume_score` targets only unscored,
   and no-op/rejected records target no trajectories;
5. reject unknown record fields or any record value outside the documented
   IDs, hashes, state labels, numeric counters, and fixed protocol literals.

`verify_session_chain(...)` performs the cryptographic and path-containment
checks. `plan_resume(...)` additionally binds the latest task state to the
chain and is the same decision function used by the runner. The CLI equivalents
used by PowerShell are `resume_protocol.py snapshot`, `plan`, `record`,
`record-official-batch`, `stage-score`, and `promote-score`.

Billing must not look only at the initial ledger named in
`run_manifest.transport`: each Resume starts a new attributable relay. Collect
the unique `relay_segment.relative_path` values from verified session records,
validate their `relay_session` contracts, and count each unique ledger's usage
events once. The five official fresh projections deliberately reference the
same batch segment; that segment is an evidence claim, not five billable copies.
Do not add overlapping prefixes or per-record segment summaries to usage totals.
The initial manifest ledger must match the fresh batch; any additional ledger
is admissible only through a verified Resume record chained to that same
manifest. Unreferenced ledgers and unreferenced fresh-batch records are rejected.
