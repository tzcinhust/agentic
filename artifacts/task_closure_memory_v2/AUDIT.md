# Task-Closure Memory v2 mechanism audit

## Verdict

The 30-task probe does **not** validate the intended low-interference Final-Closure
hypothesis. The implementation is mechanically runnable and produces a small descriptive
gain (14/30 versus archived PWM's 12/30), but the run predominantly tested a different
mechanism: pervasive pre-action and pre-claim guards injected into tool-capable turns.

The core Shopping failures remain because the current system has three coupled problems:

1. it cannot reliably discover a proactive obligation before the missing evidence has
   already been fetched;
2. once obligations exist, noisy retrieval and irreversible semantic bookkeeping corrupt
   or crowd out the important one;
3. the exposure priority almost never reaches Final Closure.

The observed +2 should therefore be treated as a probe result, not a stable improvement,
causal ablation, or SOTA evidence.

## What actually ran

```text
observable conversation
        |
        v
retrieve a changing top-8 completion set on every generation
        |
        v
same GPT-5.4 endpoint performs semantic bookkeeping
        |
        v
append/update the lifecycle ledger
        |
        +-- any open pre_action item --> inject action_guard
        |
        +-- otherwise pending evidence --> inject claim_guard
        |
        +-- otherwise final item -------> inject final_closure
        |
        v
tool-enabled frozen-PWM generation
```

This ordering is implemented in
[`_exposure()`](../../agents/task_closure_memory_agent.py#L340-L403), and the selected
prompt is injected before every main generation in
[`generate_next_turn()`](../../agents/task_closure_memory_agent.py#L420-L474).

Across the 30 tasks:

| Diagnostic | Observed |
| --- | ---: |
| Main generations | 239 |
| Generations with any Completion prompt | 235 (98.3%) |
| Action-guard exposures | 165 |
| Claim-guard exposures | 67 |
| Final-closure exposures | 3 |
| Exposure followed by tool call | 147 |
| Semantic bookkeeper calls | 269 |
| Semantic bookkeeper tokens | 3,411,015 |
| Mean initial items/task | 4.03 |
| Mean final items/task | 24.30 |
| Maximum final items in one task | 54 |

All three Final-Closure exposures occurred on one unchanged-pass task, Shopping 28.
None of the four gains or two regressions received Final Closure. Consequently, the
score changes cannot be attributed to the claimed Final-Closure component.

## Critical findings

### P0-1: Final Closure is starved by its own gate

The exposure policy checks actionable items first, then unresolved evidence, and only
then final items. See
[`task_closure_memory_agent.py:344-390`](../../agents/task_closure_memory_agent.py#L344-L390).
Because noisy learned templates continually create pre-action and pending-evidence items,
the first two branches dominate indefinitely.

Empirical consequence:

- 232/235 injections were action or claim guards;
- true Final Closure appeared in only 3/239 main generations;
- it affected no gain or regression.

The experiment therefore did not meaningfully test the mechanism named in the paper
story. Adding more templates or prompt wording cannot repair this selection-order issue.

### P0-2: the implementation does not preserve frozen PWM execution

The code calls the bookkeeper and injects its selected guard into the same tool-capable
generation that performs normal planning
([`task_closure_memory_agent.py:434-449`](../../agents/task_closure_memory_agent.py#L434-L449)).
The strict "ignore closure when calling tools" instruction exists only in the
Final-Closure text; action and claim guards have no equivalent isolation. Even a written
instruction could not prove non-interference because the model reads it before choosing
between a tool call and final text.

The run confirms that the mechanism was present on planning turns: 147 injected
generations returned tool calls, and mean tool calls rose from 8.1 to 9.4 per task.
Without a concurrent counterfactual this does not prove which individual calls changed,
but it disproves the stronger claim that planning saw the frozen PWM context.

Customer Support 34 is the clearest regression path. Both archived PWM and the new run
previewed the out-of-stock exchange. PWM then received a normal approval turn and executed
the fallback; the new guard-exposed run did not clearly invite confirmation, and the
simulator's approval arrived with `[TASK_DONE]`, leaving the mutation unexecuted. One sample
cannot prove the guard caused the simulator difference, but the result shows why a
tool-capable guard path cannot be presented as frozen execution.

### P0-3: terminal lifecycle states can contradict their own evidence

[`_transition()`](../../agents/completion_lifecycle.py#L222-L253) refuses every update from
`satisfied`, `invalidated`, or `violated`. However,
[`merge_semantic()`](../../agents/completion_lifecycle.py#L682-L690) overwrites an existing
item's description and `missing_evidence` before attempting that refused transition.

This creates impossible states such as:

```text
status = satisfied
missing_evidence = ["assistant still needs to report ..."]
```

The probe contains 28 such items across 11/30 tasks. A false-positive satisfaction is
also irreversible, so later evidence cannot reopen an achievement. In Shopping 4,
several final-report obligations remained `satisfied` while explicitly listing missing
assistant communication. This directly undermines the claim that the ledger tracks what
remains.

Only a historically observed **invariant violation** should normally be irreversible.
Achievement status and evidence need an atomic, reversible update policy.

## Major findings

### P1-1: v2 remains a Completion Accumulator

Completion retrieval runs again on every generation
([`_sync_bookkeeper()`](../../agents/task_closure_memory_agent.py#L216-L250)). Its query
includes recent assistant text and all observed tool fields
([`completion_query()`](../../agents/completion_templates.py#L126-L151)), so the retrieved
set drifts as the agent talks and acts. New templates create new items, while terminal
items remain in the ledger.

Observed growth is 4.03 initial items to 24.30 final items per task, with an average of
21.43 distinct retrieved templates per task despite nominal top-k 8. There are also 2.43
duplicate normalized descriptions per task after schema-level duplication.

The learned artifact amplifies this behavior:

- 300 source trajectories;
- 995 merged templates and 2,362 obligations;
- 951/995 templates have support 1.

This is closer to a collection of per-trajectory generations than a consolidated library
of reusable completion semantics. Exact model-generated family names fragment equivalent
conditions, while the 0.45 obligation merge threshold can also merge superficially
similar but operationally different requirements. See
[`merge_templates()`](../../scripts/build_completion_templates.py#L494-L569).

### P1-2: evidence references are syntactic, not evidential

The semantic update accepts an evidence object when its `M#` or `T#` reference merely
exists in the compact transcript
([`completion_lifecycle.py:618-655`](../../agents/completion_lifecycle.py#L618-L655)).
There is no deterministic check that the referenced content entails applicability,
satisfaction, invalidation, or violation.

The result is severe over-classification: final ledgers contain 513 satisfied and 96
violated items, despite only 30 short tasks. Travel 30 marked generic proactive-warning
conditions satisfied and unrelated invariants violated, while the exact next-tier warning
the judge required remained `pending_evidence` even though the policy and nightly rate
were present in tool results.

No bookkeeper response failed to parse, so this is not an API reliability problem. It is
a semantic validation problem.

### P1-3: proactive discovery has a circular dependency

Shopping 16, 17, and 29 all required the agent to inspect the customer profile and then
proactively disclose welcome/loyalty benefits. The relevant templates were retrieved in
early top-8 candidate sets, but their triggers require an observed account/profile signal.
The bookkeeper therefore declined to instantiate them before `get_customer_account` had
run. Because Completion Memory is forbidden from choosing the next tool, the missing
profile evidence was never acquired.

The cycle is:

```text
need profile evidence to activate obligation
        ^                         |
        |                         v
need active obligation to notice profile evidence should be acquired
```

This is why a purely final or monitor-only Completion Memory cannot recover this class of
failure. The design needs an explicit, domain-independent **evidence-need bridge**: the
completion side may state what evidence is unresolved, while Procedure Memory remains
responsible for selecting how to obtain it. Without such a bridge, proactive obligations
that are absent from the user's wording are structurally undiscoverable.

### P1-4: priority and prompt limits starve the right condition

`actionable_items()` sorts model-assigned priorities and `_dedupe()` keeps only four;
claim and final prompts are also capped
([`task_closure_memory_agent.py:298-316`](../../agents/task_closure_memory_agent.py#L298-L316)).
No calibrated relevance, confidence, support, or evidence readiness enters prompt-slot
selection.

Travel 30 demonstrates the failure. The correct learned condition—warn about the exact
next consequence when waiting crosses the hotel boundary—was present and remained open.
It was not surfaced because action guards containing false violations occupied the prompt.
The agent reported the current $90 tier but omitted the future $180 warning, exactly the
State-Pass/Task-Fail pattern the method was intended to fix.

### P1-5: confirmation binding is too dependent on assistant phrasing

[`_sync_confirmations()`](../../agents/completion_lifecycle.py#L498-L546) accepts an
affirmative user message only when the immediately preceding assistant text matches a
confirmation-request regex. A user can explicitly say "go ahead" after a preview even if
the assistant did not phrase the previous message as a question; that approval remains
unbound.

Customer Support 34 ended with exactly this state: one execution remained
`awaiting_confirmation` and its choice item remained `pending`, despite the user explicitly
authorizing the $59 store-credit fallback. The final simulator message included
`[TASK_DONE]`, so there was no later turn in which the agent could recover.

### P1-6: the cost target failed by a wide margin

Mean total tokens rose from 43,112 to 158,360 per task, a 3.67x multiplier. The 269
bookkeeper calls used 3.41M tokens, about 71.8% of all method tokens. The exact pattern was
one online bookkeeping call per main generation plus one post-trajectory call per task.

The fingerprint cache does not help during normal interaction because every new message,
tool result, candidate set, or ledger update changes the fingerprint. The success criterion
that token overhead stay near PWM is not met.

## Experimental-validity findings

### P2-1: the +2 difference is not a causal estimate

The method run was compared with an archived PWM run rather than a concurrent `pwm_only`
run. Both use one stochastic model/simulator sample. The six discordant tasks split 4 gains
versus 2 regressions; that is too little evidence to separate mechanism effect from sample
variance. The 30 tasks are also a fixed diagnostic subset, not a random or complete test
split.

Notably, none of the changed tasks received Final Closure. Travel 9 and 33 changed state;
that cannot be evidence for a closure-only mechanism and is consistent with either
planning interference or ordinary sample variance.

### P2-2: final ledger statistics include post-hoc bookkeeping

`ingest_trajectory()` performs one forced semantic sync after the episode has ended
([`task_closure_memory_agent.py:476-495`](../../agents/task_closure_memory_agent.py#L476-L495)).
The resulting `final_items`, status counts, and bookkeeper totals therefore mix online
agent state with a post-hoc update that could not affect behavior. Future logging should
separate `online_final_snapshot` from `posthoc_audit_snapshot`.

### P2-3: tests prove mechanics, not the claimed system property

The tests correctly cover preview handling, scoped confirmations, no regeneration, and a
mock `pwm_only` equality path. They do not cover:

- reopening a falsely satisfied achievement;
- rejecting `satisfied + missing_evidence`;
- explicit user authorization without a preceding confirmation question;
- prompt-starvation under mixed action/claim/final items;
- a real trace where Final Closure is the only behavioral difference;
- an empirical upper bound on planning interference or bookkeeping overhead.

The existing test for unresolved pre-claim evidence explicitly expects claim guard to
replace Final Closure
([`test_task_closure_memory_agent.py:267-309`](../../tests/test_task_closure_memory_agent.py#L267-L309)),
so the dominant behavior is currently treated as intended rather than detected as a
failure of the low-interference hypothesis.

## What worked

- The artifact is independently built from 300 fixed conversations and does not read
  `task_summary` or `task_requirements` at inference.
- Preview results are not treated as completed mutations.
- Confirmations are scoped instead of globally satisfying every pending confirmation.
- Main-agent generation has no reject/regenerate loop.
- All 30 trajectories were scored, and every semantic-bookkeeper response parsed.
- The exact code commit, completion-artifact hash, raw trajectories, logs, and checksums
  are preserved with the experiment.

These are useful implementation foundations, but they do not establish effectiveness.

## Recommended correction order

No further API experiment should run before the first four items are addressed offline.

1. **Repair lifecycle atomicity.** Reject contradictory status/evidence updates; permit
   achievements to reopen; make only evidence-backed invariant violations irreversible.
2. **Separate the mechanisms.** Add a true `closure_only` path with action/claim guards
   monitor-only. Do not call the current `full` mode Final Closure in tables.
3. **Consolidate the artifact.** Cluster equivalent families, require support or held-out
   train validation, and calibrate a retrieval threshold instead of forcing eight positive
   matches from 995 mostly singleton templates.
4. **Add the evidence-need bridge.** Represent unresolved evidence without naming tools;
   let Procedure Memory decide how to acquire it. This is necessary for welcome, loyalty,
   bundle, and deadline conditions that the user does not mention.
5. **Make exposure eligibility evidence-aware.** Surface a small requirement only when its
   trigger, scope, and supporting evidence are resolved; do not let unrelated terminal
   violations monopolize action slots.
6. **Handle explicit approval independently of question wording.** Bind a user's scoped
   authorization to the latest compatible preview even when the preceding assistant did
   not contain a confirmation-request phrase.
7. **Repeat one targeted paired probe only after offline tests pass.** Run `pwm_only` and
   the corrected method concurrently on the same 30 tasks. That single comparison is more
   informative than adding further overlapping mechanisms.

The central design choice must be stated honestly: either Completion Memory is strictly
final-only and cannot solve missing-evidence Shopping tasks, or it supplies evidence needs
before closure and is allowed to influence execution indirectly. The current implementation
claims the former while behaving like the latter, which is the main reason both the story
and the empirical result remain unstable.
