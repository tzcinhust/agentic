# LatticeGuard mechanism pilot: result and failure audit

Date: 2026-09-01  
Branch: `statebench-latticeguard-pilot`  
Baseline commit: `152bf488e8be2538db118dce73a8743c8c9dd8e1`

## Verdict

The diagnostic hypothesis passed, but the deployable-SOTA hypothesis has not passed.

- Procedural-memory injection is observably non-monotone. The full three-card PWM arm passed 6/10 tasks, while no memory, top-1 only, top-3 only, and ranks 2+3 each passed 7/10.
- A single globally fixed subset does not improve completion over the no-memory arm on this panel. Therefore the experiment does not justify an official test run or an SOTA claim.
- The best arm is task-dependent. An outcome oracle can find a passing arm for 10/10 tasks, versus 6/10 for full PWM. This is headroom for a router, not an achievable result: it selects after observing all eight outcomes.
- The reliable signal is currently UX and negative interaction, not completion. Relative to no memory, top-1 improves mean UX by 0.512 points (task bootstrap 90% interval 0.051 to 0.935) and top-3 improves it by 0.396 (0.159 to 0.651), without a certified completion gain.

## Protocol

- Data: a deterministic SHA-256-selected panel of 10 Shopping **training** tasks. No test task, test environment, or historical test failure was used.
- Intervention: all eight subsets of the unchanged baseline PWM top-three retrieval positions. Mask bits 1, 2, and 4 denote ranks 1, 2, and 3; mask 0 is no memory and mask 7 is the original full PWM injection.
- Frozen components: Actor, GPT-5.4 model, simulator, judges, tools, workflow-card text, and original ranking.
- Main panel: 10 tasks x 8 masks x 1 run = 80 scored trajectories.
- Stochasticity preflight: 2 of the 10 tasks x 8 masks x 3 runs = 48 scored trajectories, of which run 1 is also part of the main panel.
- Execution: official STATE-Bench v0.8.1 harness and inline task/state/UX scoring; 0 run errors, 0 unscored outputs, and 0 protocol errors.
- Important limitation: STATE-Bench exposes run indices but not shared deterministic simulator/model seeds. Cross-arm runs are randomized blocked observations, not exact common-random-number pairs. The task-level bootstrap is therefore primary; per-run Möbius coefficients are descriptive only.

## Ten-task arm results

| Mask | Injected ranks | Completion | State | Task | UX / 5 | Mean tokens | Completion losses vs empty |
|---:|---|---:|---:|---:|---:|---:|---:|
| 0 | none | 7/10 | 9/10 | 7/10 | 3.732 | 24,629 | 0 |
| 1 | rank 1 | 7/10 | 9/10 | 7/10 | **4.244** | 25,259 | 1 |
| 2 | rank 2 | 6/10 | 9/10 | 6/10 | 3.783 | 31,588 | 1 |
| 3 | ranks 1+2 | 5/10 | 9/10 | 5/10 | 3.675 | 35,159 | 2 |
| 4 | rank 3 | 7/10 | 9/10 | 7/10 | 4.128 | 30,191 | 1 |
| 5 | ranks 1+3 | 6/10 | 9/10 | 6/10 | 4.055 | 30,113 | 1 |
| 6 | ranks 2+3 | 7/10 | 9/10 | 7/10 | 3.885 | 43,674 | 1 |
| 7 | ranks 1+2+3 (PWM) | 6/10 | 9/10 | 6/10 | 4.230 | 36,543 | 2 |

Top-1 versus full PWM is +1 completion and +1 task pass, approximately equal UX (+0.014), and 30.9% fewer tokens. The paired task-bootstrap 90% interval for its completion difference is -0.3 to +0.5, so this is not a certified success improvement.

## Interaction evidence

The absolute interaction mass is 0.80 for completion/task and 0.732 for UX. With only ten tasks, completion interactions have wide intervals and are not individually resolved. Two UX interactions are informative:

| Möbius term | Meaning | Mean UX contribution | Task-bootstrap 90% interval |
|---:|---|---:|---:|
| mask 3 | rank 1 x rank 2 | -0.620 | -1.448 to +0.254 |
| mask 5 | rank 1 x rank 3 | **-0.585** | **-1.165 to -0.044** |
| mask 7 | rank 1 x rank 2 x rank 3 | **+1.038** | **+0.209 to +1.847** |

The positive third-order term does not imply that full PWM is best. It offsets some negative lower-order effects, while the final full-PWM completion remains 6/10. This is exactly why independent singleton scoring or monotone/submodular assumptions are unsafe here.

## Representative failure causes

1. **Shipping refresh: full PWM diluted a load-bearing write.** On `110-hard_four_to_five_items_shipping_refresh`, rank 1 alone called `set_shipping_option` after the fifth item was added and produced the required free shipping and $435 total. Full PWM omitted the refresh, retained a stale $6 shipping charge, reported $441, and failed state and task requirements. All three injected cards discussed shipping; more nominally relevant text still reduced execution reliability.

2. **Invalid promo: promo memories encouraged unsupported explanation.** On `52-invalid_promo_tool_error_recovery`, rank 3 alone honestly limited the explanation to the tool's “not found” result and passed. Full PWM speculated that the code may have been advertised incorrectly, never been a checkout code, or been retired. The judge explicitly disallowed every cause not returned by the tool. This is semantic overreach caused by extra procedural context, not a state-execution error.

3. **Category-restricted promo: one generic card under-disclosed the consequence.** On `77-category_restriction_silent_drop_on_edit`, rank 1 alone said the promo “may” no longer apply and failed the required explicit disclosure. Ranks 2+3 and full PWM clearly detected and disclosed the KITCHEN10 removal and passed. Some tasks genuinely require a complementary portfolio.

4. **Brand-bundle pivot: full context did not guarantee the correct conversational timing.** On `50-brand_bundle_hold_and_pivot`, full PWM held the policy line and eventually added a valid same-brand product, but failed the judge's required proactive recommendation timing. Rank 1 passed. This shows that the mechanism affects process realization even when final state is correct.

## Go/no-go decision

- **Keep:** the complete subset intervention agent, exact Möbius analyzer, task-level/hierarchical bootstrap, and the research claim that memory portfolios are non-additive.
- **Do not promote:** any fixed mask as an SOTA candidate. No fixed mask has a supported completion advantage, and full PWM remains stochastic.
- **Do not run official test yet:** selecting rules from test results would invalidate the benchmark, while the train-only evidence has not cleared a completion gate.
- **Next falsifiable step:** fit the predeclared task/card interaction predictor and simultaneous conformal harm certificate on a larger train-only fit/calibration split, then evaluate once on a disjoint train lockbox. The runtime policy must abstain to no memory when the certificate cannot exclude harm. If the lockbox does not improve completion while preserving state and UX, retire LatticeGuard rather than adding task-specific rules.

## Reproducibility artifacts

- `panel10_analysis.json`: complete 10-task arm, contrast, bootstrap, and per-task report.
- `analysis_runs1_3.json`: two-task, three-run stochasticity audit.
- `latticeguard_mechanism_panel10.log`: official harness execution log for the 64 newly collected trajectories.
