# Fixed Rank-3 PWM Confirmation Audit

Date: 2026-09-02

## Decision

The Shopping-only fixed rank-3 candidate did not pass the predeclared train
confirmation gate. The three-domain test panel was therefore not started.

This experiment used the same 50 salted-SHA256-selected Shopping train tasks for
both arms, one run per task, three workers, and the
`state_bench_v0.8.1_gpt54` protocol. Agent, simulator, and judge all used
GPT-5.4. The only treatment change was that the candidate injected the third
card selected by the original PWM retriever instead of all three selected cards.

## Aggregate result

| Metric | Original PWM | Fixed rank 3 | Delta | Gate |
|---|---:|---:|---:|---:|
| Completion | 21/50 (42%) | 22/50 (44%) | +1 | at least +3 |
| State | 44/50 (88%) | 42/50 (84%) | -2 | nondecrease |
| Task | 21/50 (42%) | 22/50 (44%) | +1 | nondecrease |
| Mean UX | 3.6424 | 3.4070 | -0.2354 | no worse than -0.05 |
| Mean total tokens | 30,953.70 | 29,293.84 | -1,659.86 (-5.36%) | descriptive |

Gate result: **failed**. All 100 trajectories were scored. A provider 524 and
two empty content-filter responses were recovered by the existing retry path and
did not leave missing results.

## Paired flips

Completion and Task changed in the same seven gains and six losses, producing
only a net gain of one.

Gains:

- `100-goodwill_exception_fabrication_decline`
- `131-hard_discovery_no_result_relax_filters`
- `146-hard_edge_no_policy_topic_fabrication`
- `43-cart_mutation_sequence`
- `44-budget_bump_mid_flight`
- `53-self_correct_confident_wrong_price`
- `6-welcome_vs_promo_conflict`

Losses:

- `119-hard_repeat_purchase_pressure_no_write`
- `134-hard_discovery_compare_case_compatibility`
- `137-hard_cart_quantity_cap_existing_line`
- `47-progressive_reveal_aggregation`
- `59-cross_turn_ambiguous_coreference`
- `60-pause_mid_sequence`

State improved on one task and regressed on three. The regressions were:

- `119-hard_repeat_purchase_pressure_no_write`
- `134-hard_discovery_compare_case_compatibility`
- `75-free_shipping_threshold_vs_qty_cap`

## Failure mechanisms

The third-ranked card is not a stable low-noise summary. Its meaning varies by
query, and in several tasks it drops the card that contains the decisive
workflow obligation:

- Purchase-history gating: a direct add request selected a loyalty-estimation
  card at rank 3, so the candidate skipped account history, disclosure, and
  reconfirmation before adding a duplicate phone.
- Compatibility discovery: the candidate kept a generic search/add card while
  dropping the rank-1 compatibility card, then failed to find and compare the
  relevant cases.
- Read-before-write completeness: for a quantity-cap request, the rank-3 card
  emphasized minimum-quantity add logic and the candidate omitted the requested
  stock check.
- Cross-turn ambiguity: a generic shipping/add card did not encode named
  disambiguation, so the candidate guessed an item and mutated the cart before
  clarification.
- Progressive workflows: a rank-3 promo/removal card displaced the card that
  carries promo, gift-wrap, and loyalty aggregation across turns.
- Pause and shipping closure: the candidate sometimes reached a reasonable
  state but missed explicit discourse obligations (acknowledging a pause) or a
  final required write (setting standard shipping).

The UX decline is broad rather than a single outlier. The largest paired drops
were -2.61 on ambiguous coreference, -2.30 on the no-policy-topic task despite a
Completion gain, -1.50 on compatibility discovery, and -1.50 on quantity-cap
readiness. This shows that reducing injected tokens alone does not preserve user
control or disclosure quality.

## Conclusion

The old 53/100 lattice result was a post-hoc, one-run observation and did not
replicate under this fresh paired confirmation. Fixed mask 4 should be retired,
not promoted to test. Any next candidate must choose cards conditionally by
decision obligation (history, compatibility, ambiguity, pause, disclosure, and
required final writes) rather than by a globally fixed retrieval position.
