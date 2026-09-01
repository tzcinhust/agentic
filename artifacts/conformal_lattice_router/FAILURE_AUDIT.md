# Conformal lattice router: train-only failure audit

Date: 2026-09-02  
Branch: `statebench-conformal-lattice-router`  
Baseline: original three-card PWM (`mask 7`)

## Verdict

The candidate is **retired and not deployed**. The router passed its fit gate,
but failed the independent calibration safety certificate. The planned
three-domain test panel was therefore not started (0 test trajectories).

This is a safety-certificate failure, not a claim that every routed decision
was harmful. The untouched lockbox aggregate was encouraging, but it cannot
override a gate that had already failed on calibration:

- Fit OOF: completion `+4/60`, state `+1/60`, task `+3/60`, UX `+0.065`.
- Calibration: completion `0/20`, state `+1/20`, task **`-1/20`**, UX `+0.122`.
- Lockbox: completion `+1/20`, state `0/20`, task `+2/20`, UX `-0.011`.
- Calibration conformal harm: 5/20 positive-harm cases; one-sided 90% harm
  quantile `1.0`, so the certificate failed.

Weakening the certificate after seeing calibration would be post-hoc target
movement. The artifact consequently keeps `deployment_enabled=false`, and the
runtime fails closed to the original PWM.

## Protocol and data isolation

- Official local harness: `state_bench_v0.8.1_gpt54`.
- Model for Actor, simulator, and judges: GPT-5.4; judge reasoning effort high.
- Data: Shopping **train trajectories only**. No test task, test environment,
  historical test failure, requirement text, or test judge reasoning was used.
- Intervention: all eight subsets of the unchanged PWM top-three retrieval
  positions, `100 train tasks × 8 masks × 1 run = 800` scored trajectories.
- All 800 records were present and scored; there were no task-run, protocol, or
  scoring failures.
- 70 trajectories were reused from the earlier panel only after confirming the
  same evaluation protocol, agent source hash, and memory-artifact hash. The
  other 730 trajectories were newly generated.
- Deterministic salted SHA-256 split: 60 fit, 20 calibration, 20 lockbox. The
  deployment artifact stores only counts and a manifest hash, never task IDs or
  user text.
- Runtime calls added by the router: zero.

## Complete train-lattice aggregate

Each arm below has 100 independently sampled GPT-5.4 trajectories. The table
is descriptive; run indices are not shared deterministic random seeds.

| Mask | Injected ranks | Completion | State | Task | UX / 5 | Utility |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0 | none | 44/100 | 84/100 | 45/100 | 3.509 | 0.535 |
| 1 | rank 1 | 51/100 | 88/100 | 53/100 | **3.711** | 0.599 |
| 2 | rank 2 | 51/100 | 86/100 | 51/100 | 3.622 | 0.591 |
| 3 | ranks 1+2 | 46/100 | 80/100 | 46/100 | 3.498 | 0.540 |
| 4 | rank 3 | **53/100** | 87/100 | **55/100** | 3.632 | **0.611** |
| 5 | ranks 1+3 | 45/100 | 85/100 | 46/100 | 3.662 | 0.546 |
| 6 | ranks 2+3 | 46/100 | 85/100 | 47/100 | 3.474 | 0.551 |
| 7 | ranks 1+2+3 (PWM) | 46/100 | 86/100 | 48/100 | 3.593 | 0.556 |

The larger experiment confirms the earlier mechanism result: procedural-memory
portfolios are non-monotone. Adding cards can be worse than a singleton, and
the full three-card prompt is not the best fixed arm on train data.

## Router and certificate

The selected model was multi-output ridge with `λ=10`, chosen by deterministic
five-fold OOF error from `{0.1, 1, 10, 100}`. The frozen policy required a
predicted utility gain of at least `0.10`, no predicted completion/state/task
loss, UX loss no worse than `0.02`, and otherwise abstained to mask 7.

| Split | Routed alternatives | Completion Δ | State Δ | Task Δ | UX Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fit OOF (60) | 27 | +4 | +1 | +3 | +0.065 |
| Calibration (20) | 10 | 0 | +1 | **-1** | +0.122 |
| Lockbox (20) | 11 | +1 | 0 | +2 | -0.011 |

On calibration, routed alternatives produced one completion loss and one win,
one state loss and two wins, and two task losses versus one win. Five selected
alternatives had positive harm under the predeclared max-metric score. This is
incompatible with the required 90% harm quantile of zero.

## Failure cause

The central failure is **benefit prediction without per-decision harm
calibration**:

1. On calibration, the model predicted mean selected utility gain `+0.1042`,
   while realized gain was only `+0.0037`. The selection margin was therefore
   sharply over-optimistic.
2. Binary completion/task targets are sparse and stochastic. Sixty fit tasks
   are insufficient for a 66-feature, eight-arm point predictor to resolve
   task-specific negative transfer reliably, even with ridge shrinkage.
3. The policy's aggregate fit improvement hid tail harm. Conformal evaluation
   correctly exposed that the same router could lose required process steps on
   individual tasks.
4. The lockbox improvement shows genuine subset headroom, but not a valid
   certificate for this fitted router. It does not justify bypassing the
   earlier calibration failure.

## What remains useful

- Keep the exact subset-lattice intervention and the empirical claim that PWM
  card portfolios exhibit strong non-monotone interactions.
- Keep the fail-closed runtime, privacy-safe features, fixed trajectory-level
  decision, and zero additional inference calls.
- Retire the learned eight-arm point router and do not patch it using
  calibration or lockbox task identities.

The strongest next falsifiable candidate is substantially simpler: predeclare
the fixed rank-3 singleton (`mask 4`) against full PWM and run a fresh
train-only stochastic confirmation. Mask 4 is `+7` completion and `+7` task
passes over full PWM in the 100-task descriptive lattice, but that choice is
now post-hoc and must not be promoted directly. A new confirmation should use
only aggregate gates and must precede any test evaluation.

