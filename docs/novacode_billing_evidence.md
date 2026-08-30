# NovaCode billing reconciliation

Formal `official750` validation requires a separate, offline billing evidence
file. The repository intentionally contains no NovaCode price constants: the
provider's actual rate card or invoice/export must be supplied by the user.
STATE-Bench trajectory `cost_usd` values are not used because a zero value only
means that the agent did not report provider pricing.

The evidence format is defined by
`docs/novacode_billing_evidence.schema.json`; start from
`configs/novacode_billing_evidence.template.json` (rate card) or
`configs/novacode_invoice_evidence.template.json` (invoice/export). The
templates deliberately contain invalid placeholders and cannot accidentally
pass as real evidence.

Schema `1.2.0` binds both the verified session-chain manifest and the complete
unique relay-ledger manifest. Evidence created against the earlier single-ledger
shape must be regenerated from the final result tree.

Automatic validation accepts only a machine-readable JSON source whose values
are read directly by the reconciler. Its schema is
`docs/novacode_machine_readable_evidence.schema.json`; the corresponding
source shapes are `configs/novacode_rate_card_source.template.json` and
`configs/novacode_invoice_source.template.json`. A PDF, screenshot, or arbitrary
hashed file may be retained for a human audit, but it cannot produce a machine
PASS. If NovaCode exports a different machine-readable shape, add and test a
dedicated parser for that provider export rather than copying prices into the
outer evidence file.

Two evidence modes are accepted:

- `rate_card`: attach a NovaCode-issued rate-card file and enter its actual
  uncached-input, cached-input, and output rates per million tokens. The script
  multiplies those rates by the complete set of unique verified-session relay
  ledgers (agent, simulator, judge, and any retry response for which the
  provider reported usage).
- `invoice_export`: attach a NovaCode invoice/usage export scoped to this exact
  official candidate run. Its four token totals must match that complete
  verified ledger set exactly; an unallocated account-wide invoice is not
  accepted.

In both modes the JSON source must exist locally and match its declared
SHA-256. Rates, currency, invoice usage, and invoice amount are extracted from
that source rather than accepted from the outer evidence file.
`pricing_authority` must be `novacode_provider`; OpenAI list prices and
zero-cost placeholders are rejected. The provider-origin hash must also match
the NovaCode upstream origin recorded by the run manifests. Every relay ledger
referenced by a verified fresh or Resume session is SHA-256-bound and parsed
exactly once. Official fresh execution is one five-run batch per domain; its
five run projections share one log and ledger segment, and the reconciler never
multiplies that shared segment by five. The initial manifest ledger must be
referenced by every verified fresh projection; additional ledgers are admitted
only through later records in those immutable chains. Unreferenced
`_transport/relay-*.jsonl` files, unreferenced `_batch_records/*.json`, path
escapes or aliases, conflicting declarations, and session origin/transport
drift are rejected. Each final trajectory's agent-call total must exactly match
responses across the complete ledger set under its unique audit ID. The currency is reported as supplied and is
never silently converted. Because this is offline validation, the report also
states that provider-signature authenticity was not cryptographically verified;
the source hash preserves exactly what a human auditor reviewed.

After the fresh 750-run tree exists, compute the expected binding without
reading task text or judge reasoning:

```powershell
& '..\STATE-Bench\.venv\Scripts\python.exe' -c `
  "from pathlib import Path; import json; from scripts.reconcile_novacode_billing import compute_result_binding; print(json.dumps(compute_result_binding(Path(r'outputs/selective_pwm/official750/candidate-C'))[0], indent=2))"
```

Copy that exact object into `binding`, fill the provider evidence, then run:

```powershell
& '..\STATE-Bench\.venv\Scripts\python.exe' scripts/reconcile_novacode_billing.py `
  --candidate outputs/selective_pwm/official750/candidate-C `
  --evidence C:\secure\novacode-official750-evidence.json `
  --output outputs/selective_pwm/official750/candidate-C/billing_reconciliation.json
```

The report binds the evidence JSON hash and source-document hash to a canonical
manifest of all 750 trajectory files, the sorted unique relay-ledger set, every
verified per-domain/per-run session chain, the three immutable run manifests,
and the memory/router/runner/protocol/STATE-Bench hashes recorded by those
manifests. `relay_ledger_manifest_sha256` commits to ledger paths, hashes, and
verified reference counts; `session_chain_manifest_sha256` commits to the
records that authorized those ledgers. Every final trajectory must carry a unique
32-hex `provider_request_audit_id` and a 64-hex `provider_task_key` equal to
`SHA256(domain|task_id)`. The reconciler matches its agent token counters only
to successful ledger responses carrying that exact audit ID and task key; it
does not read conversation or requirement text. The source billing document may remain
outside Git; only its hash is needed for audit.

Audited Resume can leave provider-billable calls from attempts that did not
produce one of the final 750 trajectories. Those audit IDs are reported under
`token_usage.abandoned_agent_attempts` and remain included in
`provider_billable_usage`, rate-card cost, and invoice matching. They are never
folded into `final_agent_usage` or compared against a replacement trajectory.

Finally pass the same evidence file to formal validation:

```powershell
& '..\STATE-Bench\.venv\Scripts\python.exe' scripts/validate_official_submission.py `
  --candidate outputs/selective_pwm/official750/candidate-C `
  --billing-evidence C:\secure\novacode-official750-evidence.json
```
