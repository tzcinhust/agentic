from __future__ import annotations

import hashlib
import json
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.reconcile_novacode_billing import (  # noqa: E402
    BillingEvidenceError,
    canonical_sha256,
    collect_relay_usage,
    compute_result_binding,
    file_sha256,
    reconcile_billing,
)
from scripts.resume_protocol import (  # noqa: E402
    snapshot_run,
    task_ids_sha256,
    write_official_fresh_batch_records,
    write_session_record,
)
from scripts.validate_official_submission import _billing_checks  # noqa: E402


DOMAINS = ("shopping_assistant", "travel", "customer_support")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
            for value in values
        ),
        encoding="utf-8",
    )


def _session_header(origin_hash: str) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "event": "session_start",
        "provider": "novacode",
        "upstream_origin_sha256": origin_hash,
        "rpm": 45,
        "burst": 5,
        "burst_window_seconds": 1.0,
        "attempts": 5,
    }


def _header_size(header: dict[str, Any]) -> int:
    return len(
        (json.dumps(header, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
    )


def _write_fresh_batch_sessions(
    candidate: Path,
    *,
    domain: str,
    task_ids: list[str],
    ledger_relative_path: str,
) -> tuple[Path, ...]:
    pre_snapshots = {
        run: {
            "schema_version": "1.1.0",
            "task_ids_sha256": task_ids_sha256(task_ids),
            "tasks": {
                task_id: {"state": "missing", "sha256": None}
                for task_id in task_ids
            },
        }
        for run in range(1, 6)
    }
    log_path = candidate / "_batch_logs" / domain / "fresh-runs1-5.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("fresh completed\n", encoding="utf-8")
    ledger_path = candidate / ledger_relative_path
    header = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    record_paths = write_official_fresh_batch_records(
        arm_root=candidate,
        domain=domain,
        run_manifest_path=candidate / domain / "run_manifest.json",
        all_task_ids=task_ids,
        pre_snapshots=pre_snapshots,
        log_path=log_path,
        relay_ledger_path=ledger_path,
        relay_start_offset=_header_size(header),
        process_exit_code=0,
        split="test",
        num_runs=5,
        num_runs_idx_start=1,
    )
    for record_path in record_paths:
        record_path.chmod(0o666)
    log_path.chmod(0o666)
    for run in range(1, 6):
        for trajectory_path in (candidate / domain / f"run{run}").glob("*.json"):
            trajectory_path.chmod(0o666)
    return record_paths


def _append_noop_session(
    candidate: Path,
    *,
    domain: str,
    run: int,
    ledger_relative_path: str,
    ledger_records: list[dict[str, Any]],
) -> Path:
    ledger_path = candidate / ledger_relative_path
    _write_jsonl(ledger_path, ledger_records)
    manifest = json.loads(
        (candidate / domain / "run_manifest.json").read_text(encoding="utf-8")
    )
    task_ids = manifest["run"]["task_selection"]["task_ids"]
    log_path = candidate / "_sessions" / domain / f"run{run}" / "resume-noop.log"
    log_path.write_text("resume noop completed\n", encoding="utf-8")
    record_path = write_session_record(
        arm_root=candidate,
        domain=domain,
        run_index=run,
        run_manifest_path=candidate / domain / "run_manifest.json",
        mode="resume_noop",
        all_task_ids=task_ids,
        target_task_ids=[],
        pre_snapshot=snapshot_run(candidate / domain / f"run{run}", task_ids),
        log_path=log_path,
        relay_ledger_path=ledger_path,
        relay_start_offset=_header_size(ledger_records[0]),
        process_exit_code=0,
    )
    record_path.chmod(0o666)
    log_path.chmod(0o666)
    return record_path


def _rebind_session_records_to_ledger(candidate: Path, relative: str) -> None:
    ledger_bytes = (candidate / relative).read_bytes()
    header_line = ledger_bytes.splitlines(keepends=True)[0]
    start = len(header_line)
    batch_hashes: dict[str, str] = {}
    for batch_path in sorted((candidate / "_batch_records").glob("*/*.json")):
        batch_path.chmod(0o666)
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch["relay_segment"]["relative_path"] == relative:
            batch["relay_segment"].update(
                {
                    "start_offset": start,
                    "end_offset": len(ledger_bytes),
                    "prefix_sha256": hashlib.sha256(ledger_bytes[:start]).hexdigest(),
                    "sha256": hashlib.sha256(ledger_bytes[start:]).hexdigest(),
                }
            )
        batch["batch_record_sha256"] = canonical_sha256(
            {
                key: value
                for key, value in batch.items()
                if key != "batch_record_sha256"
            }
        )
        _write_json(batch_path, batch)
        batch_hashes[batch_path.resolve().relative_to(candidate.resolve()).as_posix()] = batch[
            "batch_record_sha256"
        ]
    for session_dir in sorted((candidate / "_sessions").glob("*/run*")):
        previous_hash: str | None = None
        for record_path in sorted(session_dir.glob("*.json")):
            record_path.chmod(0o666)
            record = json.loads(record_path.read_text(encoding="utf-8"))
            record["previous_session_sha256"] = previous_hash
            if record["relay_segment"]["relative_path"] == relative:
                record["relay_segment"].update(
                    {
                        "start_offset": start,
                        "end_offset": len(ledger_bytes),
                        "prefix_sha256": hashlib.sha256(
                            ledger_bytes[:start]
                        ).hexdigest(),
                        "sha256": hashlib.sha256(ledger_bytes[start:]).hexdigest(),
                    }
                )
            fresh_batch = record.get("fresh_batch")
            if isinstance(fresh_batch, dict):
                fresh_batch["sha256"] = batch_hashes[fresh_batch["relative_path"]]
            record["record_sha256"] = canonical_sha256(
                {
                    key: value
                    for key, value in record.items()
                    if key != "record_sha256"
                }
            )
            _write_json(record_path, record)
            previous_hash = record["record_sha256"]


@pytest.fixture(scope="module")
def billing_fixture(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("novacode-billing")
    candidate = root / "candidate-C"
    artifact_hashes = {
        "memory_sha256": "1" * 64,
        "router_sha256": "2" * 64,
        "runner_sha256": "3" * 64,
        "state_bench_protocol_sha256": "4" * 64,
        "state_bench_commit": "5" * 40,
    }
    origin_hash = hashlib.sha256(b"https://ai.novacode.top").hexdigest()
    ledger_relative_path = "_transport/relay-synthetic.jsonl"
    ledger_records: list[dict[str, Any]] = [_session_header(origin_hash)]
    request_id = 0
    trajectory_number = 0
    final_agent_audits: dict[str, dict[str, Any]] = {}
    domain_task_ids = {
        domain: [f"{domain}-task-{task:02d}" for task in range(50)]
        for domain in DOMAINS
    }
    first_task_key = ""
    for domain in DOMAINS:
        manifest = {
            "schema_version": "1.0.0",
            "created_by": "scripts/run_selective_pwm.ps1",
            "stage": "official750",
            "arm": "candidate",
            "router_stage": "C",
            "domain": domain,
            "run": {
                "task_selection": {
                    "task_ids": domain_task_ids[domain],
                }
            },
            "artifacts": artifact_hashes,
            "transport": {
                "provider": "novacode",
                "upstream_origin_sha256": origin_hash,
                "relay_sha256": "6" * 64,
                "ledger_relative_path": ledger_relative_path,
            },
        }
        manifest["manifest_sha256"] = canonical_sha256(manifest)
        _write_json(candidate / domain / "run_manifest.json", manifest)
        for run in range(1, 6):
            for task in range(50):
                trajectory_number += 1
                task_id = domain_task_ids[domain][task]
                audit_id = f"{trajectory_number:032x}"
                task_key = hashlib.sha256(f"{domain}|{task_id}".encode("utf-8")).hexdigest()
                first_task_key = first_task_key or task_key
                agent_usage = {
                    "input_tokens": 1000,
                    "cached_input_tokens": 100,
                    "output_tokens": 100,
                    "reasoning_output_tokens": 10,
                    "total_tokens": 1100,
                }
                final_agent_audits[audit_id] = {
                    "task_key": task_key,
                    "usage": agent_usage,
                }
                _write_json(
                    candidate / domain / f"run{run}" / f"{task_id}.json",
                    {
                        "task_id": task_id,
                        "domain": domain,
                        "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
                        "scoring_protocol_id": "state_bench_v0.8.1_gpt54",
                        "simulator_model": "gpt-5.4",
                        "judge_model": "gpt-5.4",
                        "judge_reasoning_effort": "high",
                        "simulator_prompt_hash": "7" * 64,
                        "judge_prompt_hashes": {
                            "judge_task_requirements.md": "8" * 64
                        },
                        "agent_model": {
                            "model_name": "gpt-5.4",
                            "reasoning_level": None,
                        },
                        "task_completion_pass": 1,
                        "state_requirements_met": 1,
                        "task_requirements_met": 1,
                        "ux_score": 4.0,
                        "provider_request_audit_id": audit_id,
                        "provider_task_key": task_key,
                        # Zero provider-agnostic cost must be ignored, not
                        # mistaken for real NovaCode billing.
                        "cost_usd": 0,
                        "token_usage": {**agent_usage, "total_cost_usd": 0},
                    },
                )
                request_id += 1
                ledger_records.append(
                    {
                        "schema_version": "1.0.0",
                        "event": "upstream_response",
                        "request_id": request_id,
                        "attempt": 1,
                        "route": "agent_chat_completions",
                        "audit_id": audit_id,
                        "task_key": task_key,
                        "status_code": 200,
                        "retryable_status": False,
                        "usage": agent_usage,
                    }
                )
                request_id += 1
                ledger_records.append(
                    {
                        "schema_version": "1.0.0",
                        "event": "upstream_response",
                        "request_id": request_id,
                        "attempt": 1,
                        "route": "official_eval_responses",
                        "status_code": 200,
                        "retryable_status": False,
                        "usage": {
                            "input_tokens": 200,
                            "cached_input_tokens": 0,
                            "output_tokens": 20,
                            "reasoning_output_tokens": 5,
                            "total_tokens": 220,
                        },
                    }
                )
    abandoned_audit_id = "f" * 32
    for status_code, retryable, abandoned_usage in (
        (
            200,
            False,
            {
                "input_tokens": 50,
                "cached_input_tokens": 5,
                "output_tokens": 10,
                "reasoning_output_tokens": 1,
                "total_tokens": 60,
            },
        ),
        (
            429,
            True,
            {
                "input_tokens": 20,
                "cached_input_tokens": 0,
                "output_tokens": 5,
                "reasoning_output_tokens": 0,
                "total_tokens": 25,
            },
        ),
    ):
        request_id += 1
        ledger_records.append(
            {
                "schema_version": "1.0.0",
                "event": "upstream_response",
                "request_id": request_id,
                "attempt": 1,
                "route": "agent_chat_completions",
                "audit_id": abandoned_audit_id,
                "task_key": first_task_key,
                "status_code": status_code,
                "retryable_status": retryable,
                "usage": abandoned_usage,
            }
        )
    _write_jsonl(candidate / ledger_relative_path, ledger_records)
    for domain in DOMAINS:
        _write_fresh_batch_sessions(
            candidate,
            domain=domain,
            task_ids=domain_task_ids[domain],
            ledger_relative_path=ledger_relative_path,
        )
    source_payload = {
        "schema_version": "novacode_machine_readable_evidence_v1",
        "provider": "NovaCode",
        "document_type": "provider_rate_card",
        "model": "gpt-5.4",
        "currency": "USD",
        "issued_at": "2026-08-30",
        "rates_per_million_tokens": {
            "uncached_input": "1",
            "cached_input": "0.1",
            "output": "10",
        },
    }
    source = root / "novacode-rate-card.json"
    _write_json(source, source_payload)
    binding, usage = compute_result_binding(candidate)
    evidence = {
        "schema_version": "1.2.0",
        "provider": {
            "name": "NovaCode",
            "model": "gpt-5.4",
            "base_url_origin_sha256": origin_hash,
        },
        "binding": binding,
        "evidence": {
            "mode": "rate_card",
            "pricing_authority": "novacode_provider",
            "source_document": {
                "path": source.name,
                "sha256": file_sha256(source),
                "document_class": "provider_rate_card",
            },
        },
    }
    evidence_path = root / "evidence.json"
    _write_json(evidence_path, evidence)
    return {
        "root": root,
        "candidate": candidate,
        "source": source,
        "source_payload": source_payload,
        "binding": binding,
        "usage": usage,
        "final_agent_audits": final_agent_audits,
        "evidence": evidence,
        "evidence_path": evidence_path,
    }


def _variant(fixture: dict[str, Any], name: str, value: dict[str, Any]) -> Path:
    path = fixture["root"] / name
    _write_json(path, value)
    return path


def _candidate_variant(
    fixture: dict[str, Any], tmp_path: Path, name: str
) -> Path:
    destination = tmp_path / name
    shutil.copytree(fixture["candidate"], destination)
    return destination


def _ledger_records(candidate: Path, fixture: dict[str, Any]) -> tuple[Path, list[dict[str, Any]]]:
    relative = fixture["binding"]["transport"]["ledger_relative_path"]
    path = candidate / relative
    return path, [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_rate_card_reconciles_all_750_and_ignores_trajectory_cost_zero(
    billing_fixture: dict[str, Any],
) -> None:
    report = reconcile_billing(
        billing_fixture["candidate"], billing_fixture["evidence_path"]
    )
    assert report["passed"] is True
    assert report["token_usage"]["observations"] == 750
    assert report["token_usage"]["input_tokens"] == 750_000
    assert report["token_usage"]["cached_input_tokens"] == 75_000
    assert report["token_usage"]["output_tokens"] == 75_000
    assert report["token_usage"]["final_agent_usage"]["total_tokens"] == 825_000
    abandoned = report["token_usage"]["abandoned_agent_attempts"]
    assert abandoned["audit_id_count"] == 1
    assert abandoned["successful_response_records_with_usage"] == 1
    assert abandoned["retry_or_failed_response_records_with_usage"] == 1
    assert abandoned["provider_billable_usage"]["total_tokens"] == 85
    assert report["token_usage"]["provider_billable_usage"]["total_tokens"] == 990_085
    assert report["cost"]["reconciled_amount"] == "1.7327155"
    assert report["cost"]["mean_per_observation"] == "0.002310287333333333333333333333"
    assert report["accounting_notes"]["trajectory_cost_fields_trusted"] is False
    relay = report["token_usage"]["relay"]
    assert relay["ledger_count"] == 1
    assert relay["session_record_count"] == 15
    assert relay["ledgers"][0]["reference_count"] == 15
    assert report["binding"]["relay_ledger_manifest_sha256"] == relay[
        "relay_ledger_manifest_sha256"
    ]
    assert report["binding"]["session_chain_manifest_sha256"] == relay[
        "session_chain_manifest_sha256"
    ]


def test_resume_ledger_is_billed_and_classified_across_all_ledgers(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "resume-ledger")
    origin_hash = billing_fixture["binding"]["transport"]["upstream_origin_sha256"]
    task_key = next(iter(billing_fixture["final_agent_audits"].values()))["task_key"]
    resume_usage = {
        "input_tokens": 10,
        "cached_input_tokens": 2,
        "output_tokens": 1,
        "reasoning_output_tokens": 0,
        "total_tokens": 11,
    }
    resume_relative = "_transport/relay-resume.jsonl"
    _append_noop_session(
        candidate,
        domain="shopping_assistant",
        run=1,
        ledger_relative_path=resume_relative,
        ledger_records=[
            _session_header(origin_hash),
            {
                "schema_version": "1.0.0",
                "event": "upstream_response",
                "request_id": 1,
                "attempt": 1,
                "route": "agent_chat_completions",
                "audit_id": "e" * 32,
                "task_key": task_key,
                "status_code": 200,
                "retryable_status": False,
                "usage": resume_usage,
            },
        ],
    )

    binding, usage = compute_result_binding(candidate)

    relay = usage["relay"]
    assert relay["ledger_count"] == 2
    assert relay["session_record_count"] == 16
    assert {item["relative_path"] for item in relay["ledgers"]} == {
        billing_fixture["binding"]["transport"]["ledger_relative_path"],
        resume_relative,
    }
    assert usage["final_agent_usage"]["total_tokens"] == 825_000
    assert usage["provider_billable_usage"]["total_tokens"] == 990_096
    abandoned = usage["abandoned_agent_attempts"]
    assert abandoned["audit_id_count"] == 2
    assert abandoned["provider_billable_usage"]["total_tokens"] == 96
    assert binding["relay_ledger_manifest_sha256"] != billing_fixture["binding"][
        "relay_ledger_manifest_sha256"
    ]


def test_repeated_fresh_ledger_reference_is_not_double_billed(
    billing_fixture: dict[str, Any],
) -> None:
    _binding, usage = compute_result_binding(billing_fixture["candidate"])
    relay = usage["relay"]
    assert relay["ledgers"] == [
        {
            "relative_path": billing_fixture["binding"]["transport"][
                "ledger_relative_path"
            ],
            "sha256": billing_fixture["binding"]["transport"]["ledger_sha256"],
            "reference_count": 15,
        }
    ]
    assert usage["provider_billable_usage"]["total_tokens"] == 990_085


def test_rejects_tampered_referenced_ledger(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "tampered-ledger")
    ledger_path = (
        candidate / billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    ledger_path.write_bytes(ledger_path.read_bytes() + b"\n")

    with pytest.raises(BillingEvidenceError, match="unclaimed trailing bytes"):
        compute_result_binding(candidate)


def test_rejects_unreferenced_transport_ledger(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "orphan-ledger")
    origin_hash = billing_fixture["binding"]["transport"]["upstream_origin_sha256"]
    _write_jsonl(
        candidate / "_transport" / "relay-orphan.jsonl",
        [_session_header(origin_hash)],
    )

    with pytest.raises(BillingEvidenceError, match="unreferenced relay ledgers"):
        compute_result_binding(candidate)


def test_rejects_session_ledger_path_escape(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "escaped-ledger")
    record_path = next(
        (candidate / "_sessions" / "shopping_assistant" / "run1").glob("*.json")
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["relay_segment"]["relative_path"] = "../outside.jsonl"
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    _write_json(record_path, record)

    with pytest.raises(BillingEvidenceError, match="invalid auditable session chain"):
        compute_result_binding(candidate)


def test_rejects_resume_session_origin_drift(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "resume-origin-drift")
    _append_noop_session(
        candidate,
        domain="shopping_assistant",
        run=1,
        ledger_relative_path="_transport/relay-wrong-origin.jsonl",
        ledger_records=[_session_header("a" * 64)],
    )

    with pytest.raises(BillingEvidenceError, match="origin/transport contract mismatch"):
        compute_result_binding(candidate)


def test_rejects_same_physical_ledger_under_conflicting_paths(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "hardlink-ledger-alias")
    initial_path = (
        candidate / billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    alias_path = candidate / "_transport" / "relay-hardlink-alias.jsonl"
    alias_path.hardlink_to(initial_path)
    manifest = json.loads(
        (candidate / "shopping_assistant" / "run_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    task_ids = manifest["run"]["task_selection"]["task_ids"]
    log_path = (
        candidate
        / "_sessions"
        / "shopping_assistant"
        / "run1"
        / "resume-hardlink.log"
    )
    log_path.write_text("resume noop completed\n", encoding="utf-8")
    write_session_record(
        arm_root=candidate,
        domain="shopping_assistant",
        run_index=1,
        run_manifest_path=candidate / "shopping_assistant" / "run_manifest.json",
        mode="resume_noop",
        all_task_ids=task_ids,
        target_task_ids=[],
        pre_snapshot=snapshot_run(
            candidate / "shopping_assistant" / "run1", task_ids
        ),
        log_path=log_path,
        relay_ledger_path=alias_path,
        relay_start_offset=alias_path.stat().st_size,
        process_exit_code=0,
    )

    with pytest.raises(BillingEvidenceError, match="conflicting path/content"):
        compute_result_binding(candidate)


def test_rejects_additional_ledger_introduced_by_fresh_session(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "fresh-extra-ledger")
    initial_path = (
        candidate / billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    extra_relative = "_transport/relay-fresh-extra.jsonl"
    extra_path = candidate / extra_relative
    shutil.copyfile(initial_path, extra_path)
    record_path = next(
        (candidate / "_sessions" / "shopping_assistant" / "run1").glob("*.json")
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["relay_segment"]["relative_path"] = extra_relative
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    _write_json(record_path, record)

    with pytest.raises(BillingEvidenceError, match="Fresh-batch projection relay_segment mismatch"):
        compute_result_binding(candidate)


def test_formal_validator_billing_check_requires_and_accepts_evidence(
    billing_fixture: dict[str, Any],
) -> None:
    checks, report = _billing_checks(
        candidate_root=billing_fixture["candidate"],
        evidence_path=billing_fixture["evidence_path"],
        domains=DOMAINS,
        expected_runs=5,
        expected_tasks_per_domain=50,
    )
    assert report is not None
    assert len(checks) == 1 and checks[0].passed

    checks, report = _billing_checks(
        candidate_root=billing_fixture["candidate"],
        evidence_path=billing_fixture["root"] / "missing-evidence.json",
        domains=DOMAINS,
        expected_runs=5,
        expected_tasks_per_domain=50,
    )
    assert report is None
    assert len(checks) == 1 and not checks[0].passed


def test_rejects_openai_list_pricing_authority(billing_fixture: dict[str, Any]) -> None:
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"]["pricing_authority"] = "openai_list_price"
    path = _variant(billing_fixture, "openai-price.json", evidence)
    with pytest.raises(BillingEvidenceError, match="OpenAI list pricing"):
        reconcile_billing(billing_fixture["candidate"], path)


def test_rejects_non_novacode_or_tampered_source_document(
    billing_fixture: dict[str, Any],
) -> None:
    source_payload = deepcopy(billing_fixture["source_payload"])
    source_payload["provider"] = "OpenAI"
    source = billing_fixture["root"] / "wrong-issuer-source.json"
    _write_json(source, source_payload)
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"]["source_document"]["path"] = source.name
    evidence["evidence"]["source_document"]["sha256"] = file_sha256(source)
    path = _variant(billing_fixture, "wrong-issuer.json", evidence)
    with pytest.raises(BillingEvidenceError, match="not for NovaCode"):
        reconcile_billing(billing_fixture["candidate"], path)

    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"]["source_document"]["sha256"] = "f" * 64
    path = _variant(billing_fixture, "tampered-document.json", evidence)
    with pytest.raises(BillingEvidenceError, match="SHA-256 does not match"):
        reconcile_billing(billing_fixture["candidate"], path)

    arbitrary = billing_fixture["root"] / "arbitrary.pdf"
    arbitrary.write_bytes(b"not machine-readable provider evidence")
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"]["source_document"]["path"] = arbitrary.name
    evidence["evidence"]["source_document"]["sha256"] = file_sha256(arbitrary)
    path = _variant(billing_fixture, "arbitrary-file.json", evidence)
    with pytest.raises(BillingEvidenceError, match="requires a machine-readable NovaCode JSON"):
        reconcile_billing(billing_fixture["candidate"], path)


def test_rejects_zero_cost_placeholder(billing_fixture: dict[str, Any]) -> None:
    source_payload = deepcopy(billing_fixture["source_payload"])
    source_payload["rates_per_million_tokens"] = {
        "uncached_input": "0",
        "cached_input": "0",
        "output": "0",
    }
    source = billing_fixture["root"] / "zero-rate-source.json"
    _write_json(source, source_payload)
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"]["source_document"]["path"] = source.name
    evidence["evidence"]["source_document"]["sha256"] = file_sha256(source)
    path = _variant(billing_fixture, "zero-price.json", evidence)
    with pytest.raises(BillingEvidenceError, match="cost is zero"):
        reconcile_billing(billing_fixture["candidate"], path)


def test_rejects_result_or_artifact_binding_reuse(billing_fixture: dict[str, Any]) -> None:
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["binding"]["artifact_hashes"]["router_sha256"] = "a" * 64
    path = _variant(billing_fixture, "wrong-binding.json", evidence)
    with pytest.raises(BillingEvidenceError, match="binding does not match"):
        reconcile_billing(billing_fixture["candidate"], path)

    evidence = deepcopy(billing_fixture["evidence"])
    evidence["provider"]["base_url_origin_sha256"] = "a" * 64
    path = _variant(billing_fixture, "wrong-provider-origin.json", evidence)
    with pytest.raises(BillingEvidenceError, match="does not match the official run transport"):
        reconcile_billing(billing_fixture["candidate"], path)


def test_relay_usage_is_fail_closed_for_success_without_usage(
    billing_fixture: dict[str, Any],
    tmp_path: Path,
) -> None:
    transport = dict(billing_fixture["binding"]["transport"])
    transport.pop("ledger_sha256")
    source_ledger = billing_fixture["candidate"] / transport["ledger_relative_path"]
    records = [json.loads(line) for line in source_ledger.read_text(encoding="utf-8").splitlines()]
    records[1]["usage"] = None
    candidate = _candidate_variant(billing_fixture, tmp_path, "missing-usage-candidate")
    _write_jsonl(candidate / transport["ledger_relative_path"], records)
    _rebind_session_records_to_ledger(candidate, transport["ledger_relative_path"])
    manifests = [
        (
            domain,
            json.loads(
                (candidate / domain / "run_manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
        )
        for domain in DOMAINS
    ]
    with pytest.raises(BillingEvidenceError, match="no parseable usage"):
        collect_relay_usage(
            candidate,
            manifests,
            transport,
            final_agent_audits=billing_fixture["final_agent_audits"],
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("provider_request_audit_id", "g" * 32, "32 lowercase hexadecimal"),
        ("provider_task_key", "a" * 64, r"SHA256\(domain\|task_id\)"),
    ),
)
def test_rejects_invalid_final_trajectory_audit_binding(
    billing_fixture: dict[str, Any],
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, field)
    path = (
        candidate
        / "shopping_assistant"
        / "run1"
        / "shopping_assistant-task-00.json"
    )
    trajectory = json.loads(path.read_text(encoding="utf-8"))
    trajectory[field] = value
    _write_json(path, trajectory)
    with pytest.raises(BillingEvidenceError, match=message):
        compute_result_binding(candidate)


def test_rejects_duplicate_final_audit_id(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "duplicate-audit")
    first_path = (
        candidate
        / "shopping_assistant"
        / "run1"
        / "shopping_assistant-task-00.json"
    )
    second_path = (
        candidate
        / "shopping_assistant"
        / "run1"
        / "shopping_assistant-task-01.json"
    )
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    second["provider_request_audit_id"] = first["provider_request_audit_id"]
    _write_json(second_path, second)
    with pytest.raises(BillingEvidenceError, match="not unique"):
        compute_result_binding(candidate)


def test_rejects_invalid_agent_ledger_audit_or_task_key(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "bad-ledger-audit")
    path, records = _ledger_records(candidate, billing_fixture)
    agent = next(record for record in records if record.get("route") == "agent_chat_completions")
    agent["audit_id"] = "not-hex"
    _write_jsonl(path, records)
    _rebind_session_records_to_ledger(
        candidate, billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    with pytest.raises(BillingEvidenceError, match="32 lowercase hexadecimal"):
        compute_result_binding(candidate)

    candidate = _candidate_variant(billing_fixture, tmp_path, "bad-ledger-task")
    path, records = _ledger_records(candidate, billing_fixture)
    agent = next(record for record in records if record.get("route") == "agent_chat_completions")
    agent["task_key"] = "a" * 64
    _write_jsonl(path, records)
    _rebind_session_records_to_ledger(
        candidate, billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    with pytest.raises(BillingEvidenceError, match=r"SHA256\(domain\|task_id\)"):
        compute_result_binding(candidate)


def test_matches_agent_usage_per_audit_id_not_only_global_total(
    billing_fixture: dict[str, Any], tmp_path: Path
) -> None:
    candidate = _candidate_variant(billing_fixture, tmp_path, "per-audit-mismatch")
    path, records = _ledger_records(candidate, billing_fixture)
    final_agent_records = [
        record
        for record in records
        if record.get("route") == "agent_chat_completions"
        and record.get("audit_id") != "f" * 32
    ]
    first, second = final_agent_records[:2]
    first["usage"]["input_tokens"] += 1
    first["usage"]["total_tokens"] += 1
    second["usage"]["input_tokens"] -= 1
    second["usage"]["total_tokens"] -= 1
    _write_jsonl(path, records)
    _rebind_session_records_to_ledger(
        candidate, billing_fixture["binding"]["transport"]["ledger_relative_path"]
    )
    with pytest.raises(BillingEvidenceError, match="audit_id="):
        compute_result_binding(candidate)

def test_invoice_export_requires_exact_trajectory_token_match(
    billing_fixture: dict[str, Any],
) -> None:
    invoice_payload = {
        "schema_version": "novacode_machine_readable_evidence_v1",
        "provider": "NovaCode",
        "document_type": "provider_invoice_export",
        "model": "gpt-5.4",
        "currency": "USD",
        "issued_at": "2026-08-30",
        "invoice": {
            "invoice_id_sha256": hashlib.sha256(b"private-invoice-id").hexdigest(),
            "billing_period_start": "2026-08-30T00:00:00Z",
            "billing_period_end": "2026-08-30T23:59:59Z",
            "billing_scope": "official750_candidate_c_full_provider_traffic",
            "billable_usage": {
                field: billing_fixture["usage"]["provider_billable_usage"][field]
                for field in (
                    "input_tokens",
                    "cached_input_tokens",
                    "output_tokens",
                    "total_tokens",
                )
            },
            "amount": "1.5",
        },
    }
    source = billing_fixture["root"] / "novacode-invoice.json"
    _write_json(source, invoice_payload)
    evidence = deepcopy(billing_fixture["evidence"])
    evidence["evidence"] = {
        "mode": "invoice_export",
        "pricing_authority": "novacode_provider",
        "source_document": {
            "path": source.name,
            "sha256": file_sha256(source),
            "document_class": "provider_invoice_export",
        },
    }
    path = _variant(billing_fixture, "invoice.json", evidence)
    report = reconcile_billing(billing_fixture["candidate"], path)
    assert report["cost"]["reconciled_amount"] == "1.5"

    invoice_payload["invoice"]["billable_usage"]["input_tokens"] += 1
    source = billing_fixture["root"] / "novacode-invoice-mismatch.json"
    _write_json(source, invoice_payload)
    evidence["evidence"]["source_document"]["path"] = source.name
    evidence["evidence"]["source_document"]["sha256"] = file_sha256(source)
    path = _variant(billing_fixture, "invoice-mismatch.json", evidence)
    with pytest.raises(BillingEvidenceError, match="does not match full relay usage"):
        reconcile_billing(billing_fixture["candidate"], path)


def test_rate_evidence_conforms_to_published_json_schema(
    billing_fixture: dict[str, Any],
) -> None:
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "novacode_billing_evidence.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(billing_fixture["evidence"])
    source_schema_path = (
        Path(__file__).resolve().parents[1]
        / "docs"
        / "novacode_machine_readable_evidence.schema.json"
    )
    source_schema = json.loads(source_schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(source_schema).validate(
        billing_fixture["source_payload"]
    )
