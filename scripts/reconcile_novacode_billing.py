"""Reconcile NovaCode billing evidence against an official-750 result tree.

This command is deliberately offline.  It never calls NovaCode, never embeds a
price table, and never trusts STATE-Bench's provider-agnostic ``cost_usd``
field.  A user must supply either a NovaCode-issued rate-card document or a
NovaCode invoice/usage export, plus hashes binding that evidence to the exact
candidate trajectories, immutable run manifests, and launch artifacts.
Final agent usage is reconciled per opaque request audit ID; provider usage
from abandoned Resume attempts remains billable but is reported separately.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.resume_protocol import (
        ResumeProtocolError,
        plan_resume,
        verify_session_chain,
    )
except ModuleNotFoundError:  # direct ``python scripts/reconcile_novacode_billing.py``
    from resume_protocol import (  # type: ignore[no-redef]
        ResumeProtocolError,
        plan_resume,
        verify_session_chain,
    )


DOMAINS = ("shopping_assistant", "travel", "customer_support")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
AUDIT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
RUN_DIR_RE = re.compile(r"run[_-]?(\d+)$", flags=re.IGNORECASE)
EXPECTED_PROVIDER = "NovaCode"
EXPECTED_MODEL = "gpt-5.4"
EXPECTED_SCOPE = "official750_candidate_c_full_provider_traffic"
BILLING_SCHEMA_VERSION = "1.2.0"
USAGE_FIELDS = (
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "reasoning_output_tokens",
    "total_tokens",
)


class BillingEvidenceError(ValueError):
    """Raised when billing inputs cannot be safely reconciled."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BillingEvidenceError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise BillingEvidenceError(f"{label} must be a JSON object: {path}")
    return value


def _require_keys(
    value: Mapping[str, Any], required: set[str], allowed: set[str], *, label: str
) -> None:
    missing = required - set(value)
    extra = set(value) - allowed
    if missing:
        raise BillingEvidenceError(f"{label} is missing fields: {sorted(missing)}")
    if extra:
        raise BillingEvidenceError(f"{label} has unsupported fields: {sorted(extra)}")


def _integer(value: Any, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BillingEvidenceError(f"{label} must be a non-negative integer")
    return value


def _task_key(domain: str, task_id: str) -> str:
    """Return the opaque task binding without reading task/conversation content."""

    return hashlib.sha256(f"{domain}|{task_id}".encode("utf-8")).hexdigest()


def _audit_id(value: Any, *, label: str) -> str:
    result = str(value or "")
    if AUDIT_ID_RE.fullmatch(result) is None:
        raise BillingEvidenceError(f"{label} must be exactly 32 lowercase hexadecimal characters")
    return result


def _provider_task_key(value: Any, *, label: str) -> str:
    result = str(value or "")
    if SHA256_RE.fullmatch(result) is None:
        raise BillingEvidenceError(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return result


def _decimal(value: Any, *, label: str) -> Decimal:
    # Numeric JSON values are accepted for ergonomics, but conversion through
    # str avoids importing their binary floating-point representation.
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise BillingEvidenceError(f"{label} must be a non-negative decimal string or number")
    if isinstance(value, str) and re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", value) is None:
        raise BillingEvidenceError(f"{label} is not a plain non-negative decimal")
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise BillingEvidenceError(f"{label} is not a decimal") from exc
    if not result.is_finite() or result < 0:
        raise BillingEvidenceError(f"{label} must be finite and non-negative")
    return result


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _resolve_document(evidence_path: Path, declared_path: str) -> Path:
    value = Path(declared_path)
    if not value.is_absolute():
        value = evidence_path.resolve().parent / value
    return value.resolve()


def _artifact_binding(manifests: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, str]:
    fields = (
        "memory_sha256",
        "router_sha256",
        "runner_sha256",
        "state_bench_protocol_sha256",
        "state_bench_commit",
    )
    result: dict[str, str] = {}
    for field in fields:
        values = {
            str(manifest.get("artifacts", {}).get(field, ""))
            for _, manifest in manifests
            if isinstance(manifest.get("artifacts"), Mapping)
        }
        if len(values) != 1 or not next(iter(values), ""):
            raise BillingEvidenceError(
                f"run manifests do not agree on artifacts.{field}: {sorted(values)}"
            )
        value = next(iter(values))
        pattern = GIT_SHA_RE if field == "state_bench_commit" else SHA256_RE
        if pattern.fullmatch(value) is None:
            raise BillingEvidenceError(f"run manifest artifacts.{field} is not a valid hash")
        result[field] = value
    return result


def _transport_binding(manifests: Sequence[tuple[str, Mapping[str, Any]]]) -> dict[str, str]:
    fields = (
        "provider",
        "upstream_origin_sha256",
        "relay_sha256",
        "ledger_relative_path",
    )
    result: dict[str, str] = {}
    for field in fields:
        values = {
            str(manifest.get("transport", {}).get(field, ""))
            for _, manifest in manifests
            if isinstance(manifest.get("transport"), Mapping)
        }
        if len(values) != 1 or not next(iter(values), ""):
            raise BillingEvidenceError(
                f"run manifests do not agree on transport.{field}: {sorted(values)}"
            )
        result[field] = next(iter(values))
    if result["provider"] != "novacode":
        raise BillingEvidenceError("run manifests were not recorded with the NovaCode transport")
    for field in ("upstream_origin_sha256", "relay_sha256"):
        if SHA256_RE.fullmatch(result[field]) is None:
            raise BillingEvidenceError(f"run manifest transport.{field} is not a valid SHA-256")
    ledger_path = Path(result["ledger_relative_path"])
    if ledger_path.is_absolute() or ".." in ledger_path.parts:
        raise BillingEvidenceError("run manifest ledger_relative_path must stay below candidate root")
    return result


def _load_run_manifests(
    candidate_root: Path, domains: Sequence[str]
) -> tuple[list[tuple[str, Mapping[str, Any]]], list[dict[str, str]]]:
    manifests: list[tuple[str, Mapping[str, Any]]] = []
    entries: list[dict[str, str]] = []
    for domain in domains:
        path = candidate_root / domain / "run_manifest.json"
        manifest = _load_object(path, label=f"{domain} run manifest")
        core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
        declared_self_hash = str(manifest.get("manifest_sha256", ""))
        if declared_self_hash != canonical_sha256(core):
            raise BillingEvidenceError(f"run manifest has an invalid self-hash: {path}")
        if manifest.get("stage") != "official750" or manifest.get("arm") != "candidate":
            raise BillingEvidenceError(f"run manifest is not official750 candidate: {path}")
        if manifest.get("router_stage") != "C" or manifest.get("domain") != domain:
            raise BillingEvidenceError(f"run manifest is not candidate-C/{domain}: {path}")
        relative = path.relative_to(candidate_root).as_posix()
        manifests.append((domain, manifest))
        entries.append(
            {
                "path": relative,
                "file_sha256": file_sha256(path),
                "manifest_sha256": declared_self_hash,
            }
        )
    return manifests, entries


def _trajectory_files(
    candidate_root: Path,
    domains: Sequence[str],
    *,
    expected_runs: int,
    expected_tasks_per_domain: int,
) -> list[tuple[str, int, Path]]:
    selected: list[tuple[str, int, Path]] = []
    for domain in domains:
        domain_root = candidate_root / domain
        if not domain_root.is_dir():
            raise BillingEvidenceError(f"missing official domain result directory: {domain_root}")
        seen_runs: set[int] = set()
        for run_dir in sorted(path for path in domain_root.iterdir() if path.is_dir()):
            match = RUN_DIR_RE.fullmatch(run_dir.name)
            if match is None:
                continue
            run = int(match.group(1))
            if run < 1 or run > expected_runs:
                raise BillingEvidenceError(f"unexpected run directory: {run_dir}")
            if run in seen_runs:
                raise BillingEvidenceError(f"duplicate run number {run} below {domain_root}")
            seen_runs.add(run)
            files = sorted(run_dir.glob("*.json"))
            if len(files) != expected_tasks_per_domain:
                raise BillingEvidenceError(
                    f"{run_dir} has {len(files)} JSON trajectories; "
                    f"expected {expected_tasks_per_domain}"
                )
            selected.extend((domain, run, path) for path in files)
        expected = set(range(1, expected_runs + 1))
        if seen_runs != expected:
            raise BillingEvidenceError(
                f"{domain_root} has run numbers {sorted(seen_runs)}; expected {sorted(expected)}"
            )
    return selected


def _collect_official_usage_with_audits(
    candidate_root: Path,
    *,
    domains: Sequence[str] = DOMAINS,
    expected_runs: int = 5,
    expected_tasks_per_domain: int = 50,
) -> tuple[
    dict[str, Any],
    list[dict[str, str]],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
]:
    """Return strict token totals, hashes, and per-final-attempt audit bindings.

    The code intentionally reads only identifiers and token counters from each
    JSON object.  Conversation text, requirements, and judge reasoning are not
    used by billing reconciliation.
    """

    candidate_root = candidate_root.resolve()
    files = _trajectory_files(
        candidate_root,
        domains,
        expected_runs=expected_runs,
        expected_tasks_per_domain=expected_tasks_per_domain,
    )
    fields = USAGE_FIELDS
    overall = {field: 0 for field in fields}
    by_domain = {
        domain: {"observations": 0, **{field: 0 for field in fields}} for domain in domains
    }
    entries: list[dict[str, str]] = []
    audit_entries: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}
    seen: set[tuple[str, int, str]] = set()
    for domain, run, path in files:
        raw = _load_object(path, label="trajectory")
        declared_domain = raw.get("domain")
        if declared_domain is not None and declared_domain != domain:
            raise BillingEvidenceError(f"trajectory domain does not match its directory: {path}")
        task_id = str(raw.get("task_id", "")).strip()
        if not task_id:
            raise BillingEvidenceError(f"trajectory has no task_id: {path}")
        key = (domain, run, task_id)
        if key in seen:
            raise BillingEvidenceError(f"duplicate trajectory key {key}")
        seen.add(key)
        audit_id = _audit_id(
            raw.get("provider_request_audit_id"),
            label=f"{path}: provider_request_audit_id",
        )
        if audit_id in audits:
            raise BillingEvidenceError(
                f"provider_request_audit_id is not unique across final trajectories: {audit_id}"
            )
        task_key = _provider_task_key(
            raw.get("provider_task_key"), label=f"{path}: provider_task_key"
        )
        expected_task_key = _task_key(domain, task_id)
        if task_key != expected_task_key:
            raise BillingEvidenceError(
                f"trajectory provider_task_key does not equal SHA256(domain|task_id): {path}"
            )
        usage = raw.get("token_usage")
        if not isinstance(usage, Mapping):
            raise BillingEvidenceError(f"trajectory has no token_usage object: {path}")
        values: dict[str, int] = {}
        for field in fields:
            if field == "reasoning_output_tokens" and field not in usage:
                values[field] = 0
            else:
                values[field] = _integer(usage.get(field), label=f"{path}: token_usage.{field}")
        if values["cached_input_tokens"] > values["input_tokens"]:
            raise BillingEvidenceError(f"cached input exceeds input tokens: {path}")
        if values["total_tokens"] != values["input_tokens"] + values["output_tokens"]:
            raise BillingEvidenceError(f"token_usage.total_tokens is inconsistent: {path}")
        if values["total_tokens"] == 0:
            raise BillingEvidenceError(f"trajectory reports zero total tokens: {path}")
        for field, value in values.items():
            overall[field] += value
            by_domain[domain][field] += value
        by_domain[domain]["observations"] += 1
        entries.append(
            {
                "path": path.relative_to(candidate_root).as_posix(),
                "sha256": file_sha256(path),
            }
        )
        audit_binding = {
            "audit_id": audit_id,
            "task_key": task_key,
            "domain": domain,
            "task_id": task_id,
            "usage": values,
        }
        audits[audit_id] = audit_binding
        audit_entries.append(
            {
                "path": path.relative_to(candidate_root).as_posix(),
                "audit_id": audit_id,
                "task_key": task_key,
                "usage": values,
            }
        )
    expected_count = len(domains) * expected_runs * expected_tasks_per_domain
    if len(entries) != expected_count:
        raise BillingEvidenceError(
            f"found {len(entries)} trajectories; expected official total {expected_count}"
        )
    entries.sort(key=lambda item: item["path"])
    audit_entries.sort(key=lambda item: item["path"])
    return {
        "observations": len(entries),
        **overall,
        "uncached_input_tokens": overall["input_tokens"] - overall["cached_input_tokens"],
        "by_domain": by_domain,
    }, entries, audits, audit_entries


def collect_official_usage(
    candidate_root: Path,
    *,
    domains: Sequence[str] = DOMAINS,
    expected_runs: int = 5,
    expected_tasks_per_domain: int = 50,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return final-trajectory usage and file hashes after strict audit validation."""

    usage, entries, _audits, _audit_entries = _collect_official_usage_with_audits(
        candidate_root,
        domains=domains,
        expected_runs=expected_runs,
        expected_tasks_per_domain=expected_tasks_per_domain,
    )
    return usage, entries


def _empty_usage() -> dict[str, int]:
    return {field: 0 for field in USAGE_FIELDS}


def _parse_usage(value: Mapping[str, Any], *, label: str) -> dict[str, int]:
    parsed = {
        field: _integer(value.get(field, 0), label=f"{label}.{field}")
        for field in USAGE_FIELDS
    }
    if parsed["cached_input_tokens"] > parsed["input_tokens"]:
        raise BillingEvidenceError(f"{label}: cached input exceeds input tokens")
    if parsed["total_tokens"] != parsed["input_tokens"] + parsed["output_tokens"]:
        raise BillingEvidenceError(f"{label}: total_tokens is inconsistent")
    return parsed


def _add_usage(target: dict[str, int], value: Mapping[str, Any], *, label: str) -> None:
    parsed = _parse_usage(value, label=label)
    for field in USAGE_FIELDS:
        target[field] += parsed[field]


def _resolve_relay_ledger(
    candidate_root: Path, declared_path: Any, *, label: str
) -> tuple[Path, str]:
    raw = str(declared_path or "").strip()
    relative = Path(raw)
    if not raw or relative.is_absolute() or ".." in relative.parts:
        raise BillingEvidenceError(f"{label} must stay below the candidate root")
    resolved_root = candidate_root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        canonical_relative = resolved.relative_to(resolved_root).as_posix()
    except ValueError as exc:
        raise BillingEvidenceError(f"{label} resolves outside the candidate root") from exc
    if relative.as_posix() != canonical_relative:
        raise BillingEvidenceError(f"{label} is non-canonical or aliases another path")
    if resolved.suffix.lower() != ".jsonl":
        raise BillingEvidenceError(f"{label} must reference a JSONL relay ledger")
    if not resolved.is_file():
        raise BillingEvidenceError(f"{label} does not exist: {resolved}")
    return resolved, canonical_relative


def _ledger_identity(path: Path) -> str:
    stat_result = path.stat()
    if stat_result.st_ino:
        return f"{stat_result.st_dev}:{stat_result.st_ino}"
    return str(path.resolve()).casefold()


def _collect_verified_ledger_inventory(
    candidate_root: Path,
    manifests: Sequence[tuple[str, Mapping[str, Any]]],
    transport: Mapping[str, str],
    *,
    expected_runs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    """Verify every official session chain and return unique referenced ledgers."""

    candidate_root = candidate_root.resolve()
    expected_session = {
        "event": "session_start",
        "provider": "novacode",
        "upstream_origin_sha256": transport["upstream_origin_sha256"],
        "rpm": 45,
        "burst": 5,
        "burst_window_seconds": 1.0,
        "attempts": 5,
    }
    ledgers_by_identity: dict[str, dict[str, Any]] = {}
    segment_claims: dict[str, dict[tuple[int, int], tuple[str, str]]] = {}
    fresh_ledger_identities: set[str] = set()
    session_entries: list[dict[str, Any]] = []
    session_ids: set[str] = set()
    verified_session_paths: set[str] = set()
    verified_batch_record_paths: set[str] = set()

    for domain, manifest in manifests:
        manifest_path = candidate_root / domain / "run_manifest.json"
        run_contract = manifest.get("run")
        task_selection = (
            run_contract.get("task_selection")
            if isinstance(run_contract, Mapping)
            else None
        )
        task_ids = (
            task_selection.get("task_ids")
            if isinstance(task_selection, Mapping)
            else None
        )
        if not isinstance(task_ids, list) or not all(
            isinstance(task_id, str) for task_id in task_ids
        ):
            raise BillingEvidenceError(
                f"run manifest task IDs are invalid for {domain}"
            )
        domain_batch_members: dict[int, Mapping[str, Any]] = {}
        for run_index in range(1, expected_runs + 1):
            session_paths = sorted(
                (
                    candidate_root / "_sessions" / domain / f"run{run_index}"
                ).glob("*.json")
            )
            try:
                chain = verify_session_chain(
                    arm_root=candidate_root,
                    domain=domain,
                    run_index=run_index,
                    run_manifest_path=manifest_path,
                )
                resume_plan = plan_resume(
                    arm_root=candidate_root,
                    domain=domain,
                    run_index=run_index,
                    run_manifest_path=manifest_path,
                    task_ids=task_ids,
                )
            except (OSError, ResumeProtocolError) as exc:
                raise BillingEvidenceError(
                    f"invalid auditable session chain for {domain}/run{run_index}: {exc}"
                ) from exc
            if not chain.records:
                raise BillingEvidenceError(
                    f"auditable session chain is missing for {domain}/run{run_index}"
                )
            if len(session_paths) != len(chain.records):
                raise BillingEvidenceError(
                    f"session record inventory changed while verifying {domain}/run{run_index}"
                )
            first_batch = chain.records[0].get("fresh_batch")
            if isinstance(first_batch, Mapping):
                domain_batch_members[run_index] = first_batch
                batch_relative = Path(str(first_batch.get("relative_path", "")))
                if batch_relative.is_absolute() or ".." in batch_relative.parts:
                    raise BillingEvidenceError("fresh-batch record path is non-canonical")
                batch_path = (candidate_root / batch_relative).resolve()
                try:
                    batch_path.relative_to(candidate_root)
                except ValueError as exc:
                    raise BillingEvidenceError(
                        "fresh-batch record resolves outside candidate root"
                    ) from exc
                verified_batch_record_paths.add(str(batch_path).casefold())
            if (
                resume_plan["agent_task_ids"]
                or resume_plan["score_task_ids"]
                or resume_plan["rejected_task_ids"]
                or resume_plan["scored_task_ids"] != task_ids
            ):
                raise BillingEvidenceError(
                    f"session chain does not bind every final scored trajectory for "
                    f"{domain}/run{run_index}"
                )
            for record_path, record in zip(session_paths, chain.records, strict=True):
                relay = record.get("relay_segment")
                session = record.get("relay_session")
                if not isinstance(relay, Mapping) or not isinstance(session, Mapping):
                    raise BillingEvidenceError(
                        f"session relay evidence is absent for {domain}/run{run_index}"
                    )
                if dict(session) != expected_session:
                    raise BillingEvidenceError(
                        f"session origin/transport contract mismatch for {domain}/run{run_index}"
                    )
                ledger_path, relative = _resolve_relay_ledger(
                    candidate_root,
                    relay.get("relative_path"),
                    label=(
                        f"{domain}/run{run_index} session relay_segment.relative_path"
                    ),
                )
                identity = _ledger_identity(ledger_path)
                ledger_hash = file_sha256(ledger_path)
                existing = ledgers_by_identity.get(identity)
                if existing is None:
                    ledgers_by_identity[identity] = {
                        "path": ledger_path,
                        "relative_path": relative,
                        "sha256": ledger_hash,
                        "reference_count": 1,
                        "max_claimed_end": relay.get("end_offset"),
                    }
                else:
                    if (
                        existing["relative_path"] != relative
                        or existing["sha256"] != ledger_hash
                    ):
                        raise BillingEvidenceError(
                            "the same relay ledger has conflicting path/content declarations"
                        )
                    existing["reference_count"] += 1
                    existing["max_claimed_end"] = max(
                        int(existing["max_claimed_end"]),
                        int(relay.get("end_offset", -1)),
                    )

                start = relay.get("start_offset")
                end = relay.get("end_offset")
                claim = (
                    str(relay.get("prefix_sha256", "")),
                    str(relay.get("sha256", "")),
                )
                if not isinstance(start, int) or not isinstance(end, int):
                    raise BillingEvidenceError("session relay segment offsets are invalid")
                claim_key = (start, end)
                prior_claim = segment_claims.setdefault(identity, {}).setdefault(
                    claim_key, claim
                )
                if prior_claim != claim:
                    raise BillingEvidenceError(
                        "the same relay ledger segment has conflicting content declarations"
                    )
                if record.get("mode") == "fresh":
                    fresh_ledger_identities.add(identity)

                session_id = str(record.get("session_id", ""))
                if session_id in session_ids:
                    raise BillingEvidenceError(
                        f"session_id is reused across official runs: {session_id}"
                    )
                session_ids.add(session_id)
                record_relative = record_path.resolve().relative_to(
                    candidate_root
                ).as_posix()
                verified_session_paths.add(str(record_path.resolve()).casefold())
                session_entries.append(
                    {
                        "path": record_relative,
                        "domain": domain,
                        "run_index": run_index,
                        "sequence": record.get("sequence"),
                        "session_id": session_id,
                        "record_sha256": record.get("record_sha256"),
                        "mode": record.get("mode"),
                        "relay_relative_path": relative,
                        "relay_segment_sha256": relay.get("sha256"),
                    }
                )
        if expected_runs == 5 and manifest.get("stage") == "official750":
            batch_identities = {
                (
                    str(member.get("relative_path", "")),
                    str(member.get("sha256", "")),
                    str(member.get("batch_id", "")),
                )
                for member in domain_batch_members.values()
            }
            if (
                set(domain_batch_members) != {1, 2, 3, 4, 5}
                or len(batch_identities) != 1
                or {
                    member.get("member_run_index")
                    for member in domain_batch_members.values()
                }
                != {1, 2, 3, 4, 5}
            ):
                raise BillingEvidenceError(
                    f"official750 {domain} is not bound to one verified five-run fresh batch"
                )

    sessions_root = candidate_root / "_sessions"
    if sessions_root.is_dir():
        orphaned_sessions = [
            path.resolve().relative_to(candidate_root).as_posix()
            for path in sorted(sessions_root.rglob("*.json"))
            if path.is_file()
            and str(path.resolve()).casefold() not in verified_session_paths
        ]
        if orphaned_sessions:
            raise BillingEvidenceError(
                f"unverified session records exist below _sessions: {orphaned_sessions}"
            )
    batch_records_root = candidate_root / "_batch_records"
    if batch_records_root.is_dir():
        orphaned_batch_records = [
            path.resolve().relative_to(candidate_root).as_posix()
            for path in sorted(batch_records_root.rglob("*.json"))
            if path.is_file()
            and str(path.resolve()).casefold() not in verified_batch_record_paths
        ]
        if orphaned_batch_records:
            raise BillingEvidenceError(
                "unreferenced fresh-batch records exist below _batch_records: "
                f"{orphaned_batch_records}"
            )

    initial_path, initial_relative = _resolve_relay_ledger(
        candidate_root,
        transport["ledger_relative_path"],
        label="run manifest transport.ledger_relative_path",
    )
    initial_identity = _ledger_identity(initial_path)
    if initial_identity not in ledgers_by_identity:
        raise BillingEvidenceError(
            "the initial manifest relay ledger is not referenced by any verified session"
        )
    if fresh_ledger_identities != {initial_identity}:
        raise BillingEvidenceError(
            "every verified fresh session must reference the initial manifest relay ledger"
        )
    if ledgers_by_identity[initial_identity]["relative_path"] != initial_relative:
        raise BillingEvidenceError("initial relay ledger has a conflicting path declaration")

    for source in ledgers_by_identity.values():
        if source["max_claimed_end"] != source["path"].stat().st_size:
            raise BillingEvidenceError(
                f"relay ledger {source['relative_path']} has unclaimed trailing bytes or "
                "a conflicting final-size declaration"
            )

    transport_root = candidate_root / "_transport"
    orphaned: list[str] = []
    if transport_root.is_dir():
        for path in sorted(transport_root.rglob("relay-*.jsonl")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(candidate_root).as_posix()
            except ValueError as exc:
                raise BillingEvidenceError(
                    f"relay ledger below _transport resolves outside candidate root: {path}"
                ) from exc
            if _ledger_identity(resolved) not in ledgers_by_identity:
                orphaned.append(relative)
    if orphaned:
        raise BillingEvidenceError(
            f"unreferenced relay ledgers exist below _transport: {orphaned}"
        )

    sources = sorted(
        ledgers_by_identity.values(), key=lambda item: item["relative_path"]
    )
    session_entries.sort(
        key=lambda item: (item["domain"], item["run_index"], item["sequence"])
    )
    return sources, session_entries, initial_relative


def collect_relay_usage(
    candidate_root: Path,
    manifests: Sequence[tuple[str, Mapping[str, Any]]],
    transport: Mapping[str, str],
    *,
    final_agent_audits: Mapping[str, Mapping[str, Any]],
    expected_runs: int = 5,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Aggregate each verified provider ledger once and match final audit IDs."""

    sources, session_entries, initial_relative = _collect_verified_ledger_inventory(
        candidate_root,
        manifests,
        transport,
        expected_runs=expected_runs,
    )

    all_responses = _empty_usage()
    final_agent = _empty_usage()
    abandoned_agent = _empty_usage()
    final_audit_nontrajectory = _empty_usage()
    by_route: dict[str, dict[str, int]] = {}
    agent_usage_by_audit: dict[str, dict[str, int]] = {}
    final_success_by_audit: dict[str, dict[str, int]] = {}
    agent_task_keys: dict[str, str] = {}
    known_task_keys = {
        str(binding.get("task_key", "")) for binding in final_agent_audits.values()
    }
    response_records = 0
    usage_records = 0
    missing_usage_records = 0
    retry_response_records = 0
    abandoned_response_records_with_usage = 0
    abandoned_success_response_records_with_usage = 0
    abandoned_retry_or_failed_response_records_with_usage = 0
    total_records = 0
    for source in sources:
        ledger_path = source["path"]
        relative = source["relative_path"]
        try:
            ledger_bytes = ledger_path.read_bytes()
            if hashlib.sha256(ledger_bytes).hexdigest() != source["sha256"]:
                raise BillingEvidenceError(
                    f"relay ledger {relative} changed after session-chain verification"
                )
            lines = ledger_bytes.decode("utf-8").splitlines()
        except (OSError, UnicodeError) as exc:
            raise BillingEvidenceError(
                f"cannot read relay ledger {ledger_path}: {exc}"
            ) from exc
        if not lines:
            raise BillingEvidenceError(f"relay ledger is empty: {relative}")
        total_records += len(lines)
        seen_attempts: set[tuple[int, int]] = set()
        session_starts: list[Mapping[str, Any]] = []
        for line_number, line in enumerate(lines, 1):
            label = f"ledger {relative} line {line_number}"
            if not line.strip():
                raise BillingEvidenceError(f"{label} is blank")
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BillingEvidenceError(f"{label} is invalid JSON: {exc}") from exc
            if not isinstance(record, Mapping) or record.get("schema_version") != "1.0.0":
                raise BillingEvidenceError(f"{label} has an invalid schema")
            event = record.get("event")
            if event == "session_start":
                session_starts.append(record)
                continue
            if event not in {"upstream_response", "transport_error"}:
                raise BillingEvidenceError(f"{label} has unknown event {event!r}")
            route = str(record.get("route", ""))
            if route not in {"agent_chat_completions", "official_eval_responses", "other"}:
                raise BillingEvidenceError(f"{label} has unknown relay route {route!r}")
            audit_id: str | None = None
            task_key: str | None = None
            if route == "agent_chat_completions":
                audit_id = _audit_id(record.get("audit_id"), label=f"{label}.audit_id")
                task_key = _provider_task_key(
                    record.get("task_key"), label=f"{label}.task_key"
                )
                if task_key not in known_task_keys:
                    raise BillingEvidenceError(
                        f"{label}.task_key does not equal SHA256(domain|task_id) "
                        "for any final official task"
                    )
                previous_task_key = agent_task_keys.setdefault(audit_id, task_key)
                if previous_task_key != task_key:
                    raise BillingEvidenceError(
                        f"agent audit ID {audit_id} is bound to multiple task keys"
                    )
                if audit_id in final_agent_audits and task_key != final_agent_audits[
                    audit_id
                ].get("task_key"):
                    raise BillingEvidenceError(
                        f"final agent audit ID {audit_id} has the wrong provider task key"
                    )
            request_id = _integer(record.get("request_id"), label=f"{label}.request_id")
            attempt = _integer(record.get("attempt"), label=f"{label}.attempt")
            if request_id < 1 or attempt < 1:
                raise BillingEvidenceError("relay request_id and attempt must start at one")
            key = (request_id, attempt)
            if key in seen_attempts:
                raise BillingEvidenceError(
                    f"duplicate relay request attempt {key} in {relative}"
                )
            seen_attempts.add(key)
            if event != "upstream_response":
                if record.get("usage") is not None:
                    raise BillingEvidenceError(
                        "transport_error ledger events cannot carry token usage"
                    )
                continue
            response_records += 1
            retryable = record.get("retryable_status")
            if not isinstance(retryable, bool):
                raise BillingEvidenceError(f"{label}.retryable_status is not boolean")
            status = record.get("status_code")
            if (
                isinstance(status, bool)
                or not isinstance(status, int)
                or not 100 <= status <= 599
            ):
                raise BillingEvidenceError(f"{label}.status_code is invalid")
            usage = record.get("usage")
            if usage is None:
                missing_usage_records += 1
                if route in {"agent_chat_completions", "official_eval_responses"} and 200 <= status < 300:
                    raise BillingEvidenceError(
                        f"successful provider response has no parseable usage at {label}"
                    )
                continue
            if not isinstance(usage, Mapping):
                raise BillingEvidenceError(f"relay usage is not an object at {label}")
            usage_records += 1
            route_total = by_route.setdefault(route, _empty_usage())
            _add_usage(route_total, usage, label=f"{label}.usage")
            _add_usage(all_responses, usage, label=f"{label}.usage")
            if retryable:
                retry_response_records += 1
            if route == "agent_chat_completions":
                assert audit_id is not None
                audit_total = agent_usage_by_audit.setdefault(audit_id, _empty_usage())
                _add_usage(audit_total, usage, label=f"{label}.usage")
                successful_final_response = 200 <= status < 300 and not retryable
                if audit_id in final_agent_audits and successful_final_response:
                    matched = final_success_by_audit.setdefault(audit_id, _empty_usage())
                    _add_usage(matched, usage, label=f"{label}.usage")
                elif audit_id in final_agent_audits:
                    _add_usage(
                        final_audit_nontrajectory,
                        usage,
                        label=f"{label}.usage",
                    )
                else:
                    _add_usage(abandoned_agent, usage, label=f"{label}.usage")
                    abandoned_response_records_with_usage += 1
                    if successful_final_response:
                        abandoned_success_response_records_with_usage += 1
                    else:
                        abandoned_retry_or_failed_response_records_with_usage += 1

        if len(session_starts) != 1:
            raise BillingEvidenceError(
                f"relay ledger {relative} must contain exactly one session_start; "
                f"found {len(session_starts)}"
            )
        expected_header_contract = {
            "event": "session_start",
            "provider": "novacode",
            "upstream_origin_sha256": transport["upstream_origin_sha256"],
            "rpm": 45,
            "burst": 5,
            "burst_window_seconds": 1.0,
            "attempts": 5,
        }
        actual_header_contract = {
            key: session_starts[0].get(key) for key in expected_header_contract
        }
        if actual_header_contract != expected_header_contract:
            raise BillingEvidenceError(
                f"relay ledger {relative} session origin/transport contract mismatch"
            )

    if by_route.get("official_eval_responses", {}).get("total_tokens", 0) <= 0:
        raise BillingEvidenceError("relay ledgers have no recorded official simulator/judge token usage")
    for audit_id, trajectory in final_agent_audits.items():
        ledger_usage = final_success_by_audit.get(audit_id, _empty_usage())
        expected_usage = trajectory.get("usage")
        if not isinstance(expected_usage, Mapping) or ledger_usage != dict(expected_usage):
            raise BillingEvidenceError(
                "final trajectory usage does not match successful agent ledger responses for "
                f"audit_id={audit_id}: trajectory={expected_usage}, ledger={ledger_usage}"
            )
        _add_usage(final_agent, ledger_usage, label=f"final audit {audit_id}.usage")
    if final_agent["total_tokens"] <= 0:
        raise BillingEvidenceError("relay ledger has no audit-matched final agent token usage")
    if all_responses["total_tokens"] <= 0:
        raise BillingEvidenceError("relay ledger reports zero provider token usage")

    if len(manifests) != len(DOMAINS):
        raise BillingEvidenceError("relay usage requires all three domain run manifests")
    final_audit_ids = set(final_agent_audits)
    abandoned_audit_ids = set(agent_task_keys) - final_audit_ids
    attribution_entries = [
        {
            "audit_id": audit_id,
            "task_key": agent_task_keys[audit_id],
            "classification": "final" if audit_id in final_audit_ids else "abandoned",
            "provider_billable_usage": agent_usage_by_audit.get(audit_id, _empty_usage()),
            "final_matched_usage": final_success_by_audit.get(audit_id, _empty_usage())
            if audit_id in final_audit_ids
            else None,
        }
        for audit_id in sorted(agent_task_keys)
    ]
    public_ledgers = [
        {
            "relative_path": source["relative_path"],
            "sha256": source["sha256"],
            "reference_count": source["reference_count"],
        }
        for source in sources
    ]
    relay_ledger_manifest_sha256 = canonical_sha256(public_ledgers)
    session_chain_manifest_sha256 = canonical_sha256(session_entries)
    initial_ledger = next(
        item for item in public_ledgers if item["relative_path"] == initial_relative
    )
    ledger_binding = {
        "ledger_relative_path": initial_relative,
        "ledger_sha256": initial_ledger["sha256"],
        "relay_ledger_manifest_sha256": relay_ledger_manifest_sha256,
        "ledgers": public_ledgers,
        "session_chain_manifest_sha256": session_chain_manifest_sha256,
    }
    return {
        "ledger_relative_path": initial_relative,
        "ledger_sha256": initial_ledger["sha256"],
        "ledger_count": len(public_ledgers),
        "ledgers": public_ledgers,
        "relay_ledger_manifest_sha256": relay_ledger_manifest_sha256,
        "session_record_count": len(session_entries),
        "session_chain_manifest_sha256": session_chain_manifest_sha256,
        "records": total_records,
        "response_records": response_records,
        "usage_records": usage_records,
        "missing_usage_records": missing_usage_records,
        "retry_response_records_with_usage": retry_response_records,
        "final_agent_audit_count": len(final_audit_ids),
        "matched_final_agent_audit_count": len(final_success_by_audit),
        "final_agent_usage": final_agent,
        "final_audit_nontrajectory_usage": final_audit_nontrajectory,
        "abandoned_agent_attempts": {
            "audit_id_count": len(abandoned_audit_ids),
            "response_records_with_usage": abandoned_response_records_with_usage,
            "successful_response_records_with_usage": abandoned_success_response_records_with_usage,
            "retry_or_failed_response_records_with_usage": (
                abandoned_retry_or_failed_response_records_with_usage
            ),
            "provider_billable_usage": abandoned_agent,
        },
        "agent_attribution_sha256": canonical_sha256(attribution_entries),
        "provider_billable_usage": all_responses,
        "provider_usage_by_route": dict(sorted(by_route.items())),
    }, ledger_binding


def compute_result_binding(
    candidate_root: Path,
    *,
    domains: Sequence[str] = DOMAINS,
    expected_runs: int = 5,
    expected_tasks_per_domain: int = 50,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute the hashes an evidence author must copy into ``binding``."""

    candidate_root = candidate_root.resolve()
    trajectory_usage, trajectory_entries, final_agent_audits, audit_entries = (
        _collect_official_usage_with_audits(
            candidate_root,
            domains=domains,
            expected_runs=expected_runs,
            expected_tasks_per_domain=expected_tasks_per_domain,
        )
    )
    manifests, manifest_entries = _load_run_manifests(candidate_root, domains)
    transport = _transport_binding(manifests)
    relay_usage, ledger_binding = collect_relay_usage(
        candidate_root,
        manifests,
        transport,
        final_agent_audits=final_agent_audits,
        expected_runs=expected_runs,
    )
    transport_binding = {
        **transport,
        "ledger_sha256": ledger_binding["ledger_sha256"],
    }
    binding = {
        "trajectory_manifest_sha256": canonical_sha256(trajectory_entries),
        "final_agent_audit_manifest_sha256": canonical_sha256(audit_entries),
        "relay_agent_attribution_sha256": relay_usage["agent_attribution_sha256"],
        "relay_ledger_manifest_sha256": ledger_binding[
            "relay_ledger_manifest_sha256"
        ],
        "session_chain_manifest_sha256": ledger_binding[
            "session_chain_manifest_sha256"
        ],
        "run_manifest_bundle_sha256": canonical_sha256(manifest_entries),
        "artifact_hashes": _artifact_binding(manifests),
        "transport": transport_binding,
    }
    usage = {
        **trajectory_usage,
        "final_agent_usage": relay_usage["final_agent_usage"],
        "abandoned_agent_attempts": relay_usage["abandoned_agent_attempts"],
        "relay": relay_usage,
        "provider_billable_usage": relay_usage["provider_billable_usage"],
    }
    return binding, usage


def _check_source_document(
    evidence_path: Path, source: Mapping[str, Any], *, mode: str
) -> tuple[dict[str, str], Mapping[str, Any]]:
    _require_keys(
        source,
        {"path", "sha256", "document_class"},
        {"path", "sha256", "document_class"},
        label="evidence.source_document",
    )
    expected_class = "provider_rate_card" if mode == "rate_card" else "provider_invoice_export"
    if source.get("document_class") != expected_class:
        raise BillingEvidenceError(
            f"source document_class must be {expected_class!r} for mode {mode!r}"
        )
    declared_path = str(source.get("path", "")).strip()
    if not declared_path:
        raise BillingEvidenceError("source_document.path must not be empty")
    path = _resolve_document(evidence_path, declared_path)
    if not path.is_file():
        raise BillingEvidenceError(f"billing source document does not exist: {path}")
    if path.suffix.lower() != ".json":
        raise BillingEvidenceError(
            "automatic billing validation requires a machine-readable NovaCode JSON export; "
            "PDFs and screenshots require manual review"
        )
    declared_hash = str(source.get("sha256", ""))
    if SHA256_RE.fullmatch(declared_hash) is None or declared_hash != file_sha256(path):
        raise BillingEvidenceError("billing source document SHA-256 does not match")
    payload = _load_object(path, label="machine-readable NovaCode source document")
    common = {
        "schema_version",
        "provider",
        "document_type",
        "model",
        "currency",
        "issued_at",
    }
    specific = "rates_per_million_tokens" if mode == "rate_card" else "invoice"
    _require_keys(
        payload,
        common | {specific},
        common | {specific},
        label="machine-readable NovaCode source document",
    )
    if payload.get("schema_version") != "novacode_machine_readable_evidence_v1":
        raise BillingEvidenceError("unsupported machine-readable NovaCode evidence schema")
    if payload.get("provider") != EXPECTED_PROVIDER or payload.get("model") != EXPECTED_MODEL:
        raise BillingEvidenceError("machine-readable source is not for NovaCode gpt-5.4")
    if payload.get("document_type") != expected_class:
        raise BillingEvidenceError("machine-readable source document_type does not match mode")
    currency = str(payload.get("currency", ""))
    if re.fullmatch(r"[A-Z]{3}", currency) is None:
        raise BillingEvidenceError("source currency must be a three-letter ISO currency code")
    issued_at = str(payload.get("issued_at", "")).strip()
    if not issued_at:
        raise BillingEvidenceError("source issued_at must not be empty")
    metadata = {
        "path": str(path),
        "sha256": declared_hash,
        "document_class": expected_class,
        "issuer": EXPECTED_PROVIDER,
        "issued_at": issued_at,
        "machine_readable_schema": "novacode_machine_readable_evidence_v1",
    }
    return metadata, payload


def _validate_binding(declared: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    _require_keys(
        declared,
        {
            "trajectory_manifest_sha256",
            "final_agent_audit_manifest_sha256",
            "relay_agent_attribution_sha256",
            "relay_ledger_manifest_sha256",
            "session_chain_manifest_sha256",
            "run_manifest_bundle_sha256",
            "artifact_hashes",
            "transport",
        },
        {
            "trajectory_manifest_sha256",
            "final_agent_audit_manifest_sha256",
            "relay_agent_attribution_sha256",
            "relay_ledger_manifest_sha256",
            "session_chain_manifest_sha256",
            "run_manifest_bundle_sha256",
            "artifact_hashes",
            "transport",
        },
        label="binding",
    )
    if declared != actual:
        raise BillingEvidenceError(
            "billing evidence binding does not match this result tree; "
            f"expected binding={json.dumps(actual, sort_keys=True)}"
        )


def _rate_card_cost(source_payload: Mapping[str, Any], usage: Mapping[str, Any]) -> dict[str, Any]:
    rates = source_payload.get("rates_per_million_tokens")
    if not isinstance(rates, Mapping):
        raise BillingEvidenceError("rates_per_million_tokens must be an object")
    _require_keys(
        rates,
        {"uncached_input", "cached_input", "output"},
        {"uncached_input", "cached_input", "output"},
        label="rates_per_million_tokens",
    )
    parsed = {
        key: _decimal(rates[key], label=f"rates_per_million_tokens.{key}")
        for key in ("uncached_input", "cached_input", "output")
    }
    billable = usage.get("provider_billable_usage")
    if not isinstance(billable, Mapping):
        raise BillingEvidenceError("provider billable usage is unavailable")
    uncached_input = int(billable["input_tokens"]) - int(billable["cached_input_tokens"])
    million = Decimal(1_000_000)
    components = {
        "uncached_input": Decimal(uncached_input) * parsed["uncached_input"] / million,
        "cached_input": Decimal(int(billable["cached_input_tokens"]))
        * parsed["cached_input"]
        / million,
        "output": Decimal(int(billable["output_tokens"])) * parsed["output"] / million,
    }
    amount = sum(components.values(), Decimal(0))
    if amount <= 0:
        raise BillingEvidenceError(
            "NovaCode-reconciled cost is zero; cost=0 and zero-valued placeholder pricing are invalid"
        )
    return {
        "method": "novacode_rate_card_times_full_relay_usage",
        "billing_scope": EXPECTED_SCOPE,
        "rates_per_million_tokens": {key: _decimal_text(value) for key, value in parsed.items()},
        "components": {key: _decimal_text(value) for key, value in components.items()},
        "reconciled_amount": _decimal_text(amount),
    }


def _invoice_cost(source_payload: Mapping[str, Any], usage: Mapping[str, Any]) -> dict[str, Any]:
    invoice = source_payload.get("invoice")
    if not isinstance(invoice, Mapping):
        raise BillingEvidenceError("evidence.invoice must be an object")
    _require_keys(
        invoice,
        {
            "invoice_id_sha256",
            "billing_period_start",
            "billing_period_end",
            "billing_scope",
            "billable_usage",
            "amount",
        },
        {
            "invoice_id_sha256",
            "billing_period_start",
            "billing_period_end",
            "billing_scope",
            "billable_usage",
            "amount",
        },
        label="evidence.invoice",
    )
    if SHA256_RE.fullmatch(str(invoice.get("invoice_id_sha256", ""))) is None:
        raise BillingEvidenceError("invoice_id_sha256 must be a privacy-preserving SHA-256")
    if invoice.get("billing_scope") != EXPECTED_SCOPE:
        raise BillingEvidenceError(f"invoice.billing_scope must equal {EXPECTED_SCOPE!r}")
    billable = invoice.get("billable_usage")
    if not isinstance(billable, Mapping):
        raise BillingEvidenceError("invoice.billable_usage must be an object")
    fields = ("input_tokens", "cached_input_tokens", "output_tokens", "total_tokens")
    _require_keys(billable, set(fields), set(fields), label="invoice.billable_usage")
    parsed_usage = {
        field: _integer(billable[field], label=f"invoice.billable_usage.{field}")
        for field in fields
    }
    provider_billable = usage.get("provider_billable_usage")
    if not isinstance(provider_billable, Mapping):
        raise BillingEvidenceError("provider billable usage is unavailable")
    expected_usage = {field: int(provider_billable[field]) for field in fields}
    if parsed_usage != expected_usage:
        raise BillingEvidenceError(
            f"invoice billable usage does not match full relay usage: expected {expected_usage}"
        )
    amount = _decimal(invoice.get("amount"), label="invoice.amount")
    if amount <= 0:
        raise BillingEvidenceError("NovaCode invoice amount must be greater than zero")
    for field in ("billing_period_start", "billing_period_end"):
        if not str(invoice.get(field, "")).strip():
            raise BillingEvidenceError(f"invoice.{field} must not be empty")
    return {
        "method": "novacode_invoice_export_full_relay_token_match",
        "invoice_id_sha256": str(invoice["invoice_id_sha256"]),
        "billing_period_start": str(invoice["billing_period_start"]),
        "billing_period_end": str(invoice["billing_period_end"]),
        "billing_scope": EXPECTED_SCOPE,
        "billable_usage": parsed_usage,
        "reconciled_amount": _decimal_text(amount),
    }


def reconcile_billing(
    candidate_root: Path,
    evidence_path: Path,
    *,
    domains: Sequence[str] = DOMAINS,
    expected_runs: int = 5,
    expected_tasks_per_domain: int = 50,
) -> dict[str, Any]:
    """Validate evidence and return a deterministic reconciliation report."""

    candidate_root = candidate_root.resolve()
    evidence_path = evidence_path.resolve()
    evidence_root = _load_object(evidence_path, label="billing evidence")
    _require_keys(
        evidence_root,
        {"schema_version", "provider", "binding", "evidence"},
        {"schema_version", "provider", "binding", "evidence"},
        label="billing evidence",
    )
    if evidence_root.get("schema_version") != BILLING_SCHEMA_VERSION:
        raise BillingEvidenceError(
            f"billing evidence schema_version must equal {BILLING_SCHEMA_VERSION!r}"
        )
    provider = evidence_root.get("provider")
    if not isinstance(provider, Mapping):
        raise BillingEvidenceError("provider must be an object")
    _require_keys(
        provider,
        {"name", "model", "base_url_origin_sha256"},
        {"name", "model", "base_url_origin_sha256"},
        label="provider",
    )
    if provider.get("name") != EXPECTED_PROVIDER:
        raise BillingEvidenceError("provider.name must identify NovaCode")
    if provider.get("model") != EXPECTED_MODEL:
        raise BillingEvidenceError(f"provider.model must equal {EXPECTED_MODEL!r}")
    origin_hash = str(provider.get("base_url_origin_sha256", ""))
    if SHA256_RE.fullmatch(origin_hash) is None or set(origin_hash) == {"0"}:
        raise BillingEvidenceError("provider.base_url_origin_sha256 must be a non-placeholder SHA-256")

    actual_binding, usage = compute_result_binding(
        candidate_root,
        domains=domains,
        expected_runs=expected_runs,
        expected_tasks_per_domain=expected_tasks_per_domain,
    )
    declared_binding = evidence_root.get("binding")
    if not isinstance(declared_binding, Mapping):
        raise BillingEvidenceError("binding must be an object")
    _validate_binding(declared_binding, actual_binding)
    if origin_hash != actual_binding["transport"]["upstream_origin_sha256"]:
        raise BillingEvidenceError(
            "provider.base_url_origin_sha256 does not match the official run transport"
        )

    evidence = evidence_root.get("evidence")
    if not isinstance(evidence, Mapping):
        raise BillingEvidenceError("evidence must be an object")
    mode = str(evidence.get("mode", ""))
    if mode not in {"rate_card", "invoice_export"}:
        raise BillingEvidenceError("evidence.mode must be 'rate_card' or 'invoice_export'")
    if evidence.get("pricing_authority") != "novacode_provider":
        raise BillingEvidenceError(
            "pricing_authority must be 'novacode_provider'; OpenAI list pricing is not accepted"
        )
    _require_keys(
        evidence,
        {"mode", "pricing_authority", "source_document"},
        {"mode", "pricing_authority", "source_document"},
        label="evidence",
    )
    source = evidence.get("source_document")
    if not isinstance(source, Mapping):
        raise BillingEvidenceError("evidence.source_document must be an object")
    verified_source, source_payload = _check_source_document(evidence_path, source, mode=mode)
    currency = str(source_payload["currency"])
    cost = (
        _rate_card_cost(source_payload, usage)
        if mode == "rate_card"
        else _invoice_cost(source_payload, usage)
    )
    amount = Decimal(cost["reconciled_amount"])

    report_core = {
        "schema_version": BILLING_SCHEMA_VERSION,
        "reconciliation": "official750_novacode_user_supplied_billing",
        "passed": True,
        "provider": {
            "name": EXPECTED_PROVIDER,
            "model": EXPECTED_MODEL,
            "base_url_origin_sha256": origin_hash,
        },
        "candidate_root": str(candidate_root),
        "evidence_path": str(evidence_path),
        "evidence_sha256": file_sha256(evidence_path),
        "source_document": verified_source,
        "binding": actual_binding,
        "token_usage": usage,
        "cost": {
            "currency": currency,
            **cost,
            "mean_per_observation": _decimal_text(amount / Decimal(int(usage["observations"]))),
        },
        "accounting_notes": {
            "trajectory_cost_fields_trusted": False,
            "openai_list_pricing_used": False,
            "runtime_model_calls_added": 0,
            "provider_bill_scope": (
                "all unique verified-session relay-ledger responses with reported usage, "
                "including retries and abandoned attempts"
            ),
            "final_agent_attribution": "per unique 32-hex audit ID from final trajectories",
            "abandoned_agent_usage_in_provider_bill": True,
            "auditable_session_chains_verified": True,
            "includes_agent_simulator_and_judge": True,
            "provider_signature_verified": False,
            "source_authenticity": "user-supplied machine-readable export; hash-bound for audit",
        },
    }
    return {**report_core, "report_sha256": canonical_sha256(report_core)}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--evidence", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = reconcile_billing(args.candidate, args.evidence)
    except (OSError, UnicodeError, json.JSONDecodeError, BillingEvidenceError) as exc:
        print(f"billing reconciliation input error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        cost = report["cost"]
        usage = report["token_usage"]
        print("PASS NovaCode official750 billing reconciliation")
        print(
            f"observations={usage['observations']} "
            f"agent_tokens={usage['total_tokens']} "
            "abandoned_agent_tokens="
            f"{usage['abandoned_agent_attempts']['provider_billable_usage']['total_tokens']} "
            f"provider_billable_tokens={usage['provider_billable_usage']['total_tokens']} "
            f"cost={cost['reconciled_amount']} {cost['currency']} "
            f"mean={cost['mean_per_observation']} {cost['currency']}/task"
        )
        print(f"report_sha256={report['report_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
