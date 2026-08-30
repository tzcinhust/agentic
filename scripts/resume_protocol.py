"""Auditable, fail-closed resume helpers for selective PWM evaluation.

The PowerShell runner remains the protocol entrypoint.  This module owns the
small pieces that are easier to make deterministic and testable in Python:

* classify trajectories as missing, raw/unscored, or fully scored;
* verify the append-only, self-hashed per-run session chain;
* permit a resume only when the latest session proves an exhausted
  transport-class failure for the same task and phase;
* stage score-only retries away from the canonical output tree and promote a
  result atomically only after a complete official score is present.

Session records intentionally contain task IDs and hashes only.  They never
contain task definitions, requirements, user messages, prompts, responses, or
exception text.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "1.1.0"
OFFICIAL_PROTOCOL_ID = "state_bench_v0.8.1_gpt54"
SESSION_DIRECTORY = "_sessions"
BATCH_RECORD_DIRECTORY = "_batch_records"
OFFICIAL_FRESH_BATCH_RUN_INDICES = (1, 2, 3, 4, 5)
OFFICIAL_FRESH_BATCH_COMMAND = {
    "module": "state_bench.scripts.run_batch",
    "split": "test",
    "num_runs": 5,
    "num_runs_idx_start": 1,
}
RETRYABLE_STATUS_CODES = {429, 502, 503, 504}
TRANSPORT_ERROR_PATTERN = re.compile(
    r"(?:"
    r"shim upstream failed|"
    r"\b(?:429|502|503|504)\b|"
    r"(?:APIConnection|Connect|Read|Write|Pool|RemoteProtocol|Transport|Timeout)Error|"
    r"(?:connection|connect|read|write|pool)[ _-]?timeout"
    r")",
    re.IGNORECASE,
)


class ResumeProtocolError(RuntimeError):
    """The immutable evidence is absent, inconsistent, or non-retryable."""


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResumeProtocolError(f"Invalid JSON evidence: {path}") from exc


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError as exc:
        raise ResumeProtocolError(f"Refusing to overwrite immutable evidence: {path}") from exc
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    _make_read_only(path)


def _make_read_only(path: Path) -> None:
    try:
        path.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        # The self-hash remains authoritative on filesystems that cannot apply
        # POSIX-like modes (for example some Windows network shares).
        pass


def _assert_relative_to(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ResumeProtocolError(f"{label} must stay inside the arm output root") from exc
    return resolved


def _manifest_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def load_verified_manifest(path: Path) -> dict[str, Any]:
    raw = _read_json(path)
    if not isinstance(raw, dict):
        raise ResumeProtocolError("Run manifest must be a JSON object")
    expected = canonical_sha256(_manifest_core(raw))
    if raw.get("manifest_sha256") != expected:
        raise ResumeProtocolError("Run manifest self-hash mismatch")
    return raw


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_binary_score(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value in (0, 1)


def is_fully_scored(raw: Mapping[str, Any], *, task_id: str | None = None) -> bool:
    """Return true only for a complete official task/state + UX score.

    A completion value of zero is deliberately accepted: a scored failure is
    final evidence and must never be sampled again.
    """

    if task_id is not None and raw.get("task_id") != task_id:
        return False
    if not _is_binary_score(raw.get("task_completion_pass")):
        return False
    if not _is_binary_score(raw.get("state_requirements_met")):
        return False
    if not _is_binary_score(raw.get("task_requirements_met")):
        return False
    ux_score = raw.get("ux_score")
    if not _is_number(ux_score) or not 1.0 <= float(ux_score) <= 5.0:
        return False
    if raw.get("scoring_protocol_id") != OFFICIAL_PROTOCOL_ID:
        return False
    if raw.get("judge_model") != "gpt-5.4":
        return False
    if raw.get("judge_reasoning_effort") != "high":
        return False
    if raw.get("evaluation_protocol_id") != OFFICIAL_PROTOCOL_ID:
        return False
    if raw.get("simulator_model") != "gpt-5.4":
        return False
    if not _is_sha256(raw.get("simulator_prompt_hash")):
        return False
    agent_model = raw.get("agent_model")
    if not isinstance(agent_model, dict) or agent_model.get("model_name") != "gpt-5.4":
        return False
    prompt_hashes = raw.get("judge_prompt_hashes")
    return (
        isinstance(prompt_hashes, dict)
        and bool(prompt_hashes)
        and all(isinstance(name, str) and _is_sha256(value) for name, value in prompt_hashes.items())
    )


def trajectory_state(path: Path, *, task_id: str) -> dict[str, Any]:
    if not path.exists():
        return {"state": "missing", "sha256": None}
    if not path.is_file():
        raise ResumeProtocolError(f"Trajectory path is not a file: {path}")
    digest = file_sha256(path)
    raw = _read_json(path)
    if not isinstance(raw, dict) or raw.get("task_id") != task_id:
        raise ResumeProtocolError(f"Trajectory identity mismatch: {path}")
    state = "scored" if is_fully_scored(raw, task_id=task_id) else "unscored"
    return {"state": state, "sha256": digest}


def snapshot_run(run_dir: Path, task_ids: Sequence[str]) -> dict[str, Any]:
    unique = _normalize_task_ids(task_ids)
    return {
        "schema_version": SCHEMA_VERSION,
        "task_ids_sha256": task_ids_sha256(unique),
        "tasks": {
            task_id: trajectory_state(run_dir / f"{task_id}.json", task_id=task_id)
            for task_id in unique
        },
    }


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    return hashlib.sha256((("\n".join(task_ids)) + "\n").encode("utf-8")).hexdigest()


def _normalize_task_ids(task_ids: Iterable[str]) -> list[str]:
    result = [str(task_id) for task_id in task_ids]
    if not result or any(
        not item
        or "\x00" in item
        or Path(item).name != item
        or item in {".", ".."}
        for item in result
    ):
        raise ResumeProtocolError("Task IDs must be non-empty strings")
    if len(result) != len(set(result)):
        raise ResumeProtocolError("Task IDs must be unique")
    return result


def _load_task_ids(path: Path) -> list[str]:
    raw = _read_json(path)
    if not isinstance(raw, list) or not all(isinstance(item, str) for item in raw):
        raise ResumeProtocolError("Task-ID input must be a JSON string array")
    return _normalize_task_ids(raw)


def _record_core(record: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "record_sha256"}


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _validate_task_state_map(value: Any) -> None:
    if not isinstance(value, dict):
        raise ResumeProtocolError("Session task-state evidence must be an object")
    for task_id, state in value.items():
        if not isinstance(task_id, str) or not isinstance(state, dict) or set(state) != {"state", "sha256"}:
            raise ResumeProtocolError("Session task-state evidence has an invalid shape")
        if state["state"] not in {"missing", "unscored", "scored"}:
            raise ResumeProtocolError("Session task-state label is invalid")
        if state["state"] == "missing":
            if state["sha256"] is not None:
                raise ResumeProtocolError("Missing trajectory must not have a hash")
        elif not _is_sha256(state["sha256"]):
            raise ResumeProtocolError("Existing trajectory hash is invalid")


def _validate_record_shape(record: Mapping[str, Any]) -> None:
    expected_fields = {
        "schema_version",
        "record_type",
        "session_id",
        "sequence",
        "previous_session_sha256",
        "run_manifest_relative_path",
        "run_manifest_sha256",
        "domain",
        "run_index",
        "mode",
        "command_kind",
        "target_task_ids",
        "target_task_ids_sha256",
        "pre_state",
        "post_state",
        "process_exit_code",
        "log",
        "relay_segment",
        "relay_session",
        "transport_proof",
        "fresh_batch",
        "record_sha256",
    }
    if set(record) != expected_fields:
        raise ResumeProtocolError("Session record has unknown or missing fields")
    if record.get("schema_version") != SCHEMA_VERSION or record.get("record_type") != "auditable_resume_session":
        raise ResumeProtocolError("Session record type/version mismatch")
    if not isinstance(record.get("session_id"), str) or not re.fullmatch(r"[0-9a-f]{32}", record["session_id"]):
        raise ResumeProtocolError("Session ID is invalid")
    if not isinstance(record.get("sequence"), int) or record["sequence"] < 1:
        raise ResumeProtocolError("Session sequence is invalid")
    previous = record.get("previous_session_sha256")
    if previous is not None and not _is_sha256(previous):
        raise ResumeProtocolError("Previous-session hash is invalid")
    if not _is_sha256(record.get("run_manifest_sha256")) or not _is_sha256(record.get("record_sha256")):
        raise ResumeProtocolError("Session binding hash is invalid")
    if record.get("mode") not in {"fresh", "resume_agent", "resume_score", "resume_rejected", "resume_noop"}:
        raise ResumeProtocolError("Session mode is invalid")
    expected_command = (
        "score"
        if record["mode"] == "resume_score"
        else ("none" if record["mode"] in {"resume_noop", "resume_rejected"} else "run_batch")
    )
    if record.get("command_kind") != expected_command:
        raise ResumeProtocolError("Session command kind is invalid")
    if not isinstance(record.get("process_exit_code"), int):
        raise ResumeProtocolError("Session exit code is invalid")
    _validate_task_state_map(record.get("pre_state"))
    _validate_task_state_map(record.get("post_state"))
    if set(record["pre_state"]) != set(record["post_state"]):
        raise ResumeProtocolError("Session pre/post task sets differ")

    log = record.get("log")
    if not isinstance(log, dict) or set(log) != {"relative_path", "sha256", "size_bytes"}:
        raise ResumeProtocolError("Session log evidence shape is invalid")
    if not isinstance(log["relative_path"], str) or not _is_sha256(log["sha256"]) or not isinstance(log["size_bytes"], int):
        raise ResumeProtocolError("Session log evidence value is invalid")

    relay = record.get("relay_segment")
    relay_fields = {
        "relative_path",
        "start_offset",
        "end_offset",
        "prefix_sha256",
        "sha256",
        "exhausted_request_counts_by_route",
    }
    if not isinstance(relay, dict) or set(relay) != relay_fields:
        raise ResumeProtocolError("Relay segment evidence shape is invalid")
    if (
        not isinstance(relay["relative_path"], str)
        or not isinstance(relay["start_offset"], int)
        or not isinstance(relay["end_offset"], int)
        or not _is_sha256(relay["prefix_sha256"])
        or not _is_sha256(relay["sha256"])
    ):
        raise ResumeProtocolError("Relay segment evidence value is invalid")
    counts = relay["exhausted_request_counts_by_route"]
    if not isinstance(counts, dict) or any(
        route not in {"agent_chat_completions", "official_eval_responses", "other"}
        or not isinstance(count, int)
        or count < 0
        for route, count in counts.items()
    ):
        raise ResumeProtocolError("Relay exhausted-request counters are invalid")

    session = record.get("relay_session")
    if not isinstance(session, dict) or set(session) != {
        "event",
        "provider",
        "upstream_origin_sha256",
        "rpm",
        "burst",
        "burst_window_seconds",
        "attempts",
    }:
        raise ResumeProtocolError("Relay session evidence shape is invalid")
    if _relay_session_contract(session) != session:
        raise ResumeProtocolError("Relay session evidence value is invalid")

    proof = record.get("transport_proof")
    if not isinstance(proof, dict) or set(proof) != {
        "classification",
        "agent_or_run_task_ids",
        "scoring_task_ids",
    }:
        raise ResumeProtocolError("Transport proof shape is invalid")
    if proof["classification"] not in {"none", "transport_failure"}:
        raise ResumeProtocolError("Transport proof classification is invalid")
    targets = record.get("target_task_ids")
    if not isinstance(targets, list) or not all(isinstance(item, str) for item in targets):
        raise ResumeProtocolError("Session target task IDs are invalid")
    target_set = set(targets)
    if record.get("mode") == "resume_agent" and len(targets) != 1:
        raise ResumeProtocolError("Agent Resume must target exactly one trajectory")
    for key in ("agent_or_run_task_ids", "scoring_task_ids"):
        values = proof[key]
        if not isinstance(values, list) or len(values) != len(set(values)) or any(item not in target_set for item in values):
            raise ResumeProtocolError("Transport proof task IDs are invalid")
    has_proof = bool(proof["agent_or_run_task_ids"] or proof["scoring_task_ids"])
    if (proof["classification"] == "transport_failure") != has_proof:
        raise ResumeProtocolError("Transport proof classification disagrees with its task IDs")

    fresh_batch = record.get("fresh_batch")
    if fresh_batch is None:
        return
    if record.get("mode") != "fresh" or not isinstance(fresh_batch, dict) or set(fresh_batch) != {
        "relative_path",
        "sha256",
        "batch_id",
        "member_run_index",
    }:
        raise ResumeProtocolError("Fresh-batch reference shape is invalid")
    if (
        not isinstance(fresh_batch["relative_path"], str)
        or not _is_sha256(fresh_batch["sha256"])
        or not isinstance(fresh_batch["batch_id"], str)
        or re.fullmatch(r"[0-9a-f]{32}", fresh_batch["batch_id"]) is None
        or fresh_batch["member_run_index"] != record.get("run_index")
    ):
        raise ResumeProtocolError("Fresh-batch reference value is invalid")


def _session_dir(arm_root: Path, domain: str, run_index: int) -> Path:
    if not re.fullmatch(r"[a-z][a-z0-9_]*", domain):
        raise ResumeProtocolError("Invalid domain identifier")
    if run_index < 1:
        raise ResumeProtocolError("Run index must be positive")
    return arm_root / SESSION_DIRECTORY / domain / f"run{run_index}"


def _fresh_batch_record_path(arm_root: Path, domain: str, batch_id: str) -> Path:
    if re.fullmatch(r"[0-9a-f]{32}", batch_id) is None:
        raise ResumeProtocolError("Fresh-batch ID is invalid")
    return arm_root / BATCH_RECORD_DIRECTORY / domain / f"{batch_id}.json"


def _validate_fresh_batch_record(
    *,
    arm_root: Path,
    domain: str,
    manifest_path: Path,
    manifest_hash: str,
    manifest_task_ids: Sequence[str],
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], Path]:
    """Validate the immutable evidence for one official five-run invocation."""

    expected_fields = {
        "schema_version",
        "record_type",
        "batch_id",
        "domain",
        "run_indices",
        "command_kind",
        "command_contract",
        "run_manifest_relative_path",
        "run_manifest_sha256",
        "target_task_ids",
        "target_task_ids_sha256",
        "process_exit_code",
        "log",
        "relay_segment",
        "relay_session",
        "batch_record_sha256",
    }
    batch_id = str(reference.get("batch_id", ""))
    expected_path = _fresh_batch_record_path(arm_root, domain, batch_id).resolve()
    record_path = _assert_relative_to(
        arm_root / str(reference.get("relative_path", "")),
        arm_root,
        "Fresh-batch record",
    )
    if record_path != expected_path or not record_path.is_file():
        raise ResumeProtocolError("Fresh-batch record path is absent or non-canonical")
    raw = _read_json(record_path)
    if not isinstance(raw, dict) or set(raw) != expected_fields:
        raise ResumeProtocolError("Fresh-batch record has unknown or missing fields")
    if (
        raw.get("schema_version") != SCHEMA_VERSION
        or raw.get("record_type") != "auditable_fresh_batch"
        or raw.get("batch_id") != batch_id
        or raw.get("domain") != domain
        or raw.get("run_indices") != list(OFFICIAL_FRESH_BATCH_RUN_INDICES)
        or raw.get("command_kind") != "run_batch"
        or raw.get("command_contract") != OFFICIAL_FRESH_BATCH_COMMAND
        or raw.get("run_manifest_relative_path")
        != manifest_path.relative_to(arm_root).as_posix()
        or raw.get("run_manifest_sha256") != manifest_hash
        or raw.get("target_task_ids") != list(manifest_task_ids)
        or raw.get("target_task_ids_sha256") != task_ids_sha256(manifest_task_ids)
        or not isinstance(raw.get("process_exit_code"), int)
    ):
        raise ResumeProtocolError("Fresh-batch record contract mismatch")
    expected_hash = canonical_sha256(
        {key: value for key, value in raw.items() if key != "batch_record_sha256"}
    )
    if raw.get("batch_record_sha256") != expected_hash:
        raise ResumeProtocolError("Fresh-batch record self-hash mismatch")
    if reference.get("sha256") != expected_hash:
        raise ResumeProtocolError("Fresh-batch reference hash mismatch")

    log = raw.get("log")
    relay = raw.get("relay_segment")
    session = raw.get("relay_session")
    if not isinstance(log, dict) or set(log) != {"relative_path", "sha256", "size_bytes"}:
        raise ResumeProtocolError("Fresh-batch log evidence shape is invalid")
    if not isinstance(relay, dict) or set(relay) != {
        "relative_path",
        "start_offset",
        "end_offset",
        "prefix_sha256",
        "sha256",
        "exhausted_request_counts_by_route",
    }:
        raise ResumeProtocolError("Fresh-batch relay evidence shape is invalid")
    if not isinstance(session, dict) or _relay_session_contract(session) != session:
        raise ResumeProtocolError("Fresh-batch relay session contract is invalid")
    return raw, record_path


def _verify_fresh_batch_members(
    *,
    arm_root: Path,
    domain: str,
    manifest_path: Path,
    manifest_hash: str,
    manifest_task_ids: Sequence[str],
    current_record: Mapping[str, Any],
) -> None:
    reference = current_record.get("fresh_batch")
    if reference is None:
        return
    if not isinstance(reference, Mapping):
        raise ResumeProtocolError("Fresh-batch reference is invalid")
    batch, _ = _validate_fresh_batch_record(
        arm_root=arm_root,
        domain=domain,
        manifest_path=manifest_path,
        manifest_hash=manifest_hash,
        manifest_task_ids=manifest_task_ids,
        reference=reference,
    )
    if reference.get("member_run_index") != current_record.get("run_index"):
        raise ResumeProtocolError("Fresh-batch member/run binding mismatch")
    for field in ("process_exit_code", "log", "relay_segment", "relay_session"):
        if current_record.get(field) != batch.get(field):
            raise ResumeProtocolError(f"Fresh-batch projection {field} mismatch")

    # All five projections must exist.  This prevents a partially written batch
    # from authorizing Resume for only the members that happened to be recorded
    # before a crash.
    expected_reference = {
        "relative_path": str(reference["relative_path"]),
        "sha256": str(reference["sha256"]),
        "batch_id": str(reference["batch_id"]),
    }
    for member_run_index in OFFICIAL_FRESH_BATCH_RUN_INDICES:
        first_records = sorted(
            _session_dir(arm_root, domain, member_run_index).glob("000001-*.json")
        )
        if len(first_records) != 1:
            raise ResumeProtocolError("Fresh-batch projection set is incomplete or ambiguous")
        sibling = _read_json(first_records[0])
        if not isinstance(sibling, dict):
            raise ResumeProtocolError("Fresh-batch projection is not an object")
        _validate_record_shape(sibling)
        if sibling.get("record_sha256") != canonical_sha256(_record_core(sibling)):
            raise ResumeProtocolError("Fresh-batch projection self-hash mismatch")
        sibling_reference = sibling.get("fresh_batch")
        if (
            sibling.get("mode") != "fresh"
            or sibling.get("sequence") != 1
            or sibling.get("previous_session_sha256") is not None
            or sibling.get("domain") != domain
            or sibling.get("run_index") != member_run_index
            or not isinstance(sibling_reference, Mapping)
            or {
                key: sibling_reference.get(key)
                for key in ("relative_path", "sha256", "batch_id")
            }
            != expected_reference
            or sibling_reference.get("member_run_index") != member_run_index
        ):
            raise ResumeProtocolError("Fresh-batch sibling projection binding mismatch")
        for field in (
            "run_manifest_relative_path",
            "run_manifest_sha256",
            "target_task_ids",
            "target_task_ids_sha256",
            "process_exit_code",
            "log",
            "relay_segment",
            "relay_session",
        ):
            expected = (
                batch.get(field)
                if field in {"process_exit_code", "log", "relay_segment", "relay_session"}
                else current_record.get(field)
            )
            if sibling.get(field) != expected:
                raise ResumeProtocolError(f"Fresh-batch sibling {field} mismatch")


@dataclass(frozen=True)
class VerifiedSessionChain:
    records: tuple[dict[str, Any], ...]

    @property
    def latest_by_task(self) -> dict[str, dict[str, Any]]:
        latest: dict[str, dict[str, Any]] = {}
        for record in self.records:
            for task_id in record["target_task_ids"]:
                latest[task_id] = record
        return latest


def verify_session_chain(
    *,
    arm_root: Path,
    domain: str,
    run_index: int,
    run_manifest_path: Path,
) -> VerifiedSessionChain:
    arm_root = arm_root.resolve()
    run_manifest_path = _assert_relative_to(run_manifest_path, arm_root, "Run manifest")
    manifest = load_verified_manifest(run_manifest_path)
    manifest_hash = str(manifest["manifest_sha256"])
    manifest_task_ids = manifest.get("run", {}).get("task_selection", {}).get("task_ids")
    if not isinstance(manifest_task_ids, list) or not all(isinstance(item, str) for item in manifest_task_ids):
        raise ResumeProtocolError("Run manifest task IDs are invalid")
    manifest_task_set = set(manifest_task_ids)
    directory = _session_dir(arm_root, domain, run_index)
    if not directory.exists():
        return VerifiedSessionChain(())

    paths = sorted(directory.glob("*.json"))
    records: list[dict[str, Any]] = []
    previous_hash: str | None = None
    for expected_sequence, path in enumerate(paths, start=1):
        raw = _read_json(path)
        if not isinstance(raw, dict):
            raise ResumeProtocolError(f"Session record is not an object: {path}")
        _validate_record_shape(raw)
        record_hash = canonical_sha256(_record_core(raw))
        if raw.get("record_sha256") != record_hash:
            raise ResumeProtocolError(f"Session record self-hash mismatch: {path}")
        if raw.get("sequence") != expected_sequence:
            raise ResumeProtocolError("Session sequence is not contiguous")
        if (expected_sequence == 1) != (raw.get("mode") == "fresh"):
            raise ResumeProtocolError("Exactly the first session must be fresh")
        if raw.get("previous_session_sha256") != previous_hash:
            raise ResumeProtocolError("Session hash chain is broken")
        if raw.get("run_manifest_sha256") != manifest_hash:
            raise ResumeProtocolError("Session is bound to a different run manifest")
        if raw.get("domain") != domain or raw.get("run_index") != run_index:
            raise ResumeProtocolError("Session domain/run binding mismatch")

        log_evidence = raw.get("log")
        if not isinstance(log_evidence, dict):
            raise ResumeProtocolError("Session log evidence is absent")
        log_path = _assert_relative_to(
            arm_root / str(log_evidence.get("relative_path", "")),
            arm_root,
            "Session log",
        )
        if not log_path.is_file() or file_sha256(log_path) != log_evidence.get("sha256"):
            raise ResumeProtocolError("Session log hash mismatch")

        relay = raw.get("relay_segment")
        if not isinstance(relay, dict):
            raise ResumeProtocolError("Relay segment evidence is absent")
        ledger_path = _assert_relative_to(
            arm_root / str(relay.get("relative_path", "")),
            arm_root,
            "Relay ledger",
        )
        start = relay.get("start_offset")
        end = relay.get("end_offset")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end < start:
            raise ResumeProtocolError("Relay segment offsets are invalid")
        if not ledger_path.is_file():
            raise ResumeProtocolError("Relay ledger is absent")
        with ledger_path.open("rb") as handle:
            prefix = handle.read(start)
            handle.seek(start)
            segment = handle.read(end - start)
        if bytes_sha256(prefix) != relay.get("prefix_sha256"):
            raise ResumeProtocolError("Relay prefix hash mismatch")
        if len(segment) != end - start or bytes_sha256(segment) != relay.get("sha256"):
            raise ResumeProtocolError("Relay segment hash mismatch")
        relay_session = raw.get("relay_session")
        if not isinstance(relay_session, dict):
            raise ResumeProtocolError("Relay session contract is absent")
        prefix_events = _parse_ledger_events(prefix)
        if not prefix_events or _relay_session_contract(prefix_events[0]) != relay_session:
            raise ResumeProtocolError("Relay session contract mismatch")

        target_ids = raw.get("target_task_ids")
        if not isinstance(target_ids, list) or not all(isinstance(item, str) for item in target_ids):
            raise ResumeProtocolError("Session target IDs are invalid")
        if raw.get("target_task_ids_sha256") != task_ids_sha256(target_ids):
            raise ResumeProtocolError("Session target-ID hash mismatch")
        if (
            set(raw["pre_state"]) != manifest_task_set
            or set(raw["post_state"]) != manifest_task_set
            or any(task_id not in manifest_task_set for task_id in target_ids)
        ):
            raise ResumeProtocolError("Session task states do not match the run manifest")
        if raw["mode"] == "fresh" and target_ids != manifest_task_ids:
            raise ResumeProtocolError("Fresh session must target the complete manifest task list")
        if raw["mode"] in {"resume_noop", "resume_rejected"} and target_ids:
            raise ResumeProtocolError("No-op/rejected session must not target trajectories")
        if raw["mode"] == "resume_score" and len(target_ids) != 1:
            raise ResumeProtocolError("Score-only session must target exactly one trajectory")
        if raw["mode"] == "resume_agent" and len(target_ids) != 1:
            raise ResumeProtocolError("Agent Resume must target exactly one trajectory")
        required_pre_state = {
            "fresh": "missing",
            "resume_agent": "missing",
            "resume_score": "unscored",
        }.get(raw["mode"])
        if required_pre_state is not None and any(
            raw["pre_state"][task_id]["state"] != required_pre_state for task_id in target_ids
        ):
            raise ResumeProtocolError("Session target has an invalid pre-state")
        for task_id in manifest_task_ids:
            if task_id not in target_ids and raw["pre_state"][task_id] != raw["post_state"][task_id]:
                raise ResumeProtocolError("Session changed a non-target trajectory")
            if (
                raw["pre_state"][task_id]["state"] == "scored"
                and raw["pre_state"][task_id] != raw["post_state"][task_id]
            ):
                raise ResumeProtocolError("Session changed a fully scored trajectory")
        _verify_fresh_batch_members(
            arm_root=arm_root,
            domain=domain,
            manifest_path=run_manifest_path,
            manifest_hash=manifest_hash,
            manifest_task_ids=manifest_task_ids,
            current_record=raw,
        )
        records.append(raw)
        previous_hash = record_hash
    return VerifiedSessionChain(tuple(records))


def _parse_ledger_events(payload: bytes) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in payload.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ResumeProtocolError("Relay evidence contains malformed JSONL") from exc
        if not isinstance(event, dict):
            raise ResumeProtocolError("Relay event must be an object")
        events.append(event)
    return events


def _relay_session_contract(event: Mapping[str, Any]) -> dict[str, Any]:
    origin_hash = event.get("upstream_origin_sha256")
    contract = {
        "event": event.get("event"),
        "provider": event.get("provider"),
        "upstream_origin_sha256": origin_hash,
        "rpm": event.get("rpm"),
        "burst": event.get("burst"),
        "burst_window_seconds": event.get("burst_window_seconds"),
        "attempts": event.get("attempts"),
    }
    if (
        contract["event"] != "session_start"
        or contract["provider"] != "novacode"
        or not isinstance(origin_hash, str)
        or not re.fullmatch(r"[0-9a-f]{64}", origin_hash)
        or contract["rpm"] != 45
        or contract["burst"] != 5
        or float(contract["burst_window_seconds"] or 0) != 1.0
        or contract["attempts"] != 5
    ):
        raise ResumeProtocolError("Relay session does not match the locked transport contract")
    return contract


def _read_ledger_segment(
    path: Path,
    start_offset: int,
) -> tuple[bytes, bytes, dict[str, Any], list[dict[str, Any]]]:
    if start_offset < 0:
        raise ResumeProtocolError("Relay start offset must be non-negative")
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        end_offset = handle.tell()
        if start_offset > end_offset:
            raise ResumeProtocolError("Relay start offset is beyond end of ledger")
        handle.seek(0)
        prefix = handle.read(start_offset)
        handle.seek(start_offset)
        payload = handle.read()
    prefix_events = _parse_ledger_events(prefix)
    if not prefix_events:
        raise ResumeProtocolError("Relay ledger prefix has no session_start event")
    session_contract = _relay_session_contract(prefix_events[0])
    return prefix, payload, session_contract, _parse_ledger_events(payload)


def _exhausted_transport_counts(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    configured_attempts = 5
    for event in events:
        if event.get("event") == "session_start" and isinstance(event.get("attempts"), int):
            configured_attempts = int(event["attempts"])
    exhausted: dict[str, set[Any]] = {}
    for event in events:
        attempt = event.get("attempt")
        request_id = event.get("request_id")
        route = str(event.get("route", "other"))
        status = event.get("status_code")
        is_exhausted = (
            isinstance(attempt, int)
            and attempt >= configured_attempts
            and (
                event.get("event") == "transport_error"
                or (
                    event.get("event") == "upstream_response"
                    and event.get("retryable_status") is True
                    and status in RETRYABLE_STATUS_CODES
                )
            )
        )
        if is_exhausted:
            exhausted.setdefault(route, set()).add(request_id)
    return {route: len(request_ids) for route, request_ids in sorted(exhausted.items())}


def _line_for_task(log_text: str, task_id: str, pattern: str) -> bool:
    task = re.escape(task_id)
    expression = re.compile(pattern.format(task=task), re.IGNORECASE)
    return any(expression.search(line) for line in log_text.splitlines())


def _task_transport_error(
    log_text: str,
    task_id: str,
    *,
    phase: str,
    run_index: int,
) -> bool:
    task = re.escape(task_id)
    if phase == "agent_or_run":
        # Multi-run logs reuse task IDs across runs.  Agent errors are printed
        # in a per-run summary block, so bind the free-text error to that exact
        # block.  If a multi-run process died before emitting its summaries,
        # there is deliberately no per-run proof and Resume fails closed.
        block_match = re.search(
            rf"(?ms)^\s*Run\s+{run_index}\s+done\b[^\n]*\n"
            rf"(?P<body>.*?)(?=^\s*Run\s+\d+\s+done\b|\Z)",
            log_text,
        )
        if block_match:
            expression = re.compile(
                rf"\bERROR\s+{task}\s*:\s*(?P<error>.*)$",
                re.IGNORECASE | re.MULTILINE,
            )
            return any(
                TRANSPORT_ERROR_PATTERN.search(match.group("error")) is not None
                for match in expression.finditer(block_match.group("body"))
            )
        if re.search(r"#\s*Runs\s+\d+\.\.\d+\s+together", log_text, re.IGNORECASE):
            return False
        expression = re.compile(
            rf"\bERROR\s+{task}\s*:\s*(?P<error>.*)$",
            re.IGNORECASE,
        )
    elif phase == "scoring":
        expression = re.compile(
            rf"\brun{run_index}\s+{task}\s*:\s*OK.*score=ERR.*scoring_error=(?P<error>.*)$",
            re.IGNORECASE,
        )
    else:  # pragma: no cover - internal caller is closed over two literals
        raise ValueError(f"unknown failure phase: {phase}")
    for line in log_text.splitlines():
        match = expression.search(line)
        if match and TRANSPORT_ERROR_PATTERN.search(match.group("error")):
            return True
    return False


def _classify_retryable_tasks(
    *,
    mode: str,
    target_task_ids: Sequence[str],
    post_tasks: Mapping[str, Mapping[str, Any]],
    log_text: str,
    exhausted_counts: Mapping[str, int],
    run_index: int,
) -> tuple[list[str], list[str]]:
    agent_or_run: list[str] = []
    scoring: list[str] = []
    any_exhausted = sum(exhausted_counts.values()) > 0
    judge_exhausted = exhausted_counts.get("official_eval_responses", 0) > 0

    for task_id in target_task_ids:
        state = post_tasks[task_id]["state"]
        if state == "missing" and mode in {"fresh", "resume_agent"}:
            task_transport = _task_transport_error(
                log_text,
                task_id,
                phase="agent_or_run",
                run_index=run_index,
            )
            if any_exhausted and task_transport:
                agent_or_run.append(task_id)
        elif state == "unscored" and mode in {"fresh", "resume_agent"}:
            task_transport = _task_transport_error(
                log_text,
                task_id,
                phase="scoring",
                run_index=run_index,
            )
            if judge_exhausted and task_transport:
                scoring.append(task_id)
        elif state == "unscored" and mode == "resume_score":
            # Score retries are isolated to exactly one trajectory.  score.py
            # prints the task status but deliberately omits exception text, so
            # the exhausted official-eval request is the transport proof.
            score_error = _line_for_task(log_text, task_id, r"\b{task}\s*:\s*ERR\b")
            if len(target_task_ids) == 1 and judge_exhausted and score_error:
                scoring.append(task_id)
    return sorted(agent_or_run), sorted(scoring)


def write_session_record(
    *,
    arm_root: Path,
    domain: str,
    run_index: int,
    run_manifest_path: Path,
    mode: str,
    all_task_ids: Sequence[str],
    target_task_ids: Sequence[str],
    pre_snapshot: Mapping[str, Any],
    log_path: Path,
    relay_ledger_path: Path,
    relay_start_offset: int,
    process_exit_code: int,
    fresh_batch: Mapping[str, Any] | None = None,
) -> Path:
    if mode not in {"fresh", "resume_agent", "resume_score", "resume_rejected", "resume_noop"}:
        raise ResumeProtocolError("Unknown session mode")
    arm_root = arm_root.resolve()
    manifest_path = _assert_relative_to(run_manifest_path, arm_root, "Run manifest")
    manifest = load_verified_manifest(manifest_path)
    if manifest.get("domain") != domain:
        raise ResumeProtocolError("Run manifest domain mismatch")
    all_ids = _normalize_task_ids(all_task_ids)
    targets = list(target_task_ids)
    if len(targets) != len(set(targets)) or any(item not in all_ids for item in targets):
        raise ResumeProtocolError("Session target IDs are not a unique subset of the manifest tasks")
    if mode == "fresh" and targets != all_ids:
        raise ResumeProtocolError("Fresh session must target every manifest task in order")
    if fresh_batch is not None and mode != "fresh":
        raise ResumeProtocolError("Only a fresh session may reference a fresh batch")
    if mode in {"resume_noop", "resume_rejected"} and targets:
        raise ResumeProtocolError("No-op/rejected session must not target trajectories")
    if mode == "resume_score" and len(targets) != 1:
        raise ResumeProtocolError("Score-only session must target exactly one trajectory")
    if mode == "resume_agent" and len(targets) != 1:
        raise ResumeProtocolError("Agent Resume must target exactly one trajectory")
    manifest_ids = manifest.get("run", {}).get("task_selection", {}).get("task_ids")
    if manifest_ids != all_ids:
        raise ResumeProtocolError("Session task list does not exactly match the run manifest")

    expected_pre_hash = task_ids_sha256(all_ids)
    if pre_snapshot.get("task_ids_sha256") != expected_pre_hash:
        raise ResumeProtocolError("Pre-session snapshot task hash mismatch")
    pre_tasks = pre_snapshot.get("tasks")
    if not isinstance(pre_tasks, dict) or set(pre_tasks) != set(all_ids):
        raise ResumeProtocolError("Pre-session snapshot task set mismatch")
    run_dir = arm_root / domain / f"run{run_index}"
    post_snapshot = snapshot_run(run_dir, all_ids)
    post_tasks = post_snapshot["tasks"]

    expected_pre_state = {
        "fresh": "missing",
        "resume_agent": "missing",
        "resume_score": "unscored",
    }.get(mode)
    if expected_pre_state is not None:
        invalid = [task_id for task_id in targets if pre_tasks[task_id]["state"] != expected_pre_state]
        if invalid:
            raise ResumeProtocolError(f"{mode} targeted trajectories in an invalid pre-state")
    for task_id in all_ids:
        if task_id not in targets and pre_tasks[task_id] != post_tasks[task_id]:
            raise ResumeProtocolError("A non-target trajectory changed during the session")
        if pre_tasks[task_id]["state"] == "scored" and pre_tasks[task_id] != post_tasks[task_id]:
            raise ResumeProtocolError("A fully scored trajectory changed during the session")

    log_path = _assert_relative_to(log_path, arm_root, "Session log")
    ledger_path = _assert_relative_to(relay_ledger_path, arm_root, "Relay ledger")
    if not log_path.is_file() or not ledger_path.is_file():
        raise ResumeProtocolError("Session log or relay ledger is absent")
    log_bytes = log_path.read_bytes()
    try:
        log_text = log_bytes.decode("utf-8", errors="replace")
    except UnicodeError as exc:  # pragma: no cover - replace is total
        raise ResumeProtocolError("Cannot decode captured session log") from exc
    prefix_bytes, segment_bytes, relay_session, segment_events = _read_ledger_segment(
        ledger_path,
        relay_start_offset,
    )
    exhausted_counts = _exhausted_transport_counts(segment_events)
    agent_proof, scoring_proof = _classify_retryable_tasks(
        mode=mode,
        target_task_ids=targets,
        post_tasks=post_tasks,
        log_text=log_text,
        exhausted_counts=exhausted_counts,
        run_index=run_index,
    )

    chain = verify_session_chain(
        arm_root=arm_root,
        domain=domain,
        run_index=run_index,
        run_manifest_path=manifest_path,
    )
    if (not chain.records) != (mode == "fresh"):
        raise ResumeProtocolError("Exactly the first session must use fresh mode")
    previous_hash = chain.records[-1]["record_sha256"] if chain.records else None
    sequence = len(chain.records) + 1
    session_id = uuid.uuid4().hex
    record_core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "auditable_resume_session",
        "session_id": session_id,
        "sequence": sequence,
        "previous_session_sha256": previous_hash,
        "run_manifest_relative_path": manifest_path.relative_to(arm_root).as_posix(),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "domain": domain,
        "run_index": run_index,
        "mode": mode,
        "command_kind": "score" if mode == "resume_score" else ("none" if mode in {"resume_noop", "resume_rejected"} else "run_batch"),
        "target_task_ids": targets,
        "target_task_ids_sha256": task_ids_sha256(targets),
        "pre_state": pre_tasks,
        "post_state": post_tasks,
        "process_exit_code": int(process_exit_code),
        "log": {
            "relative_path": log_path.relative_to(arm_root).as_posix(),
            "sha256": bytes_sha256(log_bytes),
            "size_bytes": len(log_bytes),
        },
        "relay_segment": {
            "relative_path": ledger_path.relative_to(arm_root).as_posix(),
            "start_offset": relay_start_offset,
            "end_offset": relay_start_offset + len(segment_bytes),
            "prefix_sha256": bytes_sha256(prefix_bytes),
            "sha256": bytes_sha256(segment_bytes),
            "exhausted_request_counts_by_route": exhausted_counts,
        },
        "relay_session": relay_session,
        "transport_proof": {
            "classification": "transport_failure" if agent_proof or scoring_proof else "none",
            "agent_or_run_task_ids": agent_proof,
            "scoring_task_ids": scoring_proof,
        },
        "fresh_batch": dict(fresh_batch) if fresh_batch is not None else None,
    }
    record = dict(record_core)
    record["record_sha256"] = canonical_sha256(record_core)
    _validate_record_shape(record)
    record_path = _session_dir(arm_root, domain, run_index) / f"{sequence:06d}-{session_id}.json"
    _write_json_exclusive(record_path, record)
    _make_read_only(log_path)
    for task_id, state in post_tasks.items():
        if state["state"] == "scored":
            _make_read_only(run_dir / f"{task_id}.json")
    return record_path


def write_official_fresh_batch_records(
    *,
    arm_root: Path,
    domain: str,
    run_manifest_path: Path,
    all_task_ids: Sequence[str],
    pre_snapshots: Mapping[int, Mapping[str, Any]],
    log_path: Path,
    relay_ledger_path: Path,
    relay_start_offset: int,
    process_exit_code: int,
    split: str,
    num_runs: int,
    num_runs_idx_start: int,
) -> tuple[Path, ...]:
    """Project one official five-run invocation into five auditable run chains.

    The shared immutable batch record binds the exact official CLI contract,
    log, and relay byte range once.  Each run gets a first-chain projection with
    its own pre/post trajectory state, which lets later Resume remain strictly
    single-run and single-task without ever resampling scored trajectories.
    """

    command_contract = {
        "module": "state_bench.scripts.run_batch",
        "split": split,
        "num_runs": num_runs,
        "num_runs_idx_start": num_runs_idx_start,
    }
    if command_contract != OFFICIAL_FRESH_BATCH_COMMAND:
        raise ResumeProtocolError(
            "Official fresh batch must be --split test --num-runs 5 "
            "--num-runs-idx-start 1"
        )
    arm_root = arm_root.resolve()
    manifest_path = _assert_relative_to(run_manifest_path, arm_root, "Run manifest")
    manifest = load_verified_manifest(manifest_path)
    if manifest.get("domain") != domain:
        raise ResumeProtocolError("Run manifest domain mismatch")
    all_ids = _normalize_task_ids(all_task_ids)
    if manifest.get("run", {}).get("task_selection", {}).get("task_ids") != all_ids:
        raise ResumeProtocolError("Fresh-batch task list does not match immutable manifest")
    run_indices = tuple(sorted(pre_snapshots))
    if run_indices != OFFICIAL_FRESH_BATCH_RUN_INDICES:
        raise ResumeProtocolError("Official fresh batch requires pre-snapshots for runs 1..5")
    expected_task_hash = task_ids_sha256(all_ids)
    for run_index in run_indices:
        pre = pre_snapshots[run_index]
        if pre.get("task_ids_sha256") != expected_task_hash:
            raise ResumeProtocolError("Fresh-batch pre-snapshot task hash mismatch")
        tasks = pre.get("tasks")
        if not isinstance(tasks, dict) or set(tasks) != set(all_ids):
            raise ResumeProtocolError("Fresh-batch pre-snapshot task set mismatch")
        _validate_task_state_map(tasks)
        if any(tasks[task_id]["state"] != "missing" for task_id in all_ids):
            raise ResumeProtocolError("Official fresh batch may only start from missing trajectories")
        if verify_session_chain(
            arm_root=arm_root,
            domain=domain,
            run_index=run_index,
            run_manifest_path=manifest_path,
        ).records:
            raise ResumeProtocolError("Official fresh batch requires empty session chains")

    resolved_log = _assert_relative_to(log_path, arm_root, "Session log")
    resolved_ledger = _assert_relative_to(relay_ledger_path, arm_root, "Relay ledger")
    if not resolved_log.is_file() or not resolved_ledger.is_file():
        raise ResumeProtocolError("Fresh-batch log or relay ledger is absent")
    log_bytes = resolved_log.read_bytes()
    prefix_bytes, segment_bytes, relay_session, segment_events = _read_ledger_segment(
        resolved_ledger,
        relay_start_offset,
    )
    relay_evidence = {
        "relative_path": resolved_ledger.relative_to(arm_root).as_posix(),
        "start_offset": relay_start_offset,
        "end_offset": relay_start_offset + len(segment_bytes),
        "prefix_sha256": bytes_sha256(prefix_bytes),
        "sha256": bytes_sha256(segment_bytes),
        "exhausted_request_counts_by_route": _exhausted_transport_counts(segment_events),
    }
    log_evidence = {
        "relative_path": resolved_log.relative_to(arm_root).as_posix(),
        "sha256": bytes_sha256(log_bytes),
        "size_bytes": len(log_bytes),
    }
    batch_id = uuid.uuid4().hex
    batch_core: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_type": "auditable_fresh_batch",
        "batch_id": batch_id,
        "domain": domain,
        "run_indices": list(run_indices),
        "command_kind": "run_batch",
        "command_contract": command_contract,
        "run_manifest_relative_path": manifest_path.relative_to(arm_root).as_posix(),
        "run_manifest_sha256": manifest["manifest_sha256"],
        "target_task_ids": all_ids,
        "target_task_ids_sha256": expected_task_hash,
        "process_exit_code": int(process_exit_code),
        "log": log_evidence,
        "relay_segment": relay_evidence,
        "relay_session": relay_session,
    }
    batch_record = dict(batch_core)
    batch_record_hash = canonical_sha256(batch_core)
    batch_record["batch_record_sha256"] = batch_record_hash
    batch_path = _fresh_batch_record_path(arm_root, domain, batch_id)
    _write_json_exclusive(batch_path, batch_record)
    reference_base = {
        "relative_path": batch_path.relative_to(arm_root).as_posix(),
        "sha256": batch_record_hash,
        "batch_id": batch_id,
    }

    record_paths: list[Path] = []
    for run_index in run_indices:
        reference = {**reference_base, "member_run_index": run_index}
        record_paths.append(
            write_session_record(
                arm_root=arm_root,
                domain=domain,
                run_index=run_index,
                run_manifest_path=manifest_path,
                mode="fresh",
                all_task_ids=all_ids,
                target_task_ids=all_ids,
                pre_snapshot=pre_snapshots[run_index],
                log_path=resolved_log,
                relay_ledger_path=resolved_ledger,
                relay_start_offset=relay_start_offset,
                process_exit_code=process_exit_code,
                fresh_batch=reference,
            )
        )
    for run_index in run_indices:
        verify_session_chain(
            arm_root=arm_root,
            domain=domain,
            run_index=run_index,
            run_manifest_path=manifest_path,
        )
    return tuple(record_paths)


def plan_resume(
    *,
    arm_root: Path,
    domain: str,
    run_index: int,
    run_manifest_path: Path,
    task_ids: Sequence[str],
) -> dict[str, Any]:
    manifest = load_verified_manifest(run_manifest_path)
    ids = _normalize_task_ids(task_ids)
    if manifest.get("run", {}).get("task_selection", {}).get("task_ids") != ids:
        raise ResumeProtocolError("Resume task list does not match immutable manifest")
    chain = verify_session_chain(
        arm_root=arm_root,
        domain=domain,
        run_index=run_index,
        run_manifest_path=run_manifest_path,
    )
    latest = chain.latest_by_task
    current = snapshot_run(arm_root / domain / f"run{run_index}", ids)["tasks"]
    agent_tasks: list[str] = []
    score_tasks: list[str] = []
    scored_tasks: list[str] = []
    rejected_tasks: list[str] = []
    for task_id in ids:
        state = current[task_id]
        record = latest.get(task_id)
        if record is None or record.get("post_state", {}).get(task_id) != state:
            rejected_tasks.append(task_id)
            continue
        if state["state"] == "scored":
            scored_tasks.append(task_id)
            continue
        proof = record.get("transport_proof", {})
        if proof.get("classification") != "transport_failure":
            rejected_tasks.append(task_id)
        elif state["state"] == "missing" and task_id in proof.get("agent_or_run_task_ids", []):
            agent_tasks.append(task_id)
        elif state["state"] == "unscored" and task_id in proof.get("scoring_task_ids", []):
            score_tasks.append(task_id)
        else:
            rejected_tasks.append(task_id)
    return {
        "schema_version": SCHEMA_VERSION,
        "domain": domain,
        "run_index": run_index,
        "agent_task_ids": agent_tasks,
        "score_task_ids": score_tasks,
        "scored_task_ids": scored_tasks,
        "rejected_task_ids": rejected_tasks,
        "latest_session_sha256": chain.records[-1]["record_sha256"] if chain.records else None,
    }


def stage_score_only(
    *,
    source: Path,
    destination: Path,
    task_id: str,
    expected_source_sha256: str,
) -> None:
    state = trajectory_state(source, task_id=task_id)
    if state["state"] != "unscored" or state["sha256"] != expected_source_sha256:
        raise ResumeProtocolError("Score-only source is no longer the proved unscored trajectory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("xb") as output, source.open("rb") as input_handle:
            shutil.copyfileobj(input_handle, output)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError as exc:
        raise ResumeProtocolError("Refusing to overwrite score-only staging data") from exc


def promote_score_only(
    *,
    staged: Path,
    destination: Path,
    task_id: str,
    expected_destination_sha256: str,
) -> bool:
    destination_state = trajectory_state(destination, task_id=task_id)
    if (
        destination_state["state"] != "unscored"
        or destination_state["sha256"] != expected_destination_sha256
    ):
        raise ResumeProtocolError("Canonical unscored trajectory changed during isolated scoring")
    staged_state = trajectory_state(staged, task_id=task_id)
    if staged_state["state"] != "scored":
        return False
    temporary = destination.with_name(f".{destination.name}.promote-{uuid.uuid4().hex}")
    shutil.copyfile(staged, temporary)
    try:
        if file_sha256(destination) != expected_destination_sha256:
            raise ResumeProtocolError("Canonical trajectory changed before atomic promotion")
        os.replace(temporary, destination)
        _make_read_only(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return True


def _write_snapshot_command(args: argparse.Namespace) -> None:
    task_ids = _load_task_ids(Path(args.task_ids_json))
    snapshot = snapshot_run(Path(args.run_dir), task_ids)
    _write_json_exclusive(Path(args.output), snapshot)


def _plan_command(args: argparse.Namespace) -> None:
    plan = plan_resume(
        arm_root=Path(args.arm_root),
        domain=args.domain,
        run_index=args.run_index,
        run_manifest_path=Path(args.run_manifest),
        task_ids=_load_task_ids(Path(args.task_ids_json)),
    )
    print(json.dumps(plan, ensure_ascii=True, sort_keys=True, separators=(",", ":")))


def _record_command(args: argparse.Namespace) -> None:
    all_ids = _load_task_ids(Path(args.task_ids_json))
    targets = _load_task_ids(Path(args.target_task_ids_json)) if args.target_task_ids_json else []
    pre = _read_json(Path(args.pre_snapshot))
    if not isinstance(pre, dict):
        raise ResumeProtocolError("Pre-session snapshot must be an object")
    path = write_session_record(
        arm_root=Path(args.arm_root),
        domain=args.domain,
        run_index=args.run_index,
        run_manifest_path=Path(args.run_manifest),
        mode=args.mode,
        all_task_ids=all_ids,
        target_task_ids=targets,
        pre_snapshot=pre,
        log_path=Path(args.log),
        relay_ledger_path=Path(args.relay_ledger),
        relay_start_offset=args.relay_start_offset,
        process_exit_code=args.process_exit_code,
    )
    print(path)


def _record_official_batch_command(args: argparse.Namespace) -> None:
    all_ids = _load_task_ids(Path(args.task_ids_json))
    inputs = _read_json(Path(args.pre_snapshots_json))
    if not isinstance(inputs, dict) or set(inputs) != {"run_indices", "pre_snapshot_paths"}:
        raise ResumeProtocolError("Fresh-batch snapshot input has an invalid shape")
    run_indices = inputs.get("run_indices")
    raw_paths = inputs.get("pre_snapshot_paths")
    if (
        run_indices != list(OFFICIAL_FRESH_BATCH_RUN_INDICES)
        or not isinstance(raw_paths, dict)
        or set(raw_paths) != {str(index) for index in OFFICIAL_FRESH_BATCH_RUN_INDICES}
        or not all(isinstance(value, str) for value in raw_paths.values())
    ):
        raise ResumeProtocolError("Fresh-batch snapshot input must map runs 1..5")
    pre_snapshots: dict[int, Mapping[str, Any]] = {}
    for run_index in OFFICIAL_FRESH_BATCH_RUN_INDICES:
        value = _read_json(Path(raw_paths[str(run_index)]))
        if not isinstance(value, dict):
            raise ResumeProtocolError("Fresh-batch pre-snapshot must be an object")
        pre_snapshots[run_index] = value
    paths = write_official_fresh_batch_records(
        arm_root=Path(args.arm_root),
        domain=args.domain,
        run_manifest_path=Path(args.run_manifest),
        all_task_ids=all_ids,
        pre_snapshots=pre_snapshots,
        log_path=Path(args.log),
        relay_ledger_path=Path(args.relay_ledger),
        relay_start_offset=args.relay_start_offset,
        process_exit_code=args.process_exit_code,
        split=args.split,
        num_runs=args.num_runs,
        num_runs_idx_start=args.num_runs_idx_start,
    )
    print(json.dumps([str(path) for path in paths], separators=(",", ":")))


def _stage_score_command(args: argparse.Namespace) -> None:
    stage_score_only(
        source=Path(args.source),
        destination=Path(args.destination),
        task_id=args.task_id,
        expected_source_sha256=args.expected_source_sha256,
    )


def _promote_score_command(args: argparse.Namespace) -> None:
    promoted = promote_score_only(
        staged=Path(args.staged),
        destination=Path(args.destination),
        task_id=args.task_id,
        expected_destination_sha256=args.expected_destination_sha256,
    )
    print(json.dumps({"promoted": promoted}, separators=(",", ":")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    snapshot_parser = subparsers.add_parser("snapshot")
    snapshot_parser.add_argument("--run-dir", required=True)
    snapshot_parser.add_argument("--task-ids-json", required=True)
    snapshot_parser.add_argument("--output", required=True)
    snapshot_parser.set_defaults(handler=_write_snapshot_command)

    plan_parser = subparsers.add_parser("plan")
    plan_parser.add_argument("--arm-root", required=True)
    plan_parser.add_argument("--domain", required=True)
    plan_parser.add_argument("--run-index", required=True, type=int)
    plan_parser.add_argument("--run-manifest", required=True)
    plan_parser.add_argument("--task-ids-json", required=True)
    plan_parser.set_defaults(handler=_plan_command)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--arm-root", required=True)
    record_parser.add_argument("--domain", required=True)
    record_parser.add_argument("--run-index", required=True, type=int)
    record_parser.add_argument("--run-manifest", required=True)
    record_parser.add_argument(
        "--mode",
        required=True,
        choices=["fresh", "resume_agent", "resume_score", "resume_rejected", "resume_noop"],
    )
    record_parser.add_argument("--task-ids-json", required=True)
    record_parser.add_argument("--target-task-ids-json")
    record_parser.add_argument("--pre-snapshot", required=True)
    record_parser.add_argument("--log", required=True)
    record_parser.add_argument("--relay-ledger", required=True)
    record_parser.add_argument("--relay-start-offset", required=True, type=int)
    record_parser.add_argument("--process-exit-code", required=True, type=int)
    record_parser.set_defaults(handler=_record_command)

    batch_parser = subparsers.add_parser("record-official-batch")
    batch_parser.add_argument("--arm-root", required=True)
    batch_parser.add_argument("--domain", required=True)
    batch_parser.add_argument("--run-manifest", required=True)
    batch_parser.add_argument("--task-ids-json", required=True)
    batch_parser.add_argument("--pre-snapshots-json", required=True)
    batch_parser.add_argument("--log", required=True)
    batch_parser.add_argument("--relay-ledger", required=True)
    batch_parser.add_argument("--relay-start-offset", required=True, type=int)
    batch_parser.add_argument("--process-exit-code", required=True, type=int)
    batch_parser.add_argument("--split", required=True)
    batch_parser.add_argument("--num-runs", required=True, type=int)
    batch_parser.add_argument("--num-runs-idx-start", required=True, type=int)
    batch_parser.set_defaults(handler=_record_official_batch_command)

    stage_parser = subparsers.add_parser("stage-score")
    stage_parser.add_argument("--source", required=True)
    stage_parser.add_argument("--destination", required=True)
    stage_parser.add_argument("--task-id", required=True)
    stage_parser.add_argument("--expected-source-sha256", required=True)
    stage_parser.set_defaults(handler=_stage_score_command)

    promote_parser = subparsers.add_parser("promote-score")
    promote_parser.add_argument("--staged", required=True)
    promote_parser.add_argument("--destination", required=True)
    promote_parser.add_argument("--task-id", required=True)
    promote_parser.add_argument("--expected-destination-sha256", required=True)
    promote_parser.set_defaults(handler=_promote_score_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except ResumeProtocolError as exc:
        print(f"Resume protocol rejected: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
