from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.resume_protocol import (
    ResumeProtocolError,
    canonical_sha256,
    file_sha256,
    is_fully_scored,
    plan_resume,
    promote_score_only,
    snapshot_run,
    stage_score_only,
    verify_session_chain,
    write_official_fresh_batch_records,
    write_session_record,
)


DOMAIN = "shopping_assistant"
TASK_ID = "synthetic-task"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _manifest(arm_root: Path) -> Path:
    core = {
        "schema_version": "1.0.0",
        "domain": DOMAIN,
        "run": {"task_selection": {"task_ids": [TASK_ID]}},
    }
    value = dict(core)
    value["manifest_sha256"] = canonical_sha256(core)
    path = arm_root / DOMAIN / "run_manifest.json"
    _write_json(path, value)
    return path


def _raw_trajectory() -> dict[str, object]:
    return {
        "task_id": TASK_ID,
        "conversation": [{"role": "user", "content": "must never enter an audit record"}],
        "state_diff": {"created": {}, "modified": {}, "deleted": {}},
    }


def _scored_trajectory(*, completion: int) -> dict[str, object]:
    value = _raw_trajectory()
    value.update(
        {
            "task_completion_pass": completion,
            "state_requirements_met": completion,
            "task_requirements_met": completion,
            "ux_score": 4.0,
            "scoring_protocol_id": "state_bench_v0.8.1_gpt54",
            "judge_model": "gpt-5.4",
            "judge_reasoning_effort": "high",
            "judge_prompt_hashes": {"system": "0" * 64},
            "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
            "simulator_model": "gpt-5.4",
            "simulator_prompt_hash": "1" * 64,
            "agent_model": {"model_name": "gpt-5.4", "reasoning_level": None},
        }
    )
    return value


def _ledger(path: Path, *, exhausted: bool) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    session_start = {
        "schema_version": "1.0.0",
        "event": "session_start",
        "provider": "novacode",
        "upstream_origin_sha256": "a" * 64,
        "rpm": 45,
        "burst": 5,
        "burst_window_seconds": 1.0,
        "attempts": 5,
    }
    records = []
    if exhausted:
        records = [
            {
                "schema_version": "1.0.0",
                "event": "upstream_response",
                "request_id": 7,
                "attempt": attempt,
                "route": "official_eval_responses",
                "status_code": 502,
                "retryable_status": True,
            }
            for attempt in range(1, 6)
        ]
    prefix = json.dumps(session_start, sort_keys=True) + "\n"
    path.write_text(prefix + "".join(json.dumps(item, sort_keys=True) + "\n" for item in records), encoding="utf-8")
    return len(prefix.encode("utf-8"))


def _record_fresh(
    arm_root: Path,
    *,
    trajectory: dict[str, object],
    log_line: str,
    exhausted: bool,
) -> tuple[Path, Path]:
    manifest = _manifest(arm_root)
    run_dir = arm_root / DOMAIN / "run1"
    before = snapshot_run(run_dir, [TASK_ID])
    _write_json(run_dir / f"{TASK_ID}.json", trajectory)
    log = arm_root / "_batch_logs" / "fresh.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(log_line + "\n", encoding="utf-8")
    ledger = arm_root / "_transport" / "relay.jsonl"
    relay_start = _ledger(ledger, exhausted=exhausted)
    record = write_session_record(
        arm_root=arm_root,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        mode="fresh",
        all_task_ids=[TASK_ID],
        target_task_ids=[TASK_ID],
        pre_snapshot=before,
        log_path=log,
        relay_ledger_path=ledger,
        relay_start_offset=relay_start,
        process_exit_code=0,
    )
    return manifest, record


def test_scored_failure_is_immutable_and_never_planned_for_resume(tmp_path: Path) -> None:
    manifest, _ = _record_fresh(
        tmp_path,
        trajectory=_scored_trajectory(completion=0),
        log_line=f"[1/1] run1 {TASK_ID}: OK | score=OK | ux=4.0",
        exhausted=False,
    )
    trajectory_path = tmp_path / DOMAIN / "run1" / f"{TASK_ID}.json"
    before_hash = file_sha256(trajectory_path)

    plan = plan_resume(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        task_ids=[TASK_ID],
    )

    assert plan["scored_task_ids"] == [TASK_ID]
    assert plan["agent_task_ids"] == []
    assert plan["score_task_ids"] == []
    assert plan["rejected_task_ids"] == []
    assert file_sha256(trajectory_path) == before_hash


def test_replaced_scored_trajectory_is_rejected_instead_of_trusted(tmp_path: Path) -> None:
    manifest, _ = _record_fresh(
        tmp_path,
        trajectory=_scored_trajectory(completion=0),
        log_line=f"[1/1] run1 {TASK_ID}: OK | score=OK | ux=4.0",
        exhausted=False,
    )
    trajectory_path = tmp_path / DOMAIN / "run1" / f"{TASK_ID}.json"
    trajectory_path.chmod(0o666)
    replacement = _scored_trajectory(completion=1)
    _write_json(trajectory_path, replacement)

    plan = plan_resume(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        task_ids=[TASK_ID],
    )

    assert plan["rejected_task_ids"] == [TASK_ID]
    assert plan["scored_task_ids"] == []


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("state_requirements_met", None),
        ("task_requirements_met", 0.5),
        ("evaluation_protocol_id", "unofficial"),
        ("simulator_model", "gpt-5.4-mini"),
        ("simulator_prompt_hash", None),
        ("agent_model", {"model_name": "gpt-5.4-mini", "reasoning_level": None}),
    ],
)
def test_incomplete_or_unofficial_score_metadata_is_not_immutable(
    field: str,
    invalid: object,
) -> None:
    trajectory = _scored_trajectory(completion=1)
    trajectory[field] = invalid
    assert not is_fully_scored(trajectory, task_id=TASK_ID)


def test_nontransport_unscored_trajectory_is_rejected_fail_closed(tmp_path: Path) -> None:
    manifest, record_path = _record_fresh(
        tmp_path,
        trajectory=_raw_trajectory(),
        log_line=f"[1/1] run1 {TASK_ID}: OK | score=ERR | scoring_error=invalid judge JSON",
        exhausted=False,
    )

    plan = plan_resume(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        task_ids=[TASK_ID],
    )

    assert plan["rejected_task_ids"] == [TASK_ID]
    assert plan["agent_task_ids"] == []
    assert plan["score_task_ids"] == []
    record_text = record_path.read_text(encoding="utf-8")
    assert "must never enter an audit record" not in record_text
    assert "invalid judge JSON" not in record_text
    assert "conversation" not in record_text


def test_transport_scoring_failure_selects_score_only_not_agent(tmp_path: Path) -> None:
    manifest, _ = _record_fresh(
        tmp_path,
        trajectory=_raw_trajectory(),
        log_line=(
            f"[1/1] run1 {TASK_ID}: OK | score=ERR | "
            "scoring_error=Error code: 502 shim upstream failed"
        ),
        exhausted=True,
    )

    plan = plan_resume(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        task_ids=[TASK_ID],
    )

    assert plan["score_task_ids"] == [TASK_ID]
    assert plan["agent_task_ids"] == []
    assert plan["rejected_task_ids"] == []


def test_transport_agent_failure_is_the_only_way_to_rerun_missing(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    run_dir = tmp_path / DOMAIN / "run1"
    before = snapshot_run(run_dir, [TASK_ID])
    log = tmp_path / "_batch_logs" / "fresh.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(f"ERROR {TASK_ID}: APIConnectionError shim upstream failed\n", encoding="utf-8")
    ledger = tmp_path / "_transport" / "relay.jsonl"
    relay_start = _ledger(ledger, exhausted=True)
    write_session_record(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        mode="fresh",
        all_task_ids=[TASK_ID],
        target_task_ids=[TASK_ID],
        pre_snapshot=before,
        log_path=log,
        relay_ledger_path=ledger,
        relay_start_offset=relay_start,
        process_exit_code=0,
    )

    plan = plan_resume(
        arm_root=tmp_path,
        domain=DOMAIN,
        run_index=1,
        run_manifest_path=manifest,
        task_ids=[TASK_ID],
    )

    assert plan["agent_task_ids"] == [TASK_ID]
    assert plan["score_task_ids"] == []
    assert plan["rejected_task_ids"] == []


def test_score_only_staging_does_not_touch_canonical_until_atomic_promotion(tmp_path: Path) -> None:
    canonical = tmp_path / DOMAIN / "run1" / f"{TASK_ID}.json"
    staged = tmp_path / "_resume_tmp" / "one" / DOMAIN / "run1" / f"{TASK_ID}.json"
    _write_json(canonical, _raw_trajectory())
    original_hash = file_sha256(canonical)

    stage_score_only(
        source=canonical,
        destination=staged,
        task_id=TASK_ID,
        expected_source_sha256=original_hash,
    )
    _write_json(staged, _scored_trajectory(completion=0))
    assert file_sha256(canonical) == original_hash

    assert promote_score_only(
        staged=staged,
        destination=canonical,
        task_id=TASK_ID,
        expected_destination_sha256=original_hash,
    )
    assert file_sha256(canonical) != original_hash


def test_session_log_tampering_breaks_resume_chain(tmp_path: Path) -> None:
    manifest, record_path = _record_fresh(
        tmp_path,
        trajectory=_raw_trajectory(),
        log_line=(
            f"[1/1] run1 {TASK_ID}: OK | score=ERR | "
            "scoring_error=Error code: 502 shim upstream failed"
        ),
        exhausted=True,
    )
    record = json.loads(record_path.read_text(encoding="utf-8"))
    log_path = tmp_path / record["log"]["relative_path"]
    log_path.chmod(0o666)
    log_path.write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ResumeProtocolError, match="log hash mismatch"):
        plan_resume(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_index=1,
            run_manifest_path=manifest,
            task_ids=[TASK_ID],
        )


def test_session_record_rejects_rehashed_free_text_field(tmp_path: Path) -> None:
    manifest, record_path = _record_fresh(
        tmp_path,
        trajectory=_raw_trajectory(),
        log_line=(
            f"[1/1] run1 {TASK_ID}: OK | score=ERR | "
            "scoring_error=Error code: 502 shim upstream failed"
        ),
        exhausted=True,
    )
    record_path.chmod(0o666)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["forbidden_free_text"] = "synthetic user text"
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    _write_json(record_path, record)

    with pytest.raises(ResumeProtocolError, match="unknown or missing fields"):
        plan_resume(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_index=1,
            run_manifest_path=manifest,
            task_ids=[TASK_ID],
        )


def _write_synthetic_official_batch(
    arm_root: Path,
    *,
    exhausted: bool = False,
) -> tuple[Path, tuple[Path, ...]]:
    manifest = _manifest(arm_root)
    pre_snapshots = {
        run: snapshot_run(arm_root / DOMAIN / f"run{run}", [TASK_ID])
        for run in range(1, 6)
    }
    for run in range(1, 6):
        _write_json(
            arm_root / DOMAIN / f"run{run}" / f"{TASK_ID}.json",
            _scored_trajectory(completion=1),
        )
    log = arm_root / "_batch_logs" / "official-runs1-5.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("# Runs 1..5 together\nfresh completed\n", encoding="utf-8")
    ledger = arm_root / "_transport" / "relay.jsonl"
    relay_start = _ledger(ledger, exhausted=exhausted)
    records = write_official_fresh_batch_records(
        arm_root=arm_root,
        domain=DOMAIN,
        run_manifest_path=manifest,
        all_task_ids=[TASK_ID],
        pre_snapshots=pre_snapshots,
        log_path=log,
        relay_ledger_path=ledger,
        relay_start_offset=relay_start,
        process_exit_code=0,
        split="test",
        num_runs=5,
        num_runs_idx_start=1,
    )
    return manifest, records


def test_one_official_batch_projects_into_five_verified_run_chains(
    tmp_path: Path,
) -> None:
    manifest, records = _write_synthetic_official_batch(tmp_path)

    assert len(records) == 5
    batch_references = []
    for run in range(1, 6):
        chain = verify_session_chain(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_index=run,
            run_manifest_path=manifest,
        )
        assert len(chain.records) == 1
        batch_references.append(chain.records[0]["fresh_batch"])
        plan = plan_resume(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_index=run,
            run_manifest_path=manifest,
            task_ids=[TASK_ID],
        )
        assert plan["scored_task_ids"] == [TASK_ID]
        assert not plan["agent_task_ids"]
        assert not plan["score_task_ids"]
        assert not plan["rejected_task_ids"]
    assert len(
        {
            (item["relative_path"], item["sha256"], item["batch_id"])
            for item in batch_references
        }
    ) == 1
    assert {item["member_run_index"] for item in batch_references} == set(range(1, 6))


def test_incomplete_official_batch_projection_fails_closed(tmp_path: Path) -> None:
    manifest, records = _write_synthetic_official_batch(tmp_path)
    records[-1].chmod(0o666)
    records[-1].unlink()

    with pytest.raises(ResumeProtocolError, match="projection set is incomplete"):
        plan_resume(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_index=1,
            run_manifest_path=manifest,
            task_ids=[TASK_ID],
        )


def test_official_batch_rejects_nonofficial_run_contract(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path)
    pre_snapshots = {
        run: snapshot_run(tmp_path / DOMAIN / f"run{run}", [TASK_ID])
        for run in range(1, 6)
    }
    log = tmp_path / "_batch_logs" / "bad-contract.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("not executed\n", encoding="utf-8")
    ledger = tmp_path / "_transport" / "relay.jsonl"
    relay_start = _ledger(ledger, exhausted=False)

    with pytest.raises(ResumeProtocolError, match="--num-runs 5"):
        write_official_fresh_batch_records(
            arm_root=tmp_path,
            domain=DOMAIN,
            run_manifest_path=manifest,
            all_task_ids=[TASK_ID],
            pre_snapshots=pre_snapshots,
            log_path=log,
            relay_ledger_path=ledger,
            relay_start_offset=relay_start,
            process_exit_code=0,
            split="test",
            num_runs=1,
            num_runs_idx_start=1,
        )


def test_runner_uses_one_official_five_run_batch_and_single_run_resume() -> None:
    runner = (Path(__file__).parents[1] / "scripts" / "run_selective_pwm.ps1").read_text(
        encoding="utf-8"
    )
    assert runner.count('"--num-runs", "5"') == 1
    assert runner.count('"--num-runs-idx-start", "1"') == 1
    assert '"--split", "test"' in runner
    assert "$ResumeHelper record-official-batch" in runner
    assert '"--num-runs", "1"' in runner
    assert '"state_bench.scripts.score"' in runner
    assert '"_resume_tmp\\$scoreSessionId\\$domain"' in runner
    assert 'foreach ($agentTaskId in $agentTaskIds)' in runner
    assert '"--tasks", $agentTaskId' in runner
    assert '"--tasks", ($agentTaskIds -join ",")' not in runner
    assert '"--tasks", ($scoreTaskIds -join ",")' not in runner
    assert 'artifact_preflight = Join-Path $repoRoot "scripts\\preflight_training_artifacts.py"' in runner
    assert "Training-artifact provenance preflight failed closed before relay/API startup" in runner
    assert 'openai_sdk_version = "2.16.0"' in runner
    assert "openai_sdk_default_max_retries = 2" in runner
    assert "benchmark_tenacity_max_attempts = 5" in runner
    assert "OPENAI_MAX_RETRIES" not in runner


def test_runner_freezes_python_import_roots_and_checks_module_origins() -> None:
    runner = (Path(__file__).parents[1] / "scripts" / "run_selective_pwm.ps1").read_text(
        encoding="utf-8"
    )

    frozen_path = '$env:PYTHONPATH = (($repoRoot, $stateBench) -join [IO.Path]::PathSeparator)'
    assert frozen_path in runner
    assert "$env:PYTHONPATH)" not in runner
    assert '$env:PYTHONNOUSERSITE = "1"' in runner
    assert '$env:PYTHONSAFEPATH = "1"' in runner
    assert "Assert-NoRootPythonShadows" in runner
    assert '"sitecustomize.py", "usercustomize.py", "openai.py"' in runner
    assert "Assert-PythonModuleOrigins" in runner
    assert '"state_bench": benchmark / "state_bench"' in runner
    assert '"agents": repository / "agents"' in runner
    assert '"clients": repository / "clients"' in runner
    assert '"tools.eval_shim": repository / "tools"' in runner
    assert runner.index(frozen_path) < runner.index(
        "Training-artifact provenance preflight failed closed before relay/API startup"
    )


def test_runner_and_contract_lock_official_workers_to_two() -> None:
    root = Path(__file__).parents[1]
    runner = (root / "scripts" / "run_selective_pwm.ps1").read_text(encoding="utf-8")
    evaluator = (root / "scripts" / "evaluate_gate.py").read_text(encoding="utf-8")

    assert '$Stage -eq "official750" -and $Workers -ne 2' in runner
    assert 'if gate_name == "official750"' in evaluator
    assert 'expected_workers = {"minimum": 1, "maximum": 3}' in evaluator
