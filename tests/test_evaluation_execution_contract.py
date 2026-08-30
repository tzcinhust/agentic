from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_gate import (  # noqa: E402
    Dataset,
    EvaluationInputError,
    Observation,
    SPLIT_SEED,
    canonical_sha256,
    evaluate_gate,
    evaluate_execution_contract,
    file_sha256,
    load_dataset,
    load_train_inventory_evidence,
    main as evaluate_main,
    task_ids_sha256,
    train_content_manifest_sha256,
)
from resume_protocol import snapshot_run, write_session_record  # noqa: E402


DOMAINS = ("shopping_assistant", "travel", "customer_support")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _git(*args: str, root: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _rewrite_manifest(path: Path, mutate) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(path, manifest)


def _fixture(tmp_path: Path) -> dict[str, Any]:
    state_bench = tmp_path / "STATE-Bench"
    state_bench.mkdir()
    (state_bench / "pyproject.toml").write_text(
        '[project]\nname = "state-bench"\nversion = "0.8.1"\n', encoding="utf-8"
    )

    simulator_hashes: dict[str, str] = {}
    judge_hashes: dict[str, str] = {}
    normalized_simulator: dict[str, str] = {}
    normalized_judge: dict[str, dict[str, str]] = {}
    task_ids: dict[str, list[str]] = {}
    train_ids: dict[str, list[str]] = {}
    for domain in DOMAINS:
        simulator_path = (
            state_bench / "state_bench" / "domains" / domain / "prompts" / "user_sim_base.md"
        )
        judge_path = (
            state_bench
            / "state_bench"
            / "domains"
            / domain
            / "prompts"
            / "judge_task_requirements.md"
        )
        simulator_path.parent.mkdir(parents=True, exist_ok=True)
        simulator_path.write_text(f"simulator {domain}\n", encoding="utf-8")
        judge_path.write_text(f"judge {domain}\n", encoding="utf-8")
        simulator_hash = file_sha256(simulator_path)
        judge_hash = file_sha256(judge_path)
        simulator_hashes[f"{domain}/user_sim_base.md"] = simulator_hash
        judge_hashes[f"{domain}/judge_task_requirements.md"] = judge_hash
        normalized_simulator[domain] = simulator_hash
        normalized_judge[domain] = {"judge_task_requirements.md": judge_hash}
        task_ids[domain] = [f"task-{domain}"]
        train_ids[domain] = [f"train-{domain}-{index:03d}" for index in range(100)]
        for train_task_id in train_ids[domain]:
            _write_json(
                state_bench
                / "datasets"
                / "train_task_trajectories"
                / domain
                / f"{train_task_id}.json",
                {"task_id": train_task_id, "domain": domain},
            )
        _write_json(
            state_bench
            / "state_bench"
            / "domains"
            / domain
            / "splits"
            / "train_test.json",
            {"splits": {"train": train_ids[domain], "test": task_ids[domain]}},
        )

    protocol = {
        "split_version": "train_test",
        "split": "test",
        "num_runs": 5,
        "domains": list(DOMAINS),
        "official_model": "gpt-5.4",
        "simulator": {"model": "gpt-5.4", "prompt_hashes": simulator_hashes},
        "judge": {
            "model": "gpt-5.4",
            "reasoning_effort": "high",
            "prompt_hashes": judge_hashes,
        },
    }
    protocol_path = state_bench / "state_bench" / "configs" / "eval_protocols" / "gpt54.json"
    _write_json(protocol_path, protocol)
    _git("init", root=state_bench)
    _git("add", ".", root=state_bench)
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
        root=state_bench,
    )
    commit = _git("rev-parse", "HEAD", root=state_bench)
    train_inventory = load_train_inventory_evidence(state_bench, DOMAINS)
    full_manifest_sha256, full_file_count = train_content_manifest_sha256(
        train_inventory["files"], {domain: train_ids[domain] for domain in DOMAINS}
    )
    assert full_file_count == 300
    dev_ids = {domain: train_ids[domain][:10] for domain in DOMAINS}
    independent_splits: dict[str, dict[str, list[str]]] = {}
    for domain in DOMAINS:
        ordered = sorted(
            set(train_ids[domain]) - set(dev_ids[domain]),
            key=lambda task_id: (
                hashlib.sha256(
                    f"{SPLIT_SEED}|{domain}|{task_id}".encode()
                ).hexdigest(),
                task_id,
            ),
        )
        independent_splits[domain] = {
            "dev": dev_ids[domain],
            "lockbox": ordered[:10],
            "optimizer": ordered[10:],
        }

    repository = tmp_path / "selective-pwm"
    memory = repository / "artifacts" / "memory" / "process_workflows.json"
    router = repository / "artifacts" / "memory" / "workflow_router_v2.json"
    runner = repository / "scripts" / "run_selective_pwm.ps1"
    memory.parent.mkdir(parents=True, exist_ok=True)
    implementation_relatives = {
        "runner": "scripts/run_selective_pwm.ps1",
        "risk_aware_agent": "agents/risk_aware_process_workflow_memory_agent.py",
        "parent_agent": "agents/process_workflow_memory_agent.py",
        "actor_agent": "agents/opencode_agent.py",
        "agent_client": "clients/opencode_client.py",
        "relay": "tools/eval_shim.py",
        "resume_protocol": "scripts/resume_protocol.py",
        "artifact_preflight": "scripts/preflight_training_artifacts.py",
        "gate_evaluator": "scripts/evaluate_gate.py",
        "official_validator": "scripts/validate_official_submission.py",
        "billing_reconciler": "scripts/reconcile_novacode_billing.py",
        "v1_builder": "scripts/build_process_workflows.py",
        "router_builder": "scripts/build_workflow_router_v2.py",
        "optimizer_builder": "scripts/build_optimizer80_artifacts.ps1",
        "gate_config": "configs/evaluation_gates.json",
        "split_manifest": "configs/workflow_router_dev_ids.json",
        "router_schema": "docs/workflow_router_v2.schema.json",
        "billing_schema": "docs/novacode_billing_evidence.schema.json",
    }
    for name, relative in implementation_relatives.items():
        path = repository / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if name == "split_manifest":
            _write_json(
                path,
                {
                    "seed": "unit-test",
                    "method": "fixed-dev-plus-hash-lockbox",
                    **dev_ids,
                },
            )
        else:
            path.write_text(f"# frozen {name}\n", encoding="utf-8")
    dev_manifest_sha256 = file_sha256(repository / implementation_relatives["split_manifest"])
    excluded_sources = [
        "requirements",
        "task_definitions",
        "test_environments",
        "test_judge_reasoning",
        "test_outputs",
        "test_tasks",
    ]
    _write_json(
        memory,
        {
            "cards": [],
            "stats": {domain: {"trajectories": 100} for domain in DOMAINS},
            "provenance": {
                "task_split": "all",
                "task_manifest_sha256": None,
                "task_manifest_method": "all_fixed_train_trajectories",
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "inventory_file_count": 300,
                "trajectory_manifest_sha256": full_manifest_sha256,
                "read_content_manifest_sha256": full_manifest_sha256,
                "learning_input": "datasets/train_task_trajectories only",
                "excluded_sources": excluded_sources,
                "selected_counts": {domain: 100 for domain in DOMAINS},
                "split_summary": {
                    domain: {"all": train_ids[domain]} for domain in DOMAINS
                },
            },
        },
    )
    _write_json(
        router,
        {
            "source_memory_sha256": file_sha256(memory),
            "domain_configs": {
                "shopping_assistant": {"promoted": True},
                "travel": {"promoted": False},
                "customer_support": {"promoted": False},
            },
            "splits": independent_splits,
            "provenance": {
                "memory_training_split": "all",
                "lockbox_independent": False,
                "dev_manifest_sha256": dev_manifest_sha256,
                "train_file_count": 300,
                "train_manifest_sha256": full_manifest_sha256,
                "read_content_manifest_sha256": full_manifest_sha256,
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "memory_trajectory_counts": {domain: 100 for domain in DOMAINS},
                "source_task_overlap": {
                    domain: {"dev": [], "lockbox": []} for domain in DOMAINS
                },
                "excluded_sources": excluded_sources,
            },
        },
    )
    runner.write_text("# locked runner\n", encoding="utf-8")
    _git("init", root=repository)
    _git("add", ".", root=repository)
    _git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "fixture",
        root=repository,
    )
    repository_commit = _git("rev-parse", "HEAD", root=repository)
    implementation_hashes = {
        name: file_sha256(repository / relative)
        for name, relative in implementation_relatives.items()
    }

    agent_client_contract = {
        "max_tokens": 4096,
        "timeout_seconds": 120.0,
        "max_retries": 0,
        "temperature": 0.0,
    }
    official_evaluation_client_contract = {
        "openai_sdk_version": "2.16.0",
        "openai_sdk_default_max_retries": 2,
        "benchmark_tenacity_max_attempts": 5,
        "configuration_source": "pinned_state_bench_v0.8.1",
        "all_requests_via_attributable_relay": True,
        "scored_trajectory_resampling": False,
    }
    config = {
        "benchmark_contract": {
            "benchmark_version": "0.8.1",
            "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
            "state_bench_commit": commit,
            "split_version": "train_test",
            "protocol_config": "state_bench/configs/eval_protocols/gpt54.json",
            "agent_model": "gpt-5.4",
            "simulator_model": "gpt-5.4",
            "judge_model": "gpt-5.4",
            "judge_reasoning_effort": "high",
            "agent_client_contract": agent_client_contract,
            "official_evaluation_client_contract": official_evaluation_client_contract,
            "split": "test",
            "num_runs": 5,
            "retrieve_learnings_top_k": 3,
            "ignore_missing": False,
            "domains": list(DOMAINS),
            "prompt_hashes": {
                "simulator": normalized_simulator,
                "judge": normalized_judge,
            },
        },
        "run_manifest_contract": {
            "schema_version": "1.0.0",
            "created_by": "scripts/run_selective_pwm.ps1",
        },
        "artifact_policy": {
            "optimizer80_stages": ["dev", "lockbox"],
            "optimizer80_memory": str(tmp_path / "unused-memory.json"),
            "optimizer80_router": str(tmp_path / "unused-router.json"),
            "full100_stages": ["paired150", "official750"],
            "full100_memory": str(memory),
            "full100_router": str(router),
        },
        "router_policy": {
            "required_promoted_domains": ["shopping_assistant"],
            "allowed_baseline_fallback_domains": ["travel", "customer_support"],
        },
        "gates": {
            "paired150": {
                "runs": 1,
                "tasks_per_domain": 1,
                "paired": True,
                "allowed_candidate_router_stages": ["A", "B", "C"],
            }
        },
    }

    candidate_root = tmp_path / "candidate-C"
    baseline_root = tmp_path / "baseline"
    observations: dict[str, list[Observation]] = {"candidate": [], "baseline": []}
    for arm, root in (("candidate", candidate_root), ("baseline", baseline_root)):
        relay_session_id = ("a" if arm == "candidate" else "b") * 32
        origin_sha256 = "c" * 64
        transport_dir = root / "_transport"
        transport_dir.mkdir(parents=True, exist_ok=True)
        relay_relative = f"_transport/relay-{relay_session_id}"
        (root / f"{relay_relative}.log").write_text("locked relay log\n", encoding="utf-8")
        ledger_path = root / f"{relay_relative}.jsonl"
        ledger_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0.0",
                    "event": "session_start",
                    "provider": "novacode",
                    "upstream_origin_sha256": origin_sha256,
                    "rpm": 45,
                    "burst": 5,
                    "burst_window_seconds": 1.0,
                    "attempts": 5,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        for domain in DOMAINS:
            raw: dict[str, Any] = {
                "task_id": task_ids[domain][0],
                "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
                "scoring_protocol_id": "state_bench_v0.8.1_gpt54",
                "simulator_model": "gpt-5.4",
                "judge_model": "gpt-5.4",
                "judge_reasoning_effort": "high",
                "simulator_prompt_hash": normalized_simulator[domain],
                "judge_prompt_hashes": normalized_judge[domain],
                "agent_model": {"model_name": "gpt-5.4", "reasoning_level": None},
                "agent_name": "RiskAwareProcessWorkflowMemoryAgent"
                if arm == "candidate"
                else "ProcessWorkflowMemoryAgent",
                "task_completion_pass": 1,
                "state_requirements_met": 1,
                "task_requirements_met": 1,
                "ux_score": 4.0,
                "token_usage": {"input_tokens": 100},
            }
            if arm == "candidate" and domain == "shopping_assistant":
                raw["workflow_router"] = {
                    "mode": "enforce",
                    "stage": "C",
                    "router_enabled": True,
                }
            trajectory_path = root / domain / "run1" / f"{task_ids[domain][0]}.json"
            observations[arm].append(
                Observation(
                    domain=domain,
                    run=1,
                    task_id=task_ids[domain][0],
                    completion=1,
                    state=1,
                    task=1,
                    ux=4.0,
                    input_tokens=100,
                    path=trajectory_path,
                    raw=raw,
                )
            )
            manifest = {
                "schema_version": "1.0.0",
                "created_by": "scripts/run_selective_pwm.ps1",
                "stage": "paired150",
                "arm": arm,
                "router_stage": "C" if arm == "candidate" else None,
                "domain": domain,
                "protocol": {
                    "benchmark_version": "0.8.1",
                    "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
                    "split_version": "train_test",
                    "official_split": "test",
                    "official_num_runs": 5,
                    "agent_model": "gpt-5.4",
                    "simulator_model": "gpt-5.4",
                    "judge_model": "gpt-5.4",
                    "judge_reasoning_effort": "high",
                    "protocol_config_sha256": file_sha256(protocol_path),
                    "prompt_hashes": {
                        "simulator": simulator_hashes,
                        "judge": judge_hashes,
                    },
                    "evaluation_deployments": {
                        "STATE_BENCH_EVAL_DEPLOYMENTS": ["gpt-5.4"],
                    },
                },
                "run": {
                    "num_runs": 1,
                    "run_start": 1,
                    "workers": 2,
                    "retry_attempts": 1,
                    "retrieve_learnings_top_k": 3,
                    "ignore_missing_runs": False,
                    "agent_class": raw["agent_name"],
                    "agent_client_class": "OpenCodeLLMClient",
                    "memory_mode": "hybrid",
                    "agent_client_contract": agent_client_contract,
                    "official_evaluation_client_contract": official_evaluation_client_contract,
                    "task_selection": {
                        "mode": "split",
                        "source": "state_bench_official_test_split",
                        "split": "test",
                        "task_ids": task_ids[domain],
                        "task_ids_sha256": task_ids_sha256(task_ids[domain]),
                    },
                },
                "artifacts": {
                    "artifact_kind": "full100",
                    "memory_sha256": file_sha256(memory),
                    "router_sha256": file_sha256(router),
                    "runner_sha256": file_sha256(runner),
                    "repository_commit": repository_commit,
                    "repository_tracked_tree_clean": True,
                    "implementation_sha256": implementation_hashes,
                    "state_bench_commit": commit,
                    "state_bench_version": "0.8.1",
                    "state_bench_tracked_tree_clean": True,
                    "state_bench_protocol_sha256": file_sha256(protocol_path),
                    "state_bench_split_manifest_sha256": file_sha256(
                        state_bench
                        / "state_bench"
                        / "domains"
                        / domain
                        / "splits"
                        / "train_test.json"
                    ),
                },
                "transport": {
                    "provider": "novacode",
                    "upstream_origin_sha256": origin_sha256,
                    "relay_session_id": relay_session_id,
                    "relay_sha256": implementation_hashes["relay"],
                    "rpm": 45,
                    "burst": 5,
                    "burst_window_seconds": 1.0,
                    "attempts": 5,
                    "only_transport_retries": True,
                    "ledger_relative_path": f"{relay_relative}.jsonl",
                    "log_relative_path": f"{relay_relative}.log",
                },
            }
            manifest["manifest_sha256"] = canonical_sha256(manifest)
            manifest_path = root / domain / "run_manifest.json"
            _write_json(manifest_path, manifest)
            pre_snapshot = snapshot_run(trajectory_path.parent, task_ids[domain])
            _write_json(trajectory_path, raw)
            session_log = root / "_sessions" / domain / "run1" / "fresh.log"
            session_log.parent.mkdir(parents=True, exist_ok=True)
            session_log.write_text("fresh session completed\n", encoding="utf-8")
            write_session_record(
                arm_root=root,
                domain=domain,
                run_index=1,
                run_manifest_path=manifest_path,
                mode="fresh",
                all_task_ids=task_ids[domain],
                target_task_ids=task_ids[domain],
                pre_snapshot=pre_snapshot,
                log_path=session_log,
                relay_ledger_path=ledger_path,
                relay_start_offset=ledger_path.stat().st_size,
                process_exit_code=0,
            )
            _write_json(
                root / domain / "metrics.json",
                {
                    "benchmark_version": "0.8.1",
                    "evaluation_protocol_id": "state_bench_v0.8.1_gpt54",
                    "num_runs": 1,
                    "agent_model": {"model_name": "gpt-5.4", "reasoning_level": None},
                    "metrics": {
                        "task_completion_pass@1": 1.0,
                        "task_completion_pass^1": 1.0,
                        "mean_ux_score": 4.0,
                    },
                },
            )

    return {
        "config": config,
        "candidate": Dataset(candidate_root, tuple(observations["candidate"])),
        "baseline": Dataset(baseline_root, tuple(observations["baseline"])),
        "memory": memory,
        "router": router,
        "runner": runner,
        "state_bench": state_bench,
    }


def _evaluate(fixture: dict[str, Any]):
    return evaluate_execution_contract(
        gate_name="paired150",
        config=fixture["config"],
        candidate=fixture["candidate"],
        baseline=fixture["baseline"],
        memory_path=fixture["memory"],
        router_path=fixture["router"],
        runner_path=fixture["runner"],
        state_bench_root=fixture["state_bench"],
    )


def test_execution_contract_accepts_stage_c_and_expected_fallback(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    checks, stage = _evaluate(fixture)
    assert stage == "C"
    assert all(check.passed for check in checks), [check for check in checks if not check.passed]


def test_execution_contract_rejects_manifest_task_id_tampering(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["candidate"].root / "travel" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["run"]["task_selection"]["task_ids"] = ["not-the-official-task"]
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(path, manifest)
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["candidate.travel.manifest.task_ids"].passed


def test_execution_contract_scans_numbered_deployments(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["candidate"].root / "customer_support" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["protocol"]["evaluation_deployments"]["STATE_BENCH_EVAL_DEPLOYMENTS_2"] = [
        "gpt-5.4-mini"
    ]
    manifest["manifest_sha256"] = canonical_sha256(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    _write_json(path, manifest)
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["candidate.customer_support.manifest.gpt54_deployments_only"].passed


def test_execution_contract_rejects_changed_full100_train_content(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    train_path = next(
        (
            fixture["state_bench"]
            / "datasets"
            / "train_task_trajectories"
            / "shopping_assistant"
        ).glob("*.json")
    )
    trajectory = json.loads(train_path.read_text(encoding="utf-8"))
    trajectory["post_training_mutation"] = True
    _write_json(train_path, trajectory)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["full100_memory.provenance"].passed
    assert not by_name["full100_router.provenance"].passed


def test_execution_contract_rejects_manifest_agent_client_contract_drift(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    path = fixture["candidate"].root / "travel" / "run_manifest.json"

    def mutate(manifest: dict[str, Any]) -> None:
        manifest["run"]["agent_client_contract"]["max_tokens"] = 2048

    _rewrite_manifest(path, mutate)
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["candidate.travel.manifest.agent_client_contract"].passed
    assert not by_name["candidate.single_agent_client_contract"].passed


def test_execution_contract_rejects_different_origins_across_paired_arms(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    different_origin = "d" * 64
    for domain in DOMAINS:
        path = fixture["baseline"].root / domain / "run_manifest.json"

        def mutate(manifest: dict[str, Any]) -> None:
            manifest["transport"]["upstream_origin_sha256"] = different_origin

        _rewrite_manifest(path, mutate)

    ledger_path = next((fixture["baseline"].root / "_transport").glob("*.jsonl"))
    ledger_header = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    ledger_header["upstream_origin_sha256"] = different_origin
    ledger_path.write_text(
        json.dumps(ledger_header, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert by_name["candidate.single_upstream_origin"].passed
    assert by_name["baseline.single_upstream_origin"].passed
    assert not by_name["paired.same_upstream_origin"].passed


def test_execution_contract_rejects_different_client_contracts_across_paired_arms(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for domain in DOMAINS:
        path = fixture["baseline"].root / domain / "run_manifest.json"

        def mutate(manifest: dict[str, Any]) -> None:
            manifest["run"]["agent_client_contract"]["timeout_seconds"] = 60.0

        _rewrite_manifest(path, mutate)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert by_name["candidate.single_agent_client_contract"].passed
    assert by_name["baseline.single_agent_client_contract"].passed
    assert not by_name["paired.same_agent_client_contract"].passed


def test_execution_contract_records_pinned_official_evaluation_retry_layers(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    for domain in DOMAINS:
        path = fixture["baseline"].root / domain / "run_manifest.json"

        def mutate(manifest: dict[str, Any]) -> None:
            manifest["run"]["official_evaluation_client_contract"][
                "openai_sdk_default_max_retries"
            ] = 0

        _rewrite_manifest(path, mutate)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert by_name[
        "candidate.shopping_assistant.manifest.official_evaluation_client_contract"
    ].passed
    assert not by_name[
        "baseline.shopping_assistant.manifest.official_evaluation_client_contract"
    ].passed
    assert not by_name["paired.same_official_evaluation_client_contract"].passed


def test_execution_contract_rejects_wrong_ledger_burst_window(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    ledger_path = next((fixture["candidate"].root / "_transport").glob("*.jsonl"))
    ledger_header = json.loads(ledger_path.read_text(encoding="utf-8").splitlines()[0])
    ledger_header["burst_window_seconds"] = 2.0
    ledger_path.write_text(
        json.dumps(ledger_header, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name[
        "candidate.shopping_assistant.manifest.transport_files"
    ].passed


def test_execution_contract_locks_official_workers_to_exactly_two(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    fixture["config"]["gates"]["official750"] = {
        "runs": 5,
        "tasks_per_domain": 1,
        "paired": False,
        "allowed_candidate_router_stages": ["C"],
    }
    for domain in DOMAINS:
        path = fixture["candidate"].root / domain / "run_manifest.json"

        def mutate(manifest: dict[str, Any]) -> None:
            manifest["stage"] = "official750"
            manifest["run"]["num_runs"] = 5
            manifest["run"]["workers"] = 3

        _rewrite_manifest(path, mutate)

    checks, _ = evaluate_execution_contract(
        gate_name="official750",
        config=fixture["config"],
        candidate=fixture["candidate"],
        baseline=None,
        memory_path=fixture["memory"],
        router_path=fixture["router"],
        runner_path=fixture["runner"],
        state_bench_root=fixture["state_bench"],
    )
    by_name = {check.name: check for check in checks}
    for domain in DOMAINS:
        worker_check = by_name[f"candidate.{domain}.manifest.workers"]
        assert not worker_check.passed
        assert worker_check.expected == {"equal": 2}


@pytest.mark.parametrize("workers", [1, 2, 3])
def test_execution_contract_keeps_nonofficial_worker_range(
    tmp_path: Path, workers: int
) -> None:
    fixture = _fixture(tmp_path)
    for domain in DOMAINS:
        path = fixture["candidate"].root / domain / "run_manifest.json"

        def mutate(manifest: dict[str, Any]) -> None:
            manifest["run"]["workers"] = workers

        _rewrite_manifest(path, mutate)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    for domain in DOMAINS:
        worker_check = by_name[f"candidate.{domain}.manifest.workers"]
        assert worker_check.passed
        assert worker_check.expected == {"minimum": 1, "maximum": 3}


@pytest.mark.parametrize("ux_score", [0.99, 5.01])
def test_load_dataset_rejects_ux_outside_official_range(
    tmp_path: Path, ux_score: float
) -> None:
    result_root = tmp_path / "results"
    _write_json(
        result_root / "shopping_assistant" / "run1" / "task-1.json",
        {
            "task_id": "task-1",
            "domain": "shopping_assistant",
            "task_completion_pass": 1,
            "state_requirements_met": 1,
            "task_requirements_met": 1,
            "ux_score": ux_score,
            "token_usage": {"input_tokens": 1},
        },
    )

    with pytest.raises(EvaluationInputError, match=r"official \[1, 5\] range"):
        load_dataset(result_root, DOMAINS)


def test_load_dataset_ignores_interrupted_resume_staging_tree(tmp_path: Path) -> None:
    result_root = tmp_path / "results"
    trajectory = {
        "task_id": "task-1",
        "domain": "shopping_assistant",
        "run_index": 1,
        "task_completion_pass": 1,
        "state_requirements_met": 1,
        "task_requirements_met": 1,
        "ux_score": 4.0,
        "token_usage": {"input_tokens": 1},
    }
    _write_json(
        result_root / "shopping_assistant" / "run1" / "task-1.json",
        trajectory,
    )
    _write_json(
        result_root
        / "_resume_tmp"
        / "interrupted"
        / "shopping_assistant"
        / "run1"
        / "task-1.json",
        trajectory,
    )

    dataset = load_dataset(result_root, DOMAINS)

    assert len(dataset.observations) == 1
    assert dataset.observations[0].path.parent.name == "run1"


@pytest.mark.parametrize(
    "override",
    [
        {"domain": "travel"},
        {"run_index": 2},
        {"run_idx": "1"},
    ],
)
def test_load_dataset_rejects_declared_path_identity_mismatch(
    tmp_path: Path, override: dict[str, object]
) -> None:
    result_root = tmp_path / "results"
    trajectory: dict[str, object] = {
        "task_id": "task-1",
        "domain": "shopping_assistant",
        "run_index": 1,
        "task_completion_pass": 1,
        "state_requirements_met": 1,
        "task_requirements_met": 1,
        "ux_score": 4.0,
        "token_usage": {"input_tokens": 1},
    }
    trajectory.update(override)
    _write_json(
        result_root / "shopping_assistant" / "run1" / "task-1.json",
        trajectory,
    )

    with pytest.raises(EvaluationInputError, match="disagrees with path"):
        load_dataset(result_root, DOMAINS)


def test_evaluator_cli_rejects_non_frozen_config(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    alternate_config = tmp_path / "alternate-gates.json"
    _write_json(alternate_config, {})

    exit_code = evaluate_main(
        [
            "--gate",
            "dev",
            "--candidate",
            str(tmp_path / "unused-results"),
            "--config",
            str(alternate_config),
        ]
    )

    assert exit_code == 2
    assert "promotion thresholds are frozen" in capsys.readouterr().err


def test_execution_contract_requires_fallback_router_metadata_absent(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    observations = []
    for item in fixture["candidate"].observations:
        if item.domain == "travel":
            raw = dict(item.raw)
            raw["workflow_router"] = {
                "mode": "enforce",
                "stage": "C",
                "router_enabled": False,
            }
            item = replace(item, raw=raw)
        observations.append(item)
    fixture["candidate"] = Dataset(fixture["candidate"].root, tuple(observations))
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["candidate.travel.router_runtime"].passed


def test_execution_contract_requires_a_fresh_session_record(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    record_path = next(
        (
            fixture["candidate"].root
            / "_sessions"
            / "shopping_assistant"
            / "run1"
        ).glob("*.json")
    )
    record_path.chmod(0o666)
    record_path.unlink()

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name[
        "candidate.shopping_assistant.auditable_session_chain"
    ].passed


def test_execution_contract_rejects_trajectory_changed_after_session(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    trajectory_path = next(
        item.path
        for item in fixture["candidate"].observations
        if item.domain == "travel"
    )
    trajectory_path.chmod(0o666)
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    trajectory["changed_after_session"] = True
    _write_json(trajectory_path, trajectory)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["candidate.travel.auditable_session_chain"].passed


def test_execution_contract_combines_session_and_manifest_origins(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    domain = "customer_support"
    session_dir = fixture["candidate"].root / "_sessions" / domain / "run1"
    record_path = next(session_dir.glob("*.json"))
    record_path.chmod(0o666)
    record = json.loads(record_path.read_text(encoding="utf-8"))

    different_origin = "d" * 64
    session_header = dict(record["relay_session"])
    session_header["upstream_origin_sha256"] = different_origin
    ledger_bytes = (
        json.dumps(session_header, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    session_ledger = session_dir / "different-origin.jsonl"
    session_ledger.write_bytes(ledger_bytes)
    record["relay_session"] = session_header
    record["relay_segment"] = {
        "relative_path": session_ledger.relative_to(
            fixture["candidate"].root
        ).as_posix(),
        "start_offset": len(ledger_bytes),
        "end_offset": len(ledger_bytes),
        "prefix_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "sha256": hashlib.sha256(b"").hexdigest(),
        "exhausted_request_counts_by_route": {},
    }
    record["record_sha256"] = canonical_sha256(
        {key: value for key, value in record.items() if key != "record_sha256"}
    )
    _write_json(record_path, record)

    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert by_name[f"candidate.{domain}.auditable_session_chain"].passed
    assert not by_name["candidate.single_upstream_origin"].passed


def test_execution_contract_ignores_untracked_but_rejects_tracked_changes(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["state_bench"] / "user-notes.txt").write_text("untracked\n", encoding="utf-8")
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert by_name["state_bench.tracked_tree_clean"].passed

    protocol_path = (
        fixture["state_bench"] / "state_bench" / "configs" / "eval_protocols" / "gpt54.json"
    )
    protocol_path.write_text(protocol_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    checks, _ = _evaluate(fixture)
    by_name = {check.name: check for check in checks}
    assert not by_name["state_bench.tracked_tree_clean"].passed


def test_lockbox_bootstrap_is_enforced_independently_per_domain(tmp_path: Path) -> None:
    observations = []
    for domain in DOMAINS:
        for run in (1, 2):
            for index in range(10):
                observations.append(
                    Observation(
                        domain=domain,
                        run=run,
                        task_id=f"task-{index}",
                        completion=1,
                        state=1,
                        task=1,
                        ux=4.0,
                        input_tokens=100,
                        path=tmp_path / domain / f"run{run}" / f"task-{index}.json",
                        raw={},
                    )
                )
    baseline = Dataset(tmp_path / "baseline", tuple(observations))
    candidate = Dataset(tmp_path / "candidate-C", tuple(observations))
    config = {
        "benchmark_contract": {"domains": list(DOMAINS)},
        "gates": {
            "lockbox": {
                "runs": 2,
                "tasks_per_domain": 10,
                "paired": True,
                "cluster_bootstrap": {
                    "by_domain": True,
                    "confidence": 0.9,
                    "iterations": 100,
                    "seed": 20260830,
                    "minimum_lower_bound": 0.0,
                },
            }
        },
    }
    report = evaluate_gate(
        gate_name="lockbox",
        config=config,
        candidate=candidate,
        baseline=baseline,
    )
    checks = {check["name"]: check for check in report["checks"]}
    for domain in DOMAINS:
        name = f"completion.{domain}.cluster_bootstrap_lower_bound"
        assert checks[name]["passed"]
