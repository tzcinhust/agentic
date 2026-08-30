"""Validate an official selective-PWM result tree and its launch artifacts.

This is a local, read-only preflight.  It checks the 750 scored observations,
official score floors, protocol/model provenance, five-repeat completeness,
router integrity, and the checked-in runner's exact launch contract.  It does
not call STATE-Bench or any external API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.evaluate_gate import (
        DEFAULT_CONFIG,
        DEFAULT_STATE_BENCH_ROOT,
        Check,
        EvaluationInputError,
        evaluate_execution_contract,
        evaluate_gate,
        load_dataset,
    )
    from scripts.reconcile_novacode_billing import (
        BillingEvidenceError,
        reconcile_billing,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_gate import (  # type: ignore[no-redef]
        DEFAULT_CONFIG,
        DEFAULT_STATE_BENCH_ROOT,
        Check,
        EvaluationInputError,
        evaluate_execution_contract,
        evaluate_gate,
        load_dataset,
    )
    from reconcile_novacode_billing import (  # type: ignore[no-redef]
        BillingEvidenceError,
        reconcile_billing,
    )


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY = (
    REPO_ROOT / "artifacts" / "statebench_cross_domain_pwm" / "memory" / "process_workflows.json"
)
DEFAULT_ROUTER = (
    REPO_ROOT / "artifacts" / "statebench_cross_domain_pwm" / "memory" / "workflow_router_v2.json"
)
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "run_selective_pwm.ps1"

FORBIDDEN_ORACLE_KEYS = {
    "requirement",
    "requirements",
    "state_requirement",
    "state_requirements",
    "state_requirements_gt",
    "task_requirement",
    "task_requirements",
    "task_requirements_details",
    "task_definition",
    "task_definitions",
    "task_summary",
    "judge_reasoning",
    "test_task",
    "test_task_id",
    "test_result",
    "test_results",
}


def _check(
    checks: list[Check], name: str, passed: bool, actual: Any, expected: Any, detail: str = ""
) -> None:
    checks.append(Check(name=name, passed=passed, actual=actual, expected=expected, detail=detail))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _forbidden_paths(value: Any, prefix: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            path = f"{prefix}.{key}"
            if str(key).lower() in FORBIDDEN_ORACLE_KEYS:
                found.append(path)
            found.extend(_forbidden_paths(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(_forbidden_paths(child, f"{prefix}[{index}]"))
    return found


def _nonempty_text(value: Any) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, Mapping):
        return bool(value) and any(_nonempty_text(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_nonempty_text(item) for item in value)
    return False


def _model_name(value: Any) -> str:
    if isinstance(value, Mapping):
        value = value.get("model_name") or value.get("model")
    return str(value or "")


def _protocol_checks(observations, contract: Mapping[str, Any]) -> list[Check]:
    checks: list[Check] = []
    protocol = str(contract["evaluation_protocol_id"])
    model = str(contract["agent_model"])
    fields = {
        "evaluation_protocol_id": protocol,
        "scoring_protocol_id": protocol,
        "simulator_model": model,
        "judge_model": model,
    }
    for field, expected in fields.items():
        bad = [str(item.path) for item in observations if str(item.raw.get(field, "")) != expected]
        _check(
            checks,
            f"protocol.{field}",
            not bad,
            {"mismatches": len(bad)},
            {"mismatches": 0, "value": expected},
            f"first mismatches: {bad[:3]!r}" if bad else "",
        )
    bad_agent_model = [
        str(item.path) for item in observations if _model_name(item.raw.get("agent_model")) != model
    ]
    _check(
        checks,
        "protocol.agent_model",
        not bad_agent_model,
        {"mismatches": len(bad_agent_model)},
        {"mismatches": 0, "value": model},
        f"first mismatches: {bad_agent_model[:3]!r}" if bad_agent_model else "",
    )
    expected_agent = "RiskAwareProcessWorkflowMemoryAgent"
    bad_agent = [
        str(item.path)
        for item in observations
        if str(item.raw.get("agent_name", "")) != expected_agent
    ]
    _check(
        checks,
        "protocol.agent_class",
        not bad_agent,
        {"mismatches": len(bad_agent)},
        {"mismatches": 0, "value": expected_agent},
        f"first mismatches: {bad_agent[:3]!r}" if bad_agent else "",
    )
    bad_router_stage = [
        str(item.path)
        for item in observations
        if not isinstance(item.raw.get("workflow_router"), Mapping)
        or item.raw["workflow_router"].get("stage") != "C"
    ]
    _check(
        checks,
        "protocol.router_stage",
        not bad_router_stage,
        {"mismatches": len(bad_router_stage)},
        {"mismatches": 0, "value": "C"},
        f"first mismatches: {bad_router_stage[:3]!r}" if bad_router_stage else "",
    )
    return checks


def _aggregate_metadata_checks(
    candidate_root: Path, contract: Mapping[str, Any], domains: Sequence[str]
) -> list[Check]:
    checks: list[Check] = []
    expected = {
        "benchmark_version": str(contract["benchmark_version"]),
        "evaluation_protocol_id": str(contract["evaluation_protocol_id"]),
        "num_runs": int(contract["num_runs"]),
    }
    for domain in domains:
        paths = [
            path
            for path in candidate_root.rglob("metrics.json")
            if domain.lower() in {part.lower() for part in path.parts}
        ]
        _check(
            checks,
            f"aggregate.{domain}.single_metrics_file",
            len(paths) == 1,
            [str(path) for path in paths],
            {"count": 1},
        )
        if len(paths) != 1:
            continue
        try:
            payload = json.loads(paths[0].read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            _check(checks, f"aggregate.{domain}.readable", False, str(exc), "valid JSON")
            continue
        for field, value in expected.items():
            _check(
                checks,
                f"aggregate.{domain}.{field}",
                payload.get(field) == value,
                payload.get(field),
                {"equal": value},
            )
        actual_model = _model_name(payload.get("agent_model"))
        _check(
            checks,
            f"aggregate.{domain}.agent_model",
            actual_model == contract["agent_model"],
            actual_model,
            {"equal": contract["agent_model"]},
        )
    return checks


def _sidecar_checks(
    router_path: Path,
    memory_path: Path,
    domains: Sequence[str],
    router_policy: Mapping[str, Any] | None = None,
) -> list[Check]:
    checks: list[Check] = []
    if not memory_path.is_file():
        return [Check("router.source_memory_exists", False, str(memory_path), "existing file")]
    if not router_path.is_file():
        return [Check("router.exists", False, str(router_path), "existing file")]
    try:
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
        router = json.loads(router_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [Check("router.readable", False, str(exc), "valid UTF-8 JSON")]
    if not isinstance(memory, Mapping) or not isinstance(router, Mapping):
        return [Check("router.object_shape", False, type(router).__name__, "JSON object")]

    _check(
        checks,
        "router.schema_version",
        router.get("schema_version") == "2.0.0",
        router.get("schema_version"),
        {"equal": "2.0.0"},
    )
    actual_hash = _sha256(memory_path)
    provenance = router.get("provenance") if isinstance(router.get("provenance"), Mapping) else {}
    declared_hashes = {
        str(router.get("source_memory_sha256", "")),
        str(provenance.get("memory_sha256", "")),
    }
    _check(
        checks,
        "router.source_memory_hash",
        declared_hashes == {actual_hash},
        sorted(declared_hashes),
        {"equal": [actual_hash]},
    )
    defaults = router.get("defaults")
    defaults_valid = (
        isinstance(defaults, Mapping)
        and isinstance(defaults.get("weights"), Mapping)
        and isinstance(defaults.get("thresholds"), Mapping)
    )
    _check(checks, "router.defaults", defaults_valid, type(defaults).__name__, "weights + thresholds")

    domain_configs = router.get("domain_configs")
    policy = router_policy if isinstance(router_policy, Mapping) else {}
    required_promoted = set(map(str, policy.get("required_promoted_domains", [])))
    allowed_fallback = set(map(str, policy.get("allowed_baseline_fallback_domains", [])))
    actual_promoted: set[str] = set()
    for domain in domains:
        value = domain_configs.get(domain) if isinstance(domain_configs, Mapping) else None
        promoted = value.get("promoted") if isinstance(value, Mapping) else None
        if promoted is True:
            actual_promoted.add(domain)
        valid_domain_policy = isinstance(promoted, bool) and (
            promoted or domain in allowed_fallback
        )
        _check(
            checks,
            f"router.{domain}.promotion_policy",
            valid_domain_policy,
            promoted,
            {"promoted": True, "or_allowed_baseline_fallback": domain in allowed_fallback},
        )
    _check(
        checks,
        "router.required_promoted_domains",
        required_promoted.issubset(actual_promoted),
        sorted(actual_promoted),
        {"contains": sorted(required_promoted)},
    )

    memory_cards = memory.get("cards") if isinstance(memory.get("cards"), list) else []
    base_by_id = {
        str(card.get("id")): card
        for card in memory_cards
        if isinstance(card, Mapping) and card.get("id") is not None
    }
    router_cards = router.get("cards") if isinstance(router.get("cards"), Mapping) else {}
    _check(
        checks,
        "router.card_coverage",
        set(router_cards) == set(base_by_id),
        {
            "missing": sorted(set(base_by_id) - set(router_cards))[:10],
            "extra": sorted(set(router_cards) - set(base_by_id))[:10],
            "actual_count": len(router_cards),
        },
        {"missing": [], "extra": [], "actual_count": len(base_by_id)},
    )
    invalid_cards: list[str] = []
    bad_hashes: list[str] = []
    for card_id, base_card in base_by_id.items():
        sidecar = router_cards.get(card_id)
        if not isinstance(sidecar, Mapping):
            invalid_cards.append(card_id)
            continue
        contract = sidecar.get("contract")
        compiler = sidecar.get("compiler")
        utility = sidecar.get("utility")
        valid_contract = isinstance(contract, Mapping) and _nonempty_text(contract.get("trigger"))
        valid_contract = valid_contract and _nonempty_text(contract.get("scope"))
        if isinstance(contract, Mapping):
            valid_contract = valid_contract and all(
                isinstance(contract.get(field, []), list)
                for field in (
                    "required_reads",
                    "authorized_writes",
                    "verification_rules",
                    "decision_rules",
                    "required_disclosures",
                    "prohibitions",
                )
            )
        valid_shape = (
            valid_contract
            and isinstance(compiler, Mapping)
            and isinstance(compiler.get("valid"), bool)
            and _nonempty_text(sidecar.get("primary_text"))
            and _nonempty_text(sidecar.get("secondary_text"))
            and isinstance(utility, Mapping)
        )
        if not valid_shape:
            invalid_cards.append(card_id)
        declared_card_hash = sidecar.get("source_card_sha256")
        if declared_card_hash != _canonical_sha256(base_card):
            bad_hashes.append(card_id)
    _check(
        checks,
        "router.card_contract_compiler_text_utility",
        not invalid_cards,
        {"invalid_count": len(invalid_cards), "first": invalid_cards[:10]},
        {"invalid_count": 0},
    )
    _check(
        checks,
        "router.source_card_hashes",
        not bad_hashes,
        {"mismatch_count": len(bad_hashes), "first": bad_hashes[:10]},
        {"mismatch_count": 0},
    )
    forbidden = _forbidden_paths(router)
    _check(
        checks,
        "router.no_oracle_fields",
        not forbidden,
        forbidden[:20],
        {"equal": []},
        f"{len(forbidden)} forbidden key(s)" if forbidden else "",
    )
    return checks


def _runner_checks(path: Path) -> list[Check]:
    if not path.is_file():
        return [Check("runner.exists", False, str(path), "existing launch script")]
    text = path.read_text(encoding="utf-8")
    specifications = {
        "runner.stage_official750": r"official750",
        "runner.agent_class": r"RiskAwareProcessWorkflowMemoryAgent",
        "runner.split_test": r'--split["\']?\s*,\s*["\']test',
        "runner.five_runs": r'official750["\']?\)\s*\{\s*\$runs\s*=\s*5',
        "runner.top_k_three": r'--retrieve-learnings-top-k["\']?\s*,\s*["\']3',
        "runner.agent_model_gpt54": r'--agent-model-name["\']?\s*,\s*["\']gpt-5\.4',
        "runner.router_enforce": r"STATE_BENCH_WORKFLOW_ROUTER_MODE\s*=\s*[\"']enforce",
        "runner.router_stage_c": r"STATE_BENCH_WORKFLOW_ROUTER_STAGE\s*=\s*\$RouterStage",
        "runner.official_candidate_c_only": r"official750[^\r\n]+Arm[^\r\n]+candidate[^\r\n]+RouterStage[^\r\n]+C",
        "runner.numbered_deployment_scan": r"STATE_BENCH_EVAL_DEPLOYMENTS\(\?:_\\d\+\)\?",
        "runner.tracked_tree_clean": r"status[^\r\n]+--porcelain=v1[^\r\n]+--untracked-files=no",
        "runner.strict_compute_metrics": r"state_bench\.scripts\.compute_metrics",
        "runner.immutable_manifest": r"Write-RunManifestExclusive",
        "runner.optimizer80_isolation": r"workflow_router_v2_optimizer80\.json",
    }
    checks = []
    for name, pattern in specifications.items():
        matched = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE) is not None
        _check(checks, name, matched, matched, {"equal": True})
    ignore_flags = sorted(set(re.findall(r"--ignore[-_a-z0-9]*", text, flags=re.IGNORECASE)))
    _check(checks, "runner.no_ignore_flags", not ignore_flags, ignore_flags, {"equal": []})
    return checks


def _billing_checks(
    *,
    candidate_root: Path,
    evidence_path: Path,
    domains: Sequence[str],
    expected_runs: int,
    expected_tasks_per_domain: int,
) -> tuple[list[Check], Mapping[str, Any] | None]:
    """Require a positive, NovaCode-issued reconciliation for formal claims."""

    try:
        report = reconcile_billing(
            candidate_root,
            evidence_path,
            domains=domains,
            expected_runs=expected_runs,
            expected_tasks_per_domain=expected_tasks_per_domain,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, BillingEvidenceError) as exc:
        return [
            Check(
                name="billing.novacode_user_supplied_reconciled_evidence",
                passed=False,
                actual=str(exc),
                expected="positive NovaCode rate-card or invoice reconciliation bound to this run",
            )
        ], None
    summary = {
        "report_sha256": report["report_sha256"],
        "evidence_sha256": report["evidence_sha256"],
        "source_document_sha256": report["source_document"]["sha256"],
        "observations": report["token_usage"]["observations"],
        "trajectory_agent_total_tokens": report["token_usage"]["total_tokens"],
        "provider_billable_total_tokens": report["token_usage"][
            "provider_billable_usage"
        ]["total_tokens"],
        "currency": report["cost"]["currency"],
        "reconciled_amount": report["cost"]["reconciled_amount"],
        "mean_per_observation": report["cost"]["mean_per_observation"],
        "method": report["cost"]["method"],
    }
    return [
        Check(
            name="billing.novacode_user_supplied_reconciled_evidence",
            passed=True,
            actual=summary,
            expected={
                "provider": "NovaCode",
                "openai_list_pricing_used": False,
                "cost_greater_than": 0,
                "bound_to_result_and_artifacts": True,
            },
        )
    ], report


def validate(
    *,
    candidate_root: Path,
    config: Mapping[str, Any],
    memory_path: Path,
    router_path: Path,
    runner_path: Path,
    billing_evidence_path: Path,
    state_bench_root: Path = DEFAULT_STATE_BENCH_ROOT,
) -> dict[str, Any]:
    contract = config.get("benchmark_contract")
    if not isinstance(contract, Mapping):
        raise EvaluationInputError("configuration is missing benchmark_contract")
    domains = [str(value) for value in contract.get("domains", [])]
    candidate = load_dataset(candidate_root, domains)
    gate_report = evaluate_gate(
        gate_name="official750", config=config, candidate=candidate, baseline=None
    )
    checks = [
        Check(
            name=f"gate.{item['name']}",
            passed=bool(item["passed"]),
            actual=item["actual"],
            expected=item["expected"],
            detail=str(item.get("detail", "")),
        )
        for item in gate_report["checks"]
    ]
    execution_checks, candidate_stage = evaluate_execution_contract(
        gate_name="official750",
        config=config,
        candidate=candidate,
        baseline=None,
        memory_path=memory_path,
        router_path=router_path,
        state_bench_root=state_bench_root,
        runner_path=runner_path,
    )
    checks.extend(execution_checks)
    checks.extend(_aggregate_metadata_checks(candidate.root, contract, domains))
    router_policy = config.get("router_policy")
    checks.extend(
        _sidecar_checks(
            router_path,
            memory_path,
            domains,
            router_policy if isinstance(router_policy, Mapping) else None,
        )
    )
    checks.extend(_runner_checks(runner_path))
    official_gate = config.get("gates", {}).get("official750", {})
    expected_tasks = (
        int(official_gate.get("tasks_per_domain", 50))
        if isinstance(official_gate, Mapping)
        else 50
    )
    billing_checks, billing_report = _billing_checks(
        candidate_root=candidate.root,
        evidence_path=billing_evidence_path,
        domains=domains,
        expected_runs=int(contract["num_runs"]),
        expected_tasks_per_domain=expected_tasks,
    )
    checks.extend(billing_checks)
    billing_summary = None
    if billing_report is not None:
        billing_summary = {
            "report_sha256": billing_report["report_sha256"],
            "evidence_sha256": billing_report["evidence_sha256"],
            "source_document_sha256": billing_report["source_document"]["sha256"],
            "token_usage": billing_report["token_usage"],
            "cost": billing_report["cost"],
        }
    return {
        "schema_version": "1.0.0",
        "validation": "official750",
        "passed": all(check.passed for check in checks),
        "candidate_root": str(candidate.root),
        "memory_path": str(memory_path.resolve()),
        "router_path": str(router_path.resolve()),
        "runner_path": str(runner_path.resolve()),
        "billing_evidence_path": str(billing_evidence_path.resolve()),
        "state_bench_root": str(state_bench_root.resolve()),
        "candidate_router_stage": candidate_stage,
        "billing_reconciliation": billing_summary,
        "summary": {
            "observations": len(candidate.observations),
            "passed_checks": sum(check.passed for check in checks),
            "failed_checks": sum(not check.passed for check in checks),
        },
        "checks": [check.as_dict() for check in checks],
    }


def _print_report(report: Mapping[str, Any]) -> None:
    print(f"{'PASS' if report['passed'] else 'FAIL'} official750 submission validation")
    print(
        f"observations={report['summary']['observations']} "
        f"checks={report['summary']['passed_checks']} passed/"
        f"{report['summary']['failed_checks']} failed"
    )
    for check in report["checks"]:
        if check["passed"]:
            continue
        detail = f" ({check['detail']})" if check.get("detail") else ""
        print(
            f"[FAIL] {check['name']}: actual={check['actual']!r}, "
            f"expected={check['expected']!r}{detail}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY)
    parser.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument(
        "--billing-evidence",
        type=Path,
        help="NovaCode evidence JSON (defaults to CANDIDATE/billing_evidence.json)",
    )
    parser.add_argument("--state-bench-root", type=Path, default=DEFAULT_STATE_BENCH_ROOT)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.config.resolve(strict=True) != DEFAULT_CONFIG.resolve(strict=True):
            raise EvaluationInputError(
                "official thresholds are frozen; --config must reference "
                f"{DEFAULT_CONFIG}"
            )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        report = validate(
            candidate_root=args.candidate,
            config=config,
            memory_path=args.memory,
            router_path=args.router,
            runner_path=args.runner,
            billing_evidence_path=(
                args.billing_evidence
                if args.billing_evidence is not None
                else args.candidate / "billing_evidence.json"
            ),
            state_bench_root=args.state_bench_root,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, EvaluationInputError) as exc:
        print(f"submission validation input error: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_report(report)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
