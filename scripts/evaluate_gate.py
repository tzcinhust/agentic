"""Evaluate staged selective-PWM promotion gates from scored trajectories.

The evaluator works from trajectory JSON rather than aggregate ``metrics.json``
files so paired deltas, per-domain floors, repeated-run conjunctions, and the
lockbox cluster bootstrap all use the same observations.  It never invokes an
API or a benchmark runner.

Examples::

    python scripts/evaluate_gate.py --gate dev \
        --baseline outputs/selective_pwm/dev/baseline \
        --candidate outputs/selective_pwm/dev/candidate-C

    python scripts/evaluate_gate.py --gate official750 \
        --candidate outputs/selective_pwm/official750/candidate-C --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from scripts.resume_protocol import (
        ResumeProtocolError,
        plan_resume,
        verify_session_chain,
    )
except ModuleNotFoundError:  # direct ``python scripts/evaluate_gate.py`` execution
    from resume_protocol import (  # type: ignore[no-redef]
        ResumeProtocolError,
        plan_resume,
        verify_session_chain,
    )


DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "evaluation_gates.json"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MEMORY = (
    REPO_ROOT / "artifacts" / "statebench_cross_domain_pwm" / "memory" / "process_workflows.json"
)
DEFAULT_ROUTER = (
    REPO_ROOT / "artifacts" / "statebench_cross_domain_pwm" / "memory" / "workflow_router_v2.json"
)
DEFAULT_STATE_BENCH_ROOT = REPO_ROOT.parent / "STATE-Bench"
DEFAULT_RUNNER = REPO_ROOT / "scripts" / "run_selective_pwm.ps1"
SPLIT_SEED = "workflow-router-v2-sha256"
REQUIRED_EXCLUDED_SOURCES = {
    "task_definitions",
    "requirements",
    "test_tasks",
    "test_environments",
    "test_judge_reasoning",
    "test_outputs",
}

IMPLEMENTATION_PATHS = {
    "runner": Path("scripts/run_selective_pwm.ps1"),
    "risk_aware_agent": Path("agents/risk_aware_process_workflow_memory_agent.py"),
    "parent_agent": Path("agents/process_workflow_memory_agent.py"),
    "actor_agent": Path("agents/opencode_agent.py"),
    "agent_client": Path("clients/opencode_client.py"),
    "relay": Path("tools/eval_shim.py"),
    "gate_evaluator": Path("scripts/evaluate_gate.py"),
    "official_validator": Path("scripts/validate_official_submission.py"),
    "billing_reconciler": Path("scripts/reconcile_novacode_billing.py"),
    "resume_protocol": Path("scripts/resume_protocol.py"),
    "artifact_preflight": Path("scripts/preflight_training_artifacts.py"),
    "v1_builder": Path("scripts/build_process_workflows.py"),
    "router_builder": Path("scripts/build_workflow_router_v2.py"),
    "optimizer_builder": Path("scripts/build_optimizer80_artifacts.ps1"),
    "gate_config": Path("configs/evaluation_gates.json"),
    "split_manifest": Path("configs/workflow_router_dev_ids.json"),
    "router_schema": Path("docs/workflow_router_v2.schema.json"),
    "billing_schema": Path("docs/novacode_billing_evidence.schema.json"),
}


class EvaluationInputError(ValueError):
    """Raised when a result tree cannot support the requested gate."""


@dataclass(frozen=True)
class Observation:
    domain: str
    run: int
    task_id: str
    completion: int
    state: int
    task: int
    ux: float
    input_tokens: int
    path: Path
    raw: Mapping[str, Any]

    @property
    def key(self) -> tuple[str, int, str]:
        return self.domain, self.run, self.task_id

    @property
    def cluster(self) -> tuple[str, str]:
        return self.domain, self.task_id


@dataclass(frozen=True)
class Dataset:
    root: Path
    observations: tuple[Observation, ...]

    @property
    def by_key(self) -> dict[tuple[str, int, str], Observation]:
        return {item.key: item for item in self.observations}


@dataclass(frozen=True)
class Check:
    name: str
    passed: bool
    actual: Any
    expected: Any
    detail: str = ""

    def as_dict(self) -> dict[str, Any]:
        result = {
            "name": self.name,
            "passed": self.passed,
            "actual": self.actual,
            "expected": self.expected,
        }
        if self.detail:
            result["detail"] = self.detail
        return result


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def task_ids_sha256(task_ids: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(task_ids) + "\n").encode("utf-8")).hexdigest()


def load_train_inventory_evidence(
    state_bench_root: Path, domains: Sequence[str]
) -> dict[str, Any]:
    """Hash the exact train-only inventory using the builders' canonical form."""

    train_root = state_bench_root.resolve() / "datasets" / "train_task_trajectories"
    files: dict[str, dict[str, Path]] = {}
    logical_names: list[str] = []
    for domain in domains:
        domain_root = (train_root / domain).resolve()
        paths = sorted(domain_root.glob("*.json"), key=lambda path: path.name)
        if (
            len(paths) != 100
            or len({path.stem for path in paths}) != 100
            or any(path.resolve().parent != domain_root for path in paths)
        ):
            raise EvaluationInputError(
                f"expected exactly 100 unique in-directory train trajectories for {domain}"
            )
        files[domain] = {path.stem: path for path in paths}
        logical_names.extend(
            f"datasets/train_task_trajectories/{domain}/{path.name}" for path in paths
        )
    return {
        "files": files,
        "inventory_sha256": canonical_sha256(sorted(logical_names)),
        "file_count": sum(len(values) for values in files.values()),
    }


def train_content_manifest_sha256(
    inventory: Mapping[str, Mapping[str, Path]],
    selected_ids: Mapping[str, Iterable[str]],
) -> tuple[str, int]:
    entries: list[dict[str, str]] = []
    for domain, values in selected_ids.items():
        for task_id in sorted(set(map(str, values))):
            path = inventory.get(domain, {}).get(task_id)
            if path is None:
                raise EvaluationInputError(
                    f"selected train trajectory is absent: {domain}/{task_id}"
                )
            entries.append(
                {
                    "path": f"datasets/train_task_trajectories/{domain}/{path.name}",
                    "sha256": file_sha256(path),
                }
            )
    entries.sort(key=lambda item: item["path"])
    return canonical_sha256(entries), len(entries)


def load_run_manifests(root: Path, domains: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    manifests: dict[str, Mapping[str, Any]] = {}
    for path in sorted(root.resolve().rglob("run_manifest.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationInputError(f"cannot read run manifest {path}: {exc}") from exc
        if not isinstance(payload, Mapping):
            raise EvaluationInputError(f"run manifest is not an object: {path}")
        domain = str(payload.get("domain", ""))
        if domain not in domains:
            raise EvaluationInputError(f"run manifest has unknown domain {domain!r}: {path}")
        if domain in manifests:
            raise EvaluationInputError(f"multiple run manifests for {domain} below {root}")
        manifests[domain] = payload
    return manifests


def load_state_bench_evidence(
    state_bench_root: Path, contract: Mapping[str, Any], domains: Sequence[str]
) -> dict[str, Any]:
    root = state_bench_root.resolve()
    if not root.is_dir():
        raise EvaluationInputError(f"STATE-Bench root does not exist: {root}")
    protocol_relative = str(contract.get("protocol_config", ""))
    protocol_path = root / Path(protocol_relative)
    try:
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise EvaluationInputError(f"cannot read pinned STATE-Bench metadata: {exc}") from exc
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationInputError(f"cannot inspect STATE-Bench git state: {exc}") from exc
    split_version = str(contract.get("split_version", ""))
    split_manifests: dict[str, Any] = {}
    for domain in domains:
        path = root / "state_bench" / "domains" / domain / "splits" / f"{split_version}.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationInputError(f"cannot read official split manifest {path}: {exc}") from exc
        split_manifests[domain] = {
            "path": path,
            "sha256": file_sha256(path),
            "payload": payload,
        }
    prompt_file_mismatches: list[dict[str, str]] = []
    if isinstance(protocol, Mapping):
        for section in ("simulator", "judge"):
            hashes = _nested(protocol, section, "prompt_hashes", default={})
            if not isinstance(hashes, Mapping):
                prompt_file_mismatches.append(
                    {"key": section, "expected": "prompt hash mapping", "actual": type(hashes).__name__}
                )
                continue
            for key, expected in hashes.items():
                parts = str(key).split("/", 1)
                if len(parts) != 2:
                    prompt_file_mismatches.append(
                        {"key": str(key), "expected": str(expected), "actual": "invalid prompt key"}
                    )
                    continue
                domain, filename = parts
                prompt_path = root / "state_bench" / "domains" / domain / "prompts" / filename
                actual = file_sha256(prompt_path) if prompt_path.is_file() else "missing"
                if actual != expected:
                    prompt_file_mismatches.append(
                        {"key": str(key), "expected": str(expected), "actual": actual}
                    )
    return {
        "root": root,
        "commit": commit,
        "tracked_status": tracked_status,
        "version": str(pyproject.get("project", {}).get("version", "")),
        "protocol_path": protocol_path,
        "protocol_sha256": file_sha256(protocol_path),
        "protocol": protocol,
        "prompt_file_mismatches": prompt_file_mismatches,
        "splits": split_manifests,
    }


def load_repository_evidence(runner_path: Path) -> dict[str, Any]:
    """Bind a result manifest to the complete frozen local implementation."""

    root = runner_path.resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        tracked_status = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain=v1", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise EvaluationInputError(f"cannot inspect selective-PWM git state: {exc}") from exc
    hashes: dict[str, str] = {}
    for name, relative in IMPLEMENTATION_PATHS.items():
        path = root / relative
        if not path.is_file():
            raise EvaluationInputError(f"frozen implementation file is missing: {path}")
        try:
            tracked = subprocess.run(
                ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative.as_posix()],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            _ = tracked
            working_blob = subprocess.run(
                ["git", "-C", str(root), "hash-object", "--", relative.as_posix()],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            committed_blob = subprocess.run(
                ["git", "-C", str(root), "rev-parse", f"{commit}:{relative.as_posix()}"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            raise EvaluationInputError(
                f"implementation file is not tracked at the declared commit: {relative.as_posix()}"
            ) from exc
        if working_blob != committed_blob:
            raise EvaluationInputError(
                f"implementation file differs from the declared commit: {relative.as_posix()}"
            )
        hashes[name] = file_sha256(path)
    return {
        "root": root,
        "commit": commit,
        "tracked_status": tracked_status,
        "implementation_sha256": hashes,
    }


def repository_file_is_frozen(repository: Mapping[str, Any], path: Path) -> bool:
    root = Path(repository["root"])
    try:
        relative = path.resolve().relative_to(root).as_posix()
        subprocess.run(
            ["git", "-C", str(root), "ls-files", "--error-unmatch", "--", relative],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        working_blob = subprocess.run(
            ["git", "-C", str(root), "hash-object", "--", relative],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        committed_blob = subprocess.run(
            ["git", "-C", str(root), "rev-parse", f"{repository['commit']}:{relative}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
    return bool(working_blob) and working_blob == committed_blob


def _manifest_core(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def _nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _expected_task_ids(
    gate_name: str,
    domain: str,
    router: Mapping[str, Any],
    state_bench: Mapping[str, Any],
) -> list[str]:
    if gate_name in {"dev", "lockbox"}:
        values = _nested(router, "splits", domain, gate_name, default=[])
    else:
        values = _nested(state_bench, "splits", domain, "payload", "splits", "test", default=[])
    return [str(value) for value in values] if isinstance(values, list) else []


def load_independent_optimizer_splits(
    *, state_bench_root: Path, repository_root: Path, domains: Sequence[str]
) -> tuple[dict[str, dict[str, list[str]]], str]:
    """Recompute 10/10/80 from train filenames and the tracked ID-only manifest."""

    manifest_path = repository_root / "configs" / "workflow_router_dev_ids.json"
    try:
        raw = manifest_path.read_bytes()
        manifest = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot read the tracked optimizer split manifest: {exc}") from exc
    if not isinstance(manifest, Mapping) or set(manifest) - {*domains, "seed", "method"}:
        raise EvaluationInputError("optimizer split manifest contains unsupported fields")
    train_root = state_bench_root / "datasets" / "train_task_trajectories"
    result: dict[str, dict[str, list[str]]] = {}
    for domain in domains:
        paths = sorted((train_root / domain).glob("*.json"), key=lambda path: path.name)
        inventory = {path.stem for path in paths}
        dev_values = manifest.get(domain)
        if (
            len(paths) != 100
            or len(inventory) != 100
            or not isinstance(dev_values, list)
            or len(dev_values) != 10
            or len(set(map(str, dev_values))) != 10
        ):
            raise EvaluationInputError(f"invalid 100/10 train inventory for {domain}")
        dev = [str(value) for value in dev_values]
        if not set(dev).issubset(inventory):
            raise EvaluationInputError(f"dev IDs are outside the train inventory for {domain}")
        ordered = sorted(
            inventory - set(dev),
            key=lambda task_id: (
                hashlib.sha256(f"{SPLIT_SEED}|{domain}|{task_id}".encode()).hexdigest(),
                task_id,
            ),
        )
        result[domain] = {"dev": dev, "lockbox": ordered[:10], "optimizer": ordered[10:]}
    return result, hashlib.sha256(raw).hexdigest()


def _flatten_deployments(value: Any) -> list[str]:
    if not isinstance(value, Mapping):
        return []
    result: list[str] = []
    for deployments in value.values():
        if isinstance(deployments, list):
            result.extend(str(item) for item in deployments)
        elif isinstance(deployments, str):
            result.append(deployments)
    return result


def _deployment_map_valid(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for key, deployments in value.items():
        if str(key) != "STATE_BENCH_EVAL_DEPLOYMENTS":
            return False
        if not isinstance(deployments, list) or not deployments:
            return False
        if any(item != "gpt-5.4" for item in deployments):
            return False
    return True


def _execution_check(
    checks: list[Check], name: str, passed: bool, actual: Any, expected: Any, detail: str = ""
) -> None:
    checks.append(Check(name, passed, actual, expected, detail))


def _aggregate_execution_checks(
    checks: list[Check],
    *,
    arm: str,
    domain: str,
    dataset: Dataset,
    observations: Sequence[Observation],
    runs: int,
    contract: Mapping[str, Any],
) -> None:
    path = dataset.root / domain / "metrics.json"
    prefix = f"{arm}.{domain}.aggregate"
    if not path.is_file():
        _execution_check(checks, f"{prefix}.exists", False, str(path), "existing metrics.json")
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        _execution_check(checks, f"{prefix}.readable", False, str(exc), "valid UTF-8 JSON")
        return
    if not isinstance(payload, Mapping):
        _execution_check(checks, f"{prefix}.shape", False, type(payload).__name__, "JSON object")
        return
    metadata = {
        "benchmark_version": payload.get("benchmark_version"),
        "evaluation_protocol_id": payload.get("evaluation_protocol_id"),
        "num_runs": payload.get("num_runs"),
        "agent_model": _nested(payload, "agent_model", "model_name"),
    }
    expected_metadata = {
        "benchmark_version": contract.get("benchmark_version"),
        "evaluation_protocol_id": contract.get("evaluation_protocol_id"),
        "num_runs": runs,
        "agent_model": contract.get("agent_model"),
    }
    _execution_check(
        checks,
        f"{prefix}.metadata",
        metadata == expected_metadata,
        metadata,
        {"equal": expected_metadata},
    )
    clusters: dict[str, list[Observation]] = {}
    for item in observations:
        clusters.setdefault(item.task_id, []).append(item)
    expected_metrics = {
        "task_completion_pass@1": round(
            sum(item.completion for item in observations) / len(observations), 2
        ),
        f"task_completion_pass^{runs}": round(
            sum(all(item.completion for item in items) for items in clusters.values())
            / len(clusters),
            2,
        ),
        "mean_ux_score": round(sum(item.ux for item in observations) / len(observations), 2),
    }
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), Mapping) else {}
    actual_metrics = {key: metrics.get(key) for key in expected_metrics}
    _execution_check(
        checks,
        f"{prefix}.raw_alignment",
        actual_metrics == expected_metrics,
        actual_metrics,
        {"equal": expected_metrics},
    )


def evaluate_execution_contract(
    *,
    gate_name: str,
    config: Mapping[str, Any],
    candidate: Dataset,
    baseline: Dataset | None,
    memory_path: Path,
    router_path: Path,
    state_bench_root: Path,
    runner_path: Path = DEFAULT_RUNNER,
) -> tuple[list[Check], str | None]:
    """Verify manifests, task provenance, protocol stamps, and router attribution."""

    contract = config.get("benchmark_contract")
    gates = config.get("gates")
    if not isinstance(contract, Mapping) or not isinstance(gates, Mapping):
        raise EvaluationInputError("configuration lacks benchmark_contract or gates")
    gate = gates.get(gate_name)
    if not isinstance(gate, Mapping):
        raise EvaluationInputError(f"unknown gate {gate_name!r}")
    domains = [str(value) for value in contract.get("domains", [])]
    if not memory_path.is_file() or not router_path.is_file() or not runner_path.is_file():
        raise EvaluationInputError("memory, router, and runner artifacts must all exist")
    try:
        router = json.loads(router_path.read_text(encoding="utf-8"))
        memory = json.loads(memory_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationInputError(f"cannot read memory/router artifact: {exc}") from exc
    if not isinstance(router, Mapping) or not isinstance(memory, Mapping):
        raise EvaluationInputError("memory/router artifact is not an object")
    state_bench = load_state_bench_evidence(state_bench_root, contract, domains)
    repository = load_repository_evidence(runner_path)
    candidate_manifests = load_run_manifests(candidate.root, domains)
    baseline_manifests = load_run_manifests(baseline.root, domains) if baseline else {}
    checks: list[Check] = []

    artifact_policy = config.get("artifact_policy")
    if not isinstance(artifact_policy, Mapping):
        raise EvaluationInputError("configuration lacks artifact_policy")
    kind = "optimizer80" if gate_name in set(artifact_policy.get("optimizer80_stages", [])) else "full100"
    expected_memory = REPO_ROOT / str(artifact_policy[f"{kind}_memory"])
    expected_router = REPO_ROOT / str(artifact_policy[f"{kind}_router"])
    _execution_check(
        checks,
        "artifacts.stage_memory_path",
        memory_path.resolve() == expected_memory.resolve(),
        str(memory_path.resolve()),
        {"equal": str(expected_memory.resolve())},
    )
    _execution_check(
        checks,
        "artifacts.stage_router_path",
        router_path.resolve() == expected_router.resolve(),
        str(router_path.resolve()),
        {"equal": str(expected_router.resolve())},
    )
    _execution_check(
        checks,
        "artifacts.memory_tracked_at_commit",
        repository_file_is_frozen(repository, memory_path),
        str(memory_path.resolve()),
        {"tracked_and_equal_to_commit": True},
    )
    _execution_check(
        checks,
        "artifacts.router_tracked_at_commit",
        repository_file_is_frozen(repository, router_path),
        str(router_path.resolve()),
        {"tracked_and_equal_to_commit": True},
    )
    train_inventory = load_train_inventory_evidence(state_bench_root, domains)
    all_train_ids = {
        domain: set(map(str, train_inventory["files"][domain])) for domain in domains
    }
    train_test_overlap = {
        domain: sorted(
            all_train_ids[domain]
            & set(
                map(
                    str,
                    _nested(
                        state_bench,
                        "splits",
                        domain,
                        "payload",
                        "splits",
                        "test",
                        default=[],
                    ),
                )
            )
        )
        for domain in domains
    }
    _execution_check(
        checks,
        "training_inventory.disjoint_from_official_test",
        all(not values for values in train_test_overlap.values()),
        train_test_overlap,
        {"all_domains": []},
    )
    if kind == "optimizer80":
        independent_splits, dev_manifest_sha256 = load_independent_optimizer_splits(
            state_bench_root=state_bench_root.resolve(),
            repository_root=Path(repository["root"]),
            domains=domains,
        )
        optimizer_manifest_sha256, optimizer_file_count = train_content_manifest_sha256(
            train_inventory["files"],
            {
                domain: independent_splits[domain]["optimizer"] for domain in domains
            },
        )
        router_splits = router.get("splits")
        normalized_router_splits = {
            domain: {
                split: list(map(str, _nested(router_splits, domain, split, default=[])))
                for split in ("dev", "lockbox", "optimizer")
            }
            for domain in domains
        }
        _execution_check(
            checks,
            "optimizer_split.independent_recomputation",
            normalized_router_splits == independent_splits,
            normalized_router_splits,
            {"equal": independent_splits},
        )
        official_test_overlap = {
            domain: sorted(
                set().union(*map(set, independent_splits[domain].values()))
                & set(map(str, _nested(state_bench, "splits", domain, "payload", "splits", "test", default=[])))
            )
            for domain in domains
        }
        _execution_check(
            checks,
            "optimizer_split.disjoint_from_official_test",
            all(not values for values in official_test_overlap.values()),
            official_test_overlap,
            {"all_domains": []},
        )
        memory_provenance = memory.get("provenance")
        memory_selected_counts = _nested(memory_provenance, "selected_counts", default={})
        memory_split_summary = _nested(memory_provenance, "split_summary", default={})
        memory_provenance_ok = (
            isinstance(memory_provenance, Mapping)
            and memory_provenance.get("task_split") == "optimizer"
            and memory_provenance.get("task_manifest_sha256") == dev_manifest_sha256
            and memory_provenance.get("train_inventory_sha256")
            == train_inventory["inventory_sha256"]
            and memory_provenance.get("inventory_file_count")
            == train_inventory["file_count"]
            and memory_provenance.get("trajectory_manifest_sha256")
            == optimizer_manifest_sha256
            and memory_provenance.get("read_content_manifest_sha256")
            == optimizer_manifest_sha256
            and optimizer_file_count == 240
            and memory_provenance.get("learning_input")
            == "datasets/train_task_trajectories only"
            and REQUIRED_EXCLUDED_SOURCES.issubset(
                set(map(str, memory_provenance.get("excluded_sources") or []))
            )
            and all(_nested(memory_selected_counts, domain) == 80 for domain in domains)
            and all(
                _nested(memory, "stats", domain, "trajectories") == 80
                for domain in domains
            )
            and all(
                {
                    split: sorted(map(str, _nested(memory_split_summary, domain, split, default=[])))
                    for split in ("dev", "lockbox", "optimizer")
                }
                == {
                    split: sorted(independent_splits[domain][split])
                    for split in ("dev", "lockbox", "optimizer")
                }
                for domain in domains
            )
        )
        _execution_check(
            checks,
            "optimizer_memory.provenance",
            memory_provenance_ok,
            {
                "task_split": _nested(memory_provenance, "task_split"),
                "task_manifest_sha256": _nested(memory_provenance, "task_manifest_sha256"),
                "train_inventory_sha256": _nested(
                    memory_provenance, "train_inventory_sha256"
                ),
                "trajectory_manifest_sha256": _nested(
                    memory_provenance, "trajectory_manifest_sha256"
                ),
                "selected_counts": memory_selected_counts,
            },
            {
                "task_split": "optimizer",
                "task_manifest_sha256": dev_manifest_sha256,
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "trajectory_manifest_sha256": optimizer_manifest_sha256,
                "selected_counts": {domain: 80 for domain in domains},
            },
        )
        source_violations: dict[str, list[str]] = {domain: [] for domain in domains}
        for card in memory.get("cards", []):
            if not isinstance(card, Mapping):
                continue
            domain = str(card.get("domain", ""))
            if domain not in source_violations:
                continue
            outside = set(map(str, card.get("source_tasks") or [])) - set(
                independent_splits[domain]["optimizer"]
            )
            source_violations[domain].extend(sorted(outside))
        source_violations = {
            domain: sorted(set(values)) for domain, values in source_violations.items()
        }
        _execution_check(
            checks,
            "optimizer_memory.source_tasks",
            all(not values for values in source_violations.values()),
            source_violations,
            {"all_domains": []},
        )
        router_provenance = router.get("provenance")
        source_overlap = _nested(router_provenance, "source_task_overlap", default={})
        router_provenance_ok = (
            isinstance(router_provenance, Mapping)
            and router_provenance.get("memory_training_split") == "optimizer"
            and router_provenance.get("lockbox_independent") is True
            and router_provenance.get("dev_manifest_sha256") == dev_manifest_sha256
            and router_provenance.get("train_file_count") == 240
            and router_provenance.get("train_manifest_sha256")
            == optimizer_manifest_sha256
            and router_provenance.get("read_content_manifest_sha256")
            == optimizer_manifest_sha256
            and router_provenance.get("train_inventory_sha256")
            == train_inventory["inventory_sha256"]
            and all(
                _nested(router_provenance, "memory_trajectory_counts", domain) == 80
                for domain in domains
            )
            and REQUIRED_EXCLUDED_SOURCES.issubset(
                set(map(str, router_provenance.get("excluded_sources") or []))
            )
            and all(
                not _nested(source_overlap, domain, split, default=[])
                for domain in domains
                for split in ("dev", "lockbox")
            )
            and router.get("source_memory_sha256") == file_sha256(memory_path)
        )
        _execution_check(
            checks,
            "optimizer_router.provenance",
            router_provenance_ok,
            {
                "memory_training_split": _nested(router_provenance, "memory_training_split"),
                "lockbox_independent": _nested(router_provenance, "lockbox_independent"),
                "dev_manifest_sha256": _nested(router_provenance, "dev_manifest_sha256"),
                "train_file_count": _nested(router_provenance, "train_file_count"),
                "train_manifest_sha256": _nested(
                    router_provenance, "train_manifest_sha256"
                ),
                "train_inventory_sha256": _nested(
                    router_provenance, "train_inventory_sha256"
                ),
                "source_task_overlap": source_overlap,
            },
            {
                "memory_training_split": "optimizer",
                "lockbox_independent": True,
                "dev_manifest_sha256": dev_manifest_sha256,
                "train_file_count": 240,
                "train_manifest_sha256": optimizer_manifest_sha256,
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "source_task_overlap": "empty",
            },
        )
    else:
        independent_splits, dev_manifest_sha256 = load_independent_optimizer_splits(
            state_bench_root=state_bench_root.resolve(),
            repository_root=Path(repository["root"]),
            domains=domains,
        )
        full_manifest_sha256, full_file_count = train_content_manifest_sha256(
            train_inventory["files"], all_train_ids
        )
        router_splits = router.get("splits")
        normalized_router_splits = {
            domain: {
                split: list(
                    map(str, _nested(router_splits, domain, split, default=[]))
                )
                for split in ("dev", "lockbox", "optimizer")
            }
            for domain in domains
        }
        _execution_check(
            checks,
            "full100_split.independent_recomputation",
            normalized_router_splits == independent_splits,
            normalized_router_splits,
            {"equal": independent_splits},
        )
        memory_provenance = memory.get("provenance")
        selected_counts = _nested(memory_provenance, "selected_counts", default={})
        split_summary = _nested(memory_provenance, "split_summary", default={})
        memory_provenance_ok = (
            isinstance(memory_provenance, Mapping)
            and memory_provenance.get("task_split") == "all"
            and memory_provenance.get("task_manifest_sha256") is None
            and memory_provenance.get("task_manifest_method")
            == "all_fixed_train_trajectories"
            and memory_provenance.get("train_inventory_sha256")
            == train_inventory["inventory_sha256"]
            and memory_provenance.get("inventory_file_count")
            == train_inventory["file_count"]
            and memory_provenance.get("trajectory_manifest_sha256")
            == full_manifest_sha256
            and memory_provenance.get("read_content_manifest_sha256")
            == full_manifest_sha256
            and full_file_count == 300
            and memory_provenance.get("learning_input")
            == "datasets/train_task_trajectories only"
            and REQUIRED_EXCLUDED_SOURCES.issubset(
                set(map(str, memory_provenance.get("excluded_sources") or []))
            )
            and all(_nested(selected_counts, domain) == 100 for domain in domains)
            and all(
                sorted(
                    map(
                        str,
                        _nested(split_summary, domain, "all", default=[]),
                    )
                )
                == sorted(all_train_ids[domain])
                for domain in domains
            )
            and all(
                _nested(memory, "stats", domain, "trajectories") == 100
                for domain in domains
            )
        )
        _execution_check(
            checks,
            "full100_memory.provenance",
            memory_provenance_ok,
            {
                "task_split": _nested(memory_provenance, "task_split"),
                "task_manifest_sha256": _nested(
                    memory_provenance, "task_manifest_sha256"
                ),
                "train_inventory_sha256": _nested(
                    memory_provenance, "train_inventory_sha256"
                ),
                "trajectory_manifest_sha256": _nested(
                    memory_provenance, "trajectory_manifest_sha256"
                ),
                "selected_counts": selected_counts,
            },
            {
                "task_split": "all",
                "task_manifest_sha256": None,
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "trajectory_manifest_sha256": full_manifest_sha256,
                "selected_counts": {domain: 100 for domain in domains},
            },
        )
        source_violations: dict[str, list[str]] = {domain: [] for domain in domains}
        actual_overlap: dict[str, dict[str, list[str]]] = {
            domain: {"dev": [], "lockbox": []} for domain in domains
        }
        for card in memory.get("cards", []):
            if not isinstance(card, Mapping):
                continue
            domain = str(card.get("domain", ""))
            if domain not in source_violations:
                source_violations.setdefault("__invalid_domain__", []).append(domain)
                continue
            sources = set(map(str, card.get("source_tasks") or []))
            source_violations[domain].extend(
                sorted(sources - all_train_ids[domain])
            )
            for split in ("dev", "lockbox"):
                actual_overlap[domain][split].extend(
                    sorted(sources & set(independent_splits[domain][split]))
                )
        source_violations = {
            domain: sorted(set(values)) for domain, values in source_violations.items()
        }
        actual_overlap = {
            domain: {
                split: sorted(set(values)) for split, values in split_values.items()
            }
            for domain, split_values in actual_overlap.items()
        }
        _execution_check(
            checks,
            "full100_memory.source_tasks",
            all(not values for values in source_violations.values()),
            source_violations,
            {"all_domains": []},
        )
        router_provenance = router.get("provenance")
        router_provenance_ok = (
            isinstance(router_provenance, Mapping)
            and router_provenance.get("memory_training_split") == "all"
            and router_provenance.get("lockbox_independent") is False
            and router_provenance.get("dev_manifest_sha256") == dev_manifest_sha256
            and router_provenance.get("train_file_count") == 300
            and router_provenance.get("train_manifest_sha256")
            == full_manifest_sha256
            and router_provenance.get("read_content_manifest_sha256")
            == full_manifest_sha256
            and router_provenance.get("train_inventory_sha256")
            == train_inventory["inventory_sha256"]
            and all(
                _nested(router_provenance, "memory_trajectory_counts", domain) == 100
                for domain in domains
            )
            and _nested(router_provenance, "source_task_overlap", default={})
            == actual_overlap
            and REQUIRED_EXCLUDED_SOURCES.issubset(
                set(map(str, router_provenance.get("excluded_sources") or []))
            )
            and router.get("source_memory_sha256") == file_sha256(memory_path)
        )
        _execution_check(
            checks,
            "full100_router.provenance",
            router_provenance_ok,
            {
                "memory_training_split": _nested(
                    router_provenance, "memory_training_split"
                ),
                "lockbox_independent": _nested(
                    router_provenance, "lockbox_independent"
                ),
                "dev_manifest_sha256": _nested(
                    router_provenance, "dev_manifest_sha256"
                ),
                "train_file_count": _nested(router_provenance, "train_file_count"),
                "train_manifest_sha256": _nested(
                    router_provenance, "train_manifest_sha256"
                ),
                "train_inventory_sha256": _nested(
                    router_provenance, "train_inventory_sha256"
                ),
                "source_task_overlap": _nested(
                    router_provenance, "source_task_overlap", default={}
                ),
            },
            {
                "memory_training_split": "all",
                "lockbox_independent": False,
                "dev_manifest_sha256": dev_manifest_sha256,
                "train_file_count": 300,
                "train_manifest_sha256": full_manifest_sha256,
                "train_inventory_sha256": train_inventory["inventory_sha256"],
                "source_task_overlap": actual_overlap,
            },
        )
    _execution_check(
        checks,
        "state_bench.commit",
        state_bench["commit"] == contract.get("state_bench_commit"),
        state_bench["commit"],
        {"equal": contract.get("state_bench_commit")},
    )
    _execution_check(
        checks,
        "state_bench.version",
        state_bench["version"] == contract.get("benchmark_version"),
        state_bench["version"],
        {"equal": contract.get("benchmark_version")},
    )
    _execution_check(
        checks,
        "state_bench.tracked_tree_clean",
        not state_bench["tracked_status"],
        state_bench["tracked_status"],
        {"equal": ""},
        "untracked files are intentionally ignored",
    )
    _execution_check(
        checks,
        "repository.tracked_tree_clean",
        not repository["tracked_status"],
        repository["tracked_status"],
        {"equal": ""},
        "all implementation and configuration changes must be committed before evaluation",
    )
    protocol = state_bench["protocol"]
    protocol_shape = {
        "split_version": protocol.get("split_version") if isinstance(protocol, Mapping) else None,
        "split": protocol.get("split") if isinstance(protocol, Mapping) else None,
        "num_runs": protocol.get("num_runs") if isinstance(protocol, Mapping) else None,
        "domains": sorted(map(str, protocol.get("domains", [])))
        if isinstance(protocol, Mapping)
        else None,
        "official_model": protocol.get("official_model") if isinstance(protocol, Mapping) else None,
        "simulator_model": _nested(protocol, "simulator", "model"),
        "judge_model": _nested(protocol, "judge", "model"),
        "judge_reasoning_effort": _nested(protocol, "judge", "reasoning_effort"),
    }
    expected_protocol_shape = {
        "split_version": contract.get("split_version"),
        "split": contract.get("split"),
        "num_runs": contract.get("num_runs"),
        "domains": sorted(domains),
        "official_model": contract.get("agent_model"),
        "simulator_model": contract.get("simulator_model"),
        "judge_model": contract.get("judge_model"),
        "judge_reasoning_effort": contract.get("judge_reasoning_effort"),
    }
    _execution_check(
        checks,
        "state_bench.protocol_contract",
        protocol_shape == expected_protocol_shape,
        protocol_shape,
        {"equal": expected_protocol_shape},
    )
    configured_hashes = contract.get("prompt_hashes")
    protocol_hashes = {
        "simulator": {
            str(key).split("/", 1)[0]: value
            for key, value in (_nested(protocol, "simulator", "prompt_hashes", default={}) or {}).items()
        },
        "judge": {
            domain: {
                str(key).split("/", 1)[1]: value
                for key, value in (_nested(protocol, "judge", "prompt_hashes", default={}) or {}).items()
                if str(key).startswith(f"{domain}/")
            }
            for domain in domains
        },
    }
    _execution_check(
        checks,
        "state_bench.official_prompt_hashes",
        protocol_hashes == configured_hashes,
        protocol_hashes,
        {"equal": configured_hashes},
    )
    _execution_check(
        checks,
        "state_bench.prompt_files_match_protocol",
        not state_bench["prompt_file_mismatches"],
        state_bench["prompt_file_mismatches"][:10],
        {"equal": []},
    )

    expected_manifest_contract = config.get("run_manifest_contract", {})
    candidate_stages: set[str] = set()
    arm_transport_origins: dict[str, set[str]] = {}
    arm_client_contracts: dict[str, set[str]] = {}
    arm_evaluation_client_contracts: dict[str, set[str]] = {}
    arm_datasets: list[tuple[str, Dataset, Mapping[str, Mapping[str, Any]]]] = [
        ("candidate", candidate, candidate_manifests)
    ]
    if baseline is not None:
        arm_datasets.append(("baseline", baseline, baseline_manifests))
    for arm, dataset, manifests in arm_datasets:
        transport_sessions: set[str] = set()
        transport_origins: set[str] = set()
        client_contracts: set[str] = set()
        evaluation_client_contracts: set[str] = set()
        _execution_check(
            checks,
            f"{arm}.manifests.complete",
            set(manifests) == set(domains),
            sorted(manifests),
            {"equal": sorted(domains)},
        )
        by_domain = {domain: [item for item in dataset.observations if item.domain == domain] for domain in domains}
        for domain in domains:
            manifest = manifests.get(domain)
            if not isinstance(manifest, Mapping):
                continue
            prefix = f"{arm}.{domain}.manifest"
            declared_hash = str(manifest.get("manifest_sha256", ""))
            actual_hash = canonical_sha256(_manifest_core(manifest))
            _execution_check(checks, f"{prefix}.self_hash", declared_hash == actual_hash, declared_hash, {"equal": actual_hash})
            _execution_check(
                checks,
                f"{prefix}.schema",
                manifest.get("schema_version") == expected_manifest_contract.get("schema_version")
                and manifest.get("created_by") == expected_manifest_contract.get("created_by"),
                {"schema_version": manifest.get("schema_version"), "created_by": manifest.get("created_by")},
                {"equal": expected_manifest_contract},
            )
            expected_ids = _expected_task_ids(gate_name, domain, router, state_bench)
            actual_ids = _nested(manifest, "run", "task_selection", "task_ids", default=[])
            actual_ids = [str(value) for value in actual_ids] if isinstance(actual_ids, list) else []
            _execution_check(checks, f"{prefix}.task_ids", actual_ids == expected_ids, actual_ids, {"equal": expected_ids})
            _execution_check(
                checks,
                f"{prefix}.task_ids_sha256",
                _nested(manifest, "run", "task_selection", "task_ids_sha256") == task_ids_sha256(expected_ids),
                _nested(manifest, "run", "task_selection", "task_ids_sha256"),
                {"equal": task_ids_sha256(expected_ids)},
            )
            observed_ids_by_run = {
                run: sorted(item.task_id for item in by_domain[domain] if item.run == run)
                for run in sorted({item.run for item in by_domain[domain]})
            }
            bad_run_ids = {
                run: ids for run, ids in observed_ids_by_run.items() if ids != sorted(expected_ids)
            }
            _execution_check(
                checks,
                f"{arm}.{domain}.trajectory_task_ids",
                not bad_run_ids,
                bad_run_ids,
                {"every_run_equal": sorted(expected_ids)},
            )
            session_summary: dict[int, Any] = {}
            session_chain_ok = True
            official_batch_members: dict[int, Mapping[str, Any]] = {}
            for run_index in range(1, int(gate["runs"]) + 1):
                try:
                    chain = verify_session_chain(
                        arm_root=dataset.root,
                        domain=domain,
                        run_index=run_index,
                        run_manifest_path=dataset.root
                        / domain
                        / "run_manifest.json",
                    )
                    resume_plan = plan_resume(
                        arm_root=dataset.root,
                        domain=domain,
                        run_index=run_index,
                        run_manifest_path=dataset.root
                        / domain
                        / "run_manifest.json",
                        task_ids=expected_ids,
                    )
                    records = list(chain.records)
                    origins = {
                        str(_nested(record, "relay_session", "upstream_origin_sha256", default=""))
                        for record in records
                    }
                    transport_origins.update(
                        origin
                        for origin in origins
                        if re.fullmatch(r"[0-9a-f]{64}", origin)
                    )
                    complete = (
                        bool(records)
                        and not resume_plan["agent_task_ids"]
                        and not resume_plan["score_task_ids"]
                        and not resume_plan["rejected_task_ids"]
                        and resume_plan["scored_task_ids"] == expected_ids
                        and len(origins) == 1
                    )
                    first_batch = records[0].get("fresh_batch") if records else None
                    if gate_name == "official750":
                        if isinstance(first_batch, Mapping):
                            official_batch_members[run_index] = first_batch
                        complete = complete and (
                            isinstance(first_batch, Mapping)
                            and first_batch.get("member_run_index") == run_index
                        )
                    session_summary[run_index] = {
                        "records": len(records),
                        "modes": [record.get("mode") for record in records],
                        "origins": sorted(origins),
                        "final_scored": len(resume_plan["scored_task_ids"]),
                        "latest_session_sha256": resume_plan[
                            "latest_session_sha256"
                        ],
                        "fresh_batch": dict(first_batch)
                        if isinstance(first_batch, Mapping)
                        else None,
                    }
                    session_chain_ok = session_chain_ok and complete
                except (OSError, ResumeProtocolError) as exc:
                    session_chain_ok = False
                    session_summary[run_index] = {
                        "error": type(exc).__name__,
                        "detail": str(exc),
                    }
            _execution_check(
                checks,
                f"{arm}.{domain}.auditable_session_chain",
                session_chain_ok,
                session_summary,
                {
                    "each_run": "verified hash chain with all expected trajectories scored",
                    "origin_count": 1,
                },
            )
            if gate_name == "official750":
                batch_identities = {
                    (
                        str(member.get("relative_path", "")),
                        str(member.get("sha256", "")),
                        str(member.get("batch_id", "")),
                    )
                    for member in official_batch_members.values()
                }
                official_batch_ok = (
                    set(official_batch_members) == {1, 2, 3, 4, 5}
                    and len(batch_identities) == 1
                    and {
                        member.get("member_run_index")
                        for member in official_batch_members.values()
                    }
                    == {1, 2, 3, 4, 5}
                )
                _execution_check(
                    checks,
                    f"{arm}.{domain}.official_fresh_batch",
                    official_batch_ok,
                    {
                        "members": {
                            run: dict(member)
                            for run, member in official_batch_members.items()
                        },
                        "batch_identity_count": len(batch_identities),
                    },
                    {
                        "one_shared_batch": True,
                        "run_indices": [1, 2, 3, 4, 5],
                        "command": (
                            "run_batch --split test --num-runs 5 "
                            "--num-runs-idx-start 1"
                        ),
                    },
                )
            expected_agent = "RiskAwareProcessWorkflowMemoryAgent" if arm == "candidate" else "ProcessWorkflowMemoryAgent"
            manifest_stage = manifest.get("router_stage")
            if arm == "candidate" and isinstance(manifest_stage, str):
                candidate_stages.add(manifest_stage)
            expected_protocol = {
                "benchmark_version": contract.get("benchmark_version"),
                "evaluation_protocol_id": contract.get("evaluation_protocol_id"),
                "split_version": contract.get("split_version"),
                "official_split": contract.get("split"),
                "official_num_runs": contract.get("num_runs"),
                "agent_model": contract.get("agent_model"),
                "simulator_model": contract.get("simulator_model"),
                "judge_model": contract.get("judge_model"),
                "judge_reasoning_effort": contract.get("judge_reasoning_effort"),
                "protocol_config_sha256": state_bench["protocol_sha256"],
                "prompt_hashes": {
                    "simulator": _nested(protocol, "simulator", "prompt_hashes", default={}),
                    "judge": _nested(protocol, "judge", "prompt_hashes", default={}),
                },
            }
            actual_protocol = {
                key: _nested(manifest, "protocol", key) for key in expected_protocol
            }
            _execution_check(checks, f"{prefix}.protocol", actual_protocol == expected_protocol, actual_protocol, {"equal": expected_protocol})
            _execution_check(
                checks,
                f"{prefix}.gpt54_deployments_only",
                _deployment_map_valid(_nested(manifest, "protocol", "evaluation_deployments", default={})),
                _nested(manifest, "protocol", "evaluation_deployments", default={}),
                {"keys": "STATE_BENCH_EVAL_DEPLOYMENTS[_N]", "unique_value": "gpt-5.4"},
            )
            expected_artifacts = {
                "artifact_kind": kind,
                "memory_sha256": file_sha256(memory_path),
                "router_sha256": file_sha256(router_path),
                "runner_sha256": file_sha256(runner_path),
                "repository_commit": repository["commit"],
                "repository_tracked_tree_clean": True,
                "implementation_sha256": repository["implementation_sha256"],
                "state_bench_commit": state_bench["commit"],
                "state_bench_version": state_bench["version"],
                "state_bench_tracked_tree_clean": True,
                "state_bench_protocol_sha256": state_bench["protocol_sha256"],
                "state_bench_split_manifest_sha256": state_bench["splits"][domain]["sha256"],
            }
            actual_artifacts = {key: _nested(manifest, "artifacts", key) for key in expected_artifacts}
            _execution_check(checks, f"{prefix}.artifacts", actual_artifacts == expected_artifacts, actual_artifacts, {"equal": expected_artifacts})
            expected_run = {
                "num_runs": int(gate["runs"]),
                "run_start": 1,
                "retry_attempts": 1,
                "retrieve_learnings_top_k": int(contract["retrieve_learnings_top_k"]),
                "ignore_missing_runs": bool(contract["ignore_missing"]),
                "agent_class": expected_agent,
                "agent_client_class": "OpenCodeLLMClient",
                "memory_mode": "hybrid",
            }
            actual_run = {key: _nested(manifest, "run", key) for key in expected_run}
            _execution_check(checks, f"{prefix}.run_contract", actual_run == expected_run, actual_run, {"equal": expected_run})
            expected_client_contract = contract.get("agent_client_contract")
            actual_client_contract = _nested(
                manifest, "run", "agent_client_contract", default={}
            )
            _execution_check(
                checks,
                f"{prefix}.agent_client_contract",
                isinstance(expected_client_contract, Mapping)
                and actual_client_contract == expected_client_contract,
                actual_client_contract,
                {"equal": expected_client_contract},
            )
            if isinstance(actual_client_contract, Mapping):
                client_contracts.add(canonical_sha256(actual_client_contract))
            expected_evaluation_client_contract = contract.get(
                "official_evaluation_client_contract"
            )
            actual_evaluation_client_contract = _nested(
                manifest,
                "run",
                "official_evaluation_client_contract",
                default={},
            )
            _execution_check(
                checks,
                f"{prefix}.official_evaluation_client_contract",
                isinstance(expected_evaluation_client_contract, Mapping)
                and actual_evaluation_client_contract
                == expected_evaluation_client_contract,
                actual_evaluation_client_contract,
                {"equal": expected_evaluation_client_contract},
            )
            if isinstance(actual_evaluation_client_contract, Mapping):
                evaluation_client_contracts.add(
                    canonical_sha256(actual_evaluation_client_contract)
                )
            transport = manifest.get("transport")
            transport_shape = {
                "provider": _nested(transport, "provider"),
                "relay_sha256": _nested(transport, "relay_sha256"),
                "rpm": _nested(transport, "rpm"),
                "burst": _nested(transport, "burst"),
                "burst_window_seconds": _nested(transport, "burst_window_seconds"),
                "attempts": _nested(transport, "attempts"),
                "only_transport_retries": _nested(transport, "only_transport_retries"),
            }
            expected_transport_shape = {
                "provider": "novacode",
                "relay_sha256": repository["implementation_sha256"]["relay"],
                "rpm": 45,
                "burst": 5,
                "burst_window_seconds": 1.0,
                "attempts": 5,
                "only_transport_retries": True,
            }
            origin_hash = str(_nested(transport, "upstream_origin_sha256", default=""))
            session_id = str(_nested(transport, "relay_session_id", default=""))
            ledger_relative = str(_nested(transport, "ledger_relative_path", default=""))
            log_relative = str(_nested(transport, "log_relative_path", default=""))
            expected_ledger_relative = f"_transport/relay-{session_id}.jsonl"
            expected_log_relative = f"_transport/relay-{session_id}.log"
            ledger_path = (dataset.root / ledger_relative).resolve()
            log_path = (dataset.root / log_relative).resolve()
            paths_within_root = (
                dataset.root.resolve() in ledger_path.parents
                and dataset.root.resolve() in log_path.parents
            )
            ledger_header: Mapping[str, Any] = {}
            if paths_within_root and ledger_path.is_file():
                try:
                    first_line = ledger_path.read_text(encoding="utf-8").splitlines()[0]
                    parsed_header = json.loads(first_line)
                    if isinstance(parsed_header, Mapping):
                        ledger_header = parsed_header
                except (IndexError, OSError, UnicodeError, json.JSONDecodeError):
                    ledger_header = {}
            header_ok = (
                ledger_header.get("event") == "session_start"
                and ledger_header.get("provider") == "novacode"
                and ledger_header.get("upstream_origin_sha256") == origin_hash
                and ledger_header.get("rpm") == 45
                and ledger_header.get("burst") == 5
                and ledger_header.get("burst_window_seconds") == 1.0
                and ledger_header.get("attempts") == 5
            )
            if re.fullmatch(r"[0-9a-f]{64}", origin_hash):
                transport_origins.add(origin_hash)
            if re.fullmatch(r"[0-9a-f]{32}", session_id):
                transport_sessions.add(session_id)
            _execution_check(
                checks,
                f"{prefix}.transport",
                transport_shape == expected_transport_shape
                and re.fullmatch(r"[0-9a-f]{64}", origin_hash) is not None
                and re.fullmatch(r"[0-9a-f]{32}", session_id) is not None,
                {**transport_shape, "upstream_origin_sha256": origin_hash, "relay_session_id": session_id},
                {**expected_transport_shape, "origin_hash": "64 lowercase hex", "session_id": "32 lowercase hex"},
            )
            _execution_check(
                checks,
                f"{prefix}.transport_files",
                paths_within_root
                and ledger_relative == expected_ledger_relative
                and log_relative == expected_log_relative
                and ledger_path.is_file()
                and log_path.is_file()
                and header_ok,
                {
                    "ledger_relative_path": ledger_relative,
                    "log_relative_path": log_relative,
                    "ledger_exists": ledger_path.is_file(),
                    "log_exists": log_path.is_file(),
                    "ledger_header_valid": header_ok,
                },
                {
                    "ledger_relative_path": expected_ledger_relative,
                    "log_relative_path": expected_log_relative,
                    "ledger_exists": True,
                    "log_exists": True,
                    "ledger_header_valid": True,
                },
            )
            workers = _nested(manifest, "run", "workers")
            workers_is_integer = isinstance(workers, int) and not isinstance(
                workers, bool
            )
            if gate_name == "official750":
                valid_worker_count = workers_is_integer and workers == 2
                expected_workers = {"equal": 2}
            else:
                valid_worker_count = workers_is_integer and 1 <= workers <= 3
                expected_workers = {"minimum": 1, "maximum": 3}
            _execution_check(
                checks,
                f"{prefix}.workers",
                valid_worker_count,
                workers,
                expected_workers,
            )
            expected_router_stage = manifest_stage if arm == "candidate" else None
            expected_task_selection = {
                "mode": "explicit" if gate_name in {"dev", "lockbox"} else "split",
                "source": f"optimizer80_router_{gate_name}"
                if gate_name in {"dev", "lockbox"}
                else "state_bench_official_test_split",
                "split": None if gate_name in {"dev", "lockbox"} else "test",
            }
            actual_task_selection = {
                key: _nested(manifest, "run", "task_selection", key)
                for key in expected_task_selection
            }
            _execution_check(
                checks,
                f"{prefix}.arm_stage_domain",
                manifest.get("stage") == gate_name
                and manifest.get("arm") == arm
                and manifest.get("domain") == domain
                and manifest.get("router_stage") == expected_router_stage,
                {
                    "stage": manifest.get("stage"),
                    "arm": manifest.get("arm"),
                    "domain": manifest.get("domain"),
                    "router_stage": manifest.get("router_stage"),
                },
                {
                    "equal": {
                        "stage": gate_name,
                        "arm": arm,
                        "domain": domain,
                        "router_stage": expected_router_stage,
                    }
                },
            )
            _execution_check(
                checks,
                f"{prefix}.task_selection",
                actual_task_selection == expected_task_selection,
                actual_task_selection,
                {"equal": expected_task_selection},
            )
            expected_simulator_hash = _nested(configured_hashes, "simulator", domain)
            expected_judge_hashes = _nested(configured_hashes, "judge", domain, default={})
            bad_protocol = [
                str(item.path)
                for item in by_domain[domain]
                if item.raw.get("evaluation_protocol_id") != contract.get("evaluation_protocol_id")
                or item.raw.get("scoring_protocol_id") != contract.get("evaluation_protocol_id")
                or item.raw.get("simulator_model") != contract.get("simulator_model")
                or item.raw.get("judge_model") != contract.get("judge_model")
                or item.raw.get("judge_reasoning_effort") != contract.get("judge_reasoning_effort")
                or _nested(item.raw, "agent_model", "model_name") != contract.get("agent_model")
                or item.raw.get("simulator_prompt_hash") != expected_simulator_hash
                or item.raw.get("judge_prompt_hashes") != expected_judge_hashes
                or item.raw.get("agent_name") != expected_agent
            ]
            _execution_check(checks, f"{arm}.{domain}.trajectory_protocol", not bad_protocol, {"mismatches": len(bad_protocol)}, {"mismatches": 0}, f"first: {bad_protocol[:3]!r}" if bad_protocol else "")
            if arm == "baseline":
                unexpected_router = [
                    str(item.path) for item in by_domain[domain] if "workflow_router" in item.raw
                ]
                _execution_check(
                    checks,
                    f"baseline.{domain}.router_absent",
                    not unexpected_router,
                    {"mismatches": len(unexpected_router)},
                    {"mismatches": 0},
                    f"first: {unexpected_router[:3]!r}" if unexpected_router else "",
                )
            if gate_name in {"paired150", "official750"}:
                _aggregate_execution_checks(
                    checks,
                    arm=arm,
                    domain=domain,
                    dataset=dataset,
                    observations=by_domain[domain],
                    runs=int(gate["runs"]),
                    contract=contract,
                )
        _execution_check(
            checks,
            f"{arm}.single_transport_session",
            len(transport_sessions) == 1,
            sorted(transport_sessions),
            {"count": 1},
        )
        _execution_check(
            checks,
            f"{arm}.single_upstream_origin",
            len(transport_origins) == 1,
            sorted(transport_origins),
            {"count": 1},
        )
        _execution_check(
            checks,
            f"{arm}.single_agent_client_contract",
            len(client_contracts) == 1,
            sorted(client_contracts),
            {"count": 1},
        )
        _execution_check(
            checks,
            f"{arm}.single_official_evaluation_client_contract",
            len(evaluation_client_contracts) == 1,
            sorted(evaluation_client_contracts),
            {"count": 1},
        )
        arm_transport_origins[arm] = transport_origins
        arm_client_contracts[arm] = client_contracts
        arm_evaluation_client_contracts[arm] = evaluation_client_contracts

    if baseline is not None:
        candidate_origins = arm_transport_origins.get("candidate", set())
        baseline_origins = arm_transport_origins.get("baseline", set())
        _execution_check(
            checks,
            "paired.same_upstream_origin",
            len(candidate_origins) == 1 and candidate_origins == baseline_origins,
            {
                "candidate": sorted(candidate_origins),
                "baseline": sorted(baseline_origins),
            },
            {"sets_equal_with_count": 1},
        )
        candidate_evaluation_clients = arm_evaluation_client_contracts.get(
            "candidate", set()
        )
        baseline_evaluation_clients = arm_evaluation_client_contracts.get(
            "baseline", set()
        )
        _execution_check(
            checks,
            "paired.same_official_evaluation_client_contract",
            len(candidate_evaluation_clients) == 1
            and candidate_evaluation_clients == baseline_evaluation_clients,
            {
                "candidate": sorted(candidate_evaluation_clients),
                "baseline": sorted(baseline_evaluation_clients),
            },
            {"sets_equal_with_count": 1},
        )
        candidate_clients = arm_client_contracts.get("candidate", set())
        baseline_clients = arm_client_contracts.get("baseline", set())
        _execution_check(
            checks,
            "paired.same_agent_client_contract",
            len(candidate_clients) == 1 and candidate_clients == baseline_clients,
            {
                "candidate": sorted(candidate_clients),
                "baseline": sorted(baseline_clients),
            },
            {"sets_equal_with_count": 1},
        )

    _execution_check(
        checks,
        "candidate.single_router_stage",
        len(candidate_stages) == 1,
        sorted(candidate_stages),
        {"count": 1},
    )
    candidate_stage = next(iter(candidate_stages)) if len(candidate_stages) == 1 else None
    required_stage = gate.get("required_candidate_router_stage")
    allowed_stages = set(map(str, gate.get("allowed_candidate_router_stages", [])))
    stage_ok = candidate_stage == required_stage if required_stage else candidate_stage in allowed_stages
    _execution_check(
        checks,
        "candidate.router_stage_policy",
        stage_ok,
        candidate_stage,
        {"equal": required_stage} if required_stage else {"one_of": sorted(allowed_stages)},
    )
    expected_candidate_name = f"candidate-{candidate_stage}" if candidate_stage else None
    _execution_check(
        checks,
        "candidate.output_path",
        candidate.root.name == expected_candidate_name,
        candidate.root.name,
        {"equal": expected_candidate_name},
    )
    if baseline is not None:
        _execution_check(
            checks,
            "baseline.output_path",
            baseline.root.name == "baseline",
            baseline.root.name,
            {"equal": "baseline"},
        )

    router_policy = config.get("router_policy") if isinstance(config.get("router_policy"), Mapping) else {}
    required_promoted = set(map(str, router_policy.get("required_promoted_domains", [])))
    fallback_domains = set(map(str, router_policy.get("allowed_baseline_fallback_domains", [])))
    for domain in domains:
        promoted = _nested(router, "domain_configs", domain, "promoted")
        expected_promoted = domain in required_promoted
        _execution_check(
            checks,
            f"router.{domain}.promotion",
            promoted is expected_promoted,
            promoted,
            {"equal": expected_promoted},
        )
        domain_items = [item for item in candidate.observations if item.domain == domain]
        if domain in required_promoted:
            bad_router = [
                str(item.path)
                for item in domain_items
                if not isinstance(item.raw.get("workflow_router"), Mapping)
                or item.raw["workflow_router"].get("mode") != "enforce"
                or item.raw["workflow_router"].get("stage") != candidate_stage
                or item.raw["workflow_router"].get("router_enabled") is not True
            ]
            expected_router_state: Any = {"mode": "enforce", "stage": candidate_stage, "router_enabled": True}
        elif domain in fallback_domains:
            bad_router = [str(item.path) for item in domain_items if "workflow_router" in item.raw]
            expected_router_state = "workflow_router absent (parent byte-equivalent fallback)"
        else:
            bad_router = [str(item.path) for item in domain_items]
            expected_router_state = "domain must be declared promoted or fallback"
        _execution_check(checks, f"candidate.{domain}.router_runtime", not bad_router, {"mismatches": len(bad_router)}, {"mismatches": 0, "state": expected_router_state}, f"first: {bad_router[:3]!r}" if bad_router else "")
    return checks, candidate_stage


def _number(value: Any, *, field: str, path: Path) -> float:
    if isinstance(value, bool):
        return float(value)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise EvaluationInputError(f"{path}: {field} is missing or non-numeric") from exc
    if not math.isfinite(result):
        raise EvaluationInputError(f"{path}: {field} is not finite")
    return result


def _binary(value: Any, *, field: str, path: Path) -> int:
    number = _number(value, field=field, path=path)
    if number not in (0.0, 1.0):
        raise EvaluationInputError(f"{path}: {field} must be binary, got {value!r}")
    return int(number)


def _domain_for(path: Path, raw: Mapping[str, Any], domains: Sequence[str]) -> str | None:
    declared = raw.get("domain")
    if declared in domains:
        return str(declared)
    lowered_parts = {part.lower(): part for part in path.parts}
    matches = [domain for domain in domains if domain.lower() in lowered_parts]
    return matches[-1] if matches else None


def _run_for(path: Path, raw: Mapping[str, Any]) -> int:
    for field in ("run_index", "run_idx", "run"):
        value = raw.get(field)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
    for part in reversed(path.parts):
        match = re.fullmatch(r"run[_-]?(\d+)", part, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1


def _input_tokens(raw: Mapping[str, Any], path: Path) -> int:
    usage = raw.get("token_usage")
    value = usage.get("input_tokens") if isinstance(usage, Mapping) else None
    if value is None:
        value = raw.get("input_tokens")
    number = _number(value, field="input_tokens", path=path)
    if number < 0 or not number.is_integer():
        raise EvaluationInputError(f"{path}: input_tokens must be a non-negative integer")
    return int(number)


def load_dataset(root: Path, domains: Sequence[str]) -> Dataset:
    """Load only canonical ``<root>/<domain>/runN/*.json`` trajectories.

    Resume staging, metrics, and audit trees are intentionally outside this
    inventory.  Declared domain/run metadata, when present, must agree with the
    canonical path rather than overriding it.
    """

    root = root.resolve()
    if not root.is_dir():
        raise EvaluationInputError(f"result directory does not exist: {root}")
    observations: list[Observation] = []
    seen: dict[tuple[str, int, str], Path] = {}
    canonical_paths: list[tuple[str, int, Path]] = []
    for path_domain in domains:
        domain_root = root / path_domain
        if not domain_root.is_dir():
            continue
        for run_dir in sorted(domain_root.iterdir()):
            if not run_dir.is_dir():
                continue
            match = re.fullmatch(r"run(\d+)", run_dir.name)
            if match is None:
                continue
            path_run = int(match.group(1))
            canonical_paths.extend(
                (path_domain, path_run, path) for path in sorted(run_dir.glob("*.json"))
            )
    for path_domain, path_run, path in canonical_paths:
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise EvaluationInputError(f"cannot read JSON result {path}: {exc}") from exc
        if not isinstance(raw, Mapping) or raw.get("task_id") is None:
            continue
        declared_domain = raw.get("domain")
        if declared_domain is not None and declared_domain != path_domain:
            raise EvaluationInputError(
                f"{path}: declared domain {declared_domain!r} disagrees with path domain {path_domain!r}"
            )
        for field in ("run_index", "run_idx", "run"):
            declared_run = raw.get(field)
            if declared_run is not None and (
                not isinstance(declared_run, int)
                or isinstance(declared_run, bool)
                or declared_run != path_run
            ):
                raise EvaluationInputError(
                    f"{path}: declared {field}={declared_run!r} disagrees with path run{path_run}"
                )
        task_id = str(raw.get("task_id", "")).strip()
        if not task_id:
            raise EvaluationInputError(f"{path}: task_id is empty")
        ux = _number(raw.get("ux_score"), field="ux_score", path=path)
        if not 1.0 <= ux <= 5.0:
            raise EvaluationInputError(
                f"{path}: ux_score must be within the official [1, 5] range"
            )
        observation = Observation(
            domain=path_domain,
            run=path_run,
            task_id=task_id,
            completion=_binary(raw.get("task_completion_pass"), field="task_completion_pass", path=path),
            state=_binary(raw.get("state_requirements_met"), field="state_requirements_met", path=path),
            task=_binary(raw.get("task_requirements_met"), field="task_requirements_met", path=path),
            ux=ux,
            input_tokens=_input_tokens(raw, path),
            path=path,
            raw=raw,
        )
        if observation.key in seen:
            raise EvaluationInputError(
                f"duplicate observation {observation.key}: {seen[observation.key]} and {path}"
            )
        seen[observation.key] = path
        observations.append(observation)
    if not observations:
        raise EvaluationInputError(f"no scored task trajectories found below {root}")
    return Dataset(root=root, observations=tuple(observations))


def _count(items: Iterable[Observation], field: str) -> int:
    return sum(int(getattr(item, field)) for item in items)


def _mean(items: Sequence[Observation], field: str) -> float:
    if not items:
        raise EvaluationInputError(f"cannot compute {field} mean from zero observations")
    return sum(float(getattr(item, field)) for item in items) / len(items)


def _macro_ux(items: Sequence[Observation], domains: Sequence[str]) -> float:
    means = []
    for domain in domains:
        selected = [item for item in items if item.domain == domain]
        means.append(_mean(selected, "ux"))
    return sum(means) / len(means)


def _all_runs_pass(items: Sequence[Observation], expected_runs: int) -> dict[tuple[str, str], int]:
    grouped: dict[tuple[str, str], list[Observation]] = {}
    for item in items:
        grouped.setdefault(item.cluster, []).append(item)
    result: dict[tuple[str, str], int] = {}
    for cluster, observations in grouped.items():
        if len(observations) != expected_runs:
            raise EvaluationInputError(
                f"cluster {cluster} has {len(observations)} run(s), expected {expected_runs}"
            )
        result[cluster] = int(all(item.completion for item in observations))
    return result


def _cluster_deltas(
    baseline: Dataset, candidate: Dataset
) -> list[tuple[tuple[str, str], float]]:
    base_by_key = baseline.by_key
    candidate_by_key = candidate.by_key
    if set(base_by_key) != set(candidate_by_key):
        raise EvaluationInputError("cluster deltas require identical paired observations")
    grouped: dict[tuple[str, str], list[float]] = {}
    for key, candidate_item in candidate_by_key.items():
        grouped.setdefault(candidate_item.cluster, []).append(
            candidate_item.completion - base_by_key[key].completion
        )
    return sorted((cluster, sum(values) / len(values)) for cluster, values in grouped.items())


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise EvaluationInputError("cannot take a quantile of no values")
    ordered = sorted(values)
    position = (len(ordered) - 1) * min(1.0, max(0.0, probability))
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap_lower_bound(
    deltas: Sequence[float], *, confidence: float, iterations: int, seed: int
) -> float:
    """One-sided percentile lower bound, resampling paired task clusters."""

    if not deltas:
        raise EvaluationInputError("cluster bootstrap needs at least one paired task")
    if not 0.0 < confidence < 1.0:
        raise EvaluationInputError("bootstrap confidence must be between zero and one")
    if iterations < 100:
        raise EvaluationInputError("bootstrap iterations must be at least 100")
    generator = random.Random(seed)
    size = len(deltas)
    estimates = [
        sum(deltas[generator.randrange(size)] for _ in range(size)) / size
        for _ in range(iterations)
    ]
    return _quantile(estimates, 1.0 - confidence)


def _add_minimum(
    checks: list[Check], name: str, actual: float | int, expected: float | int, detail: str = ""
) -> None:
    checks.append(Check(name, actual >= expected, actual, {"minimum": expected}, detail))


def _add_maximum(
    checks: list[Check], name: str, actual: float | int, expected: float | int, detail: str = ""
) -> None:
    checks.append(Check(name, actual <= expected, actual, {"maximum": expected}, detail))


def _validate_shape(dataset: Dataset, gate: Mapping[str, Any], domains: Sequence[str], label: str) -> list[Check]:
    checks: list[Check] = []
    runs = int(gate["runs"])
    tasks = int(gate["tasks_per_domain"])
    expected_observations = runs * tasks
    for domain in domains:
        selected = [item for item in dataset.observations if item.domain == domain]
        clusters = {item.task_id for item in selected}
        run_ids = {item.run for item in selected}
        checks.append(
            Check(
                f"{label}.{domain}.observations",
                len(selected) == expected_observations,
                len(selected),
                {"equal": expected_observations},
            )
        )
        checks.append(
            Check(
                f"{label}.{domain}.tasks",
                len(clusters) == tasks,
                len(clusters),
                {"equal": tasks},
            )
        )
        checks.append(
            Check(
                f"{label}.{domain}.runs",
                run_ids == set(range(1, runs + 1)),
                sorted(run_ids),
                {"equal": list(range(1, runs + 1))},
            )
        )
        counts = {task_id: 0 for task_id in clusters}
        for item in selected:
            counts[item.task_id] += 1
        irregular = sorted(task_id for task_id, count in counts.items() if count != runs)
        checks.append(
            Check(
                f"{label}.{domain}.complete_clusters",
                not irregular,
                irregular,
                {"runs_per_task": runs},
            )
        )
    configured = set(domains)
    extras = sorted({item.domain for item in dataset.observations} - configured)
    checks.append(Check(f"{label}.configured_domains_only", not extras, extras, {"equal": []}))
    return checks


def evaluate_gate(
    *,
    gate_name: str,
    config: Mapping[str, Any],
    candidate: Dataset,
    baseline: Dataset | None = None,
) -> dict[str, Any]:
    gates = config.get("gates")
    if not isinstance(gates, Mapping) or gate_name not in gates:
        available = ", ".join(sorted(gates or {}))
        raise EvaluationInputError(f"unknown gate {gate_name!r}; available: {available}")
    gate = gates[gate_name]
    if not isinstance(gate, Mapping):
        raise EvaluationInputError(f"gate {gate_name!r} is not an object")
    contract = config.get("benchmark_contract")
    if not isinstance(contract, Mapping):
        raise EvaluationInputError("configuration is missing benchmark_contract")
    domains = [str(domain) for domain in contract.get("domains", [])]
    if not domains:
        raise EvaluationInputError("benchmark_contract.domains is empty")
    paired = bool(gate.get("paired"))
    if paired and baseline is None:
        raise EvaluationInputError(f"gate {gate_name!r} requires --baseline")

    checks = _validate_shape(candidate, gate, domains, "candidate")
    if baseline is not None:
        checks.extend(_validate_shape(baseline, gate, domains, "baseline"))
    if paired and baseline is not None:
        candidate_keys = set(candidate.by_key)
        baseline_keys = set(baseline.by_key)
        missing_candidate = sorted(baseline_keys - candidate_keys)
        missing_baseline = sorted(candidate_keys - baseline_keys)
        checks.append(
            Check(
                "paired.identical_observations",
                not missing_candidate and not missing_baseline,
                {
                    "missing_candidate": len(missing_candidate),
                    "missing_baseline": len(missing_baseline),
                },
                {"missing_candidate": 0, "missing_baseline": 0},
                "first mismatches: " + repr((missing_candidate + missing_baseline)[:5])
                if missing_candidate or missing_baseline
                else "",
            )
        )

    candidate_items = list(candidate.observations)
    baseline_items = list(baseline.observations) if baseline is not None else []
    candidate_completion = _count(candidate_items, "completion")
    candidate_state = _count(candidate_items, "state")
    candidate_task = _count(candidate_items, "task")

    if "minimum_candidate_completion" in gate:
        _add_minimum(
            checks,
            "completion.candidate",
            candidate_completion,
            int(gate["minimum_candidate_completion"]),
        )
    if baseline is not None:
        baseline_completion = _count(baseline_items, "completion")
        baseline_state = _count(baseline_items, "state")
        baseline_task = _count(baseline_items, "task")
        if "minimum_completion_gain" in gate:
            _add_minimum(
                checks,
                "completion.gain",
                candidate_completion - baseline_completion,
                int(gate["minimum_completion_gain"]),
            )
        if "minimum_state_requirements_gain" in gate:
            _add_minimum(
                checks,
                "state_requirements.gain",
                candidate_state - baseline_state,
                int(gate["minimum_state_requirements_gain"]),
            )
        if "minimum_task_requirements_gain" in gate:
            _add_minimum(
                checks,
                "task_requirements.gain",
                candidate_task - baseline_task,
                int(gate["minimum_task_requirements_gain"]),
            )

    domain_candidate_floor = gate.get("minimum_domain_candidate_completion", {})
    if isinstance(domain_candidate_floor, Mapping):
        for domain, minimum in domain_candidate_floor.items():
            actual = _count([item for item in candidate_items if item.domain == domain], "completion")
            _add_minimum(checks, f"completion.{domain}.candidate", actual, int(minimum))

    domain_gain_floor = gate.get("minimum_domain_completion_gain", {})
    if isinstance(domain_gain_floor, Mapping):
        if baseline is None and domain_gain_floor:
            raise EvaluationInputError("domain completion gains require a baseline")
        for domain, minimum in domain_gain_floor.items():
            candidate_count = _count(
                [item for item in candidate_items if item.domain == domain], "completion"
            )
            baseline_count = _count(
                [item for item in baseline_items if item.domain == domain], "completion"
            )
            _add_minimum(
                checks,
                f"completion.{domain}.gain",
                candidate_count - baseline_count,
                int(minimum),
            )

    if "minimum_ux_delta" in gate:
        if baseline is None:
            raise EvaluationInputError("UX delta requires a baseline")
        actual = _mean(candidate_items, "ux") - _mean(baseline_items, "ux")
        _add_minimum(checks, "ux.delta", round(actual, 8), float(gate["minimum_ux_delta"]))

    if "maximum_input_token_ratio" in gate:
        if baseline is None:
            raise EvaluationInputError("input-token ratio requires a baseline")
        denominator = sum(item.input_tokens for item in baseline_items)
        ratio = (
            sum(item.input_tokens for item in candidate_items) / denominator
            if denominator
            else math.inf
        )
        _add_maximum(
            checks,
            "input_tokens.ratio",
            round(ratio, 8),
            float(gate["maximum_input_token_ratio"]),
        )

    if "minimum_macro_ux" in gate:
        candidate_macro_ux = _macro_ux(candidate_items, domains)
        floor = float(gate["minimum_macro_ux"])
        if "minimum_macro_ux_relative_to_baseline" in gate:
            if baseline is None:
                raise EvaluationInputError("relative macro UX floor requires a baseline")
            floor = max(
                floor,
                _macro_ux(baseline_items, domains)
                + float(gate["minimum_macro_ux_relative_to_baseline"]),
            )
        _add_minimum(checks, "ux.macro", round(candidate_macro_ux, 8), round(floor, 8))

    expected_runs = int(gate["runs"])
    if "minimum_all_runs_pass" in gate or "minimum_all_runs_pass_gain" in gate:
        candidate_all = _all_runs_pass(candidate_items, expected_runs)
        if "minimum_all_runs_pass" in gate:
            _add_minimum(
                checks,
                "completion.all_runs_pass",
                sum(candidate_all.values()),
                int(gate["minimum_all_runs_pass"]),
            )
        if "minimum_all_runs_pass_gain" in gate:
            if baseline is None:
                raise EvaluationInputError("all-runs-pass gain requires a baseline")
            baseline_all = _all_runs_pass(baseline_items, expected_runs)
            if set(candidate_all) != set(baseline_all):
                raise EvaluationInputError("all-runs-pass gain requires identical task clusters")
            gain = sum(candidate_all.values()) - sum(baseline_all.values())
            _add_minimum(
                checks,
                "completion.all_runs_pass_gain",
                gain,
                int(gate["minimum_all_runs_pass_gain"]),
            )

    domain_all_floor = gate.get("minimum_domain_all_runs_pass", {})
    if isinstance(domain_all_floor, Mapping) and domain_all_floor:
        candidate_all = _all_runs_pass(candidate_items, expected_runs)
        for domain, minimum in domain_all_floor.items():
            actual = sum(
                value for (cluster_domain, _), value in candidate_all.items() if cluster_domain == domain
            )
            _add_minimum(checks, f"completion.{domain}.all_runs_pass", actual, int(minimum))

    domain_ux_floor = gate.get("minimum_domain_ux", {})
    if isinstance(domain_ux_floor, Mapping):
        for domain, minimum in domain_ux_floor.items():
            actual = _mean([item for item in candidate_items if item.domain == domain], "ux")
            _add_minimum(checks, f"ux.{domain}", round(actual, 8), float(minimum))

    bootstrap = gate.get("cluster_bootstrap")
    if isinstance(bootstrap, Mapping):
        if baseline is None:
            raise EvaluationInputError("cluster bootstrap requires a baseline")
        clustered = _cluster_deltas(baseline, candidate)
        groups: list[tuple[str, list[float]]]
        if bool(bootstrap.get("by_domain", False)):
            groups = [
                (
                    domain,
                    [value for (cluster_domain, _), value in clustered if cluster_domain == domain],
                )
                for domain in domains
            ]
        else:
            groups = [("overall", [value for _, value in clustered])]
        for offset, (label, deltas) in enumerate(groups):
            lower_bound = cluster_bootstrap_lower_bound(
                deltas,
                confidence=float(bootstrap["confidence"]),
                iterations=int(bootstrap["iterations"]),
                seed=int(bootstrap["seed"]) + offset,
            )
            name = (
                f"completion.{label}.cluster_bootstrap_lower_bound"
                if label != "overall"
                else "completion.cluster_bootstrap_lower_bound"
            )
            _add_minimum(
                checks,
                name,
                round(lower_bound, 8),
                float(bootstrap["minimum_lower_bound"]),
                f"one-sided {float(bootstrap['confidence']):.0%} percentile interval",
            )

    passed = all(check.passed for check in checks)
    return {
        "schema_version": "1.0.0",
        "gate": gate_name,
        "passed": passed,
        "candidate_root": str(candidate.root),
        "baseline_root": str(baseline.root) if baseline is not None else None,
        "summary": {
            "candidate_observations": len(candidate_items),
            "candidate_completion": candidate_completion,
            "candidate_state_requirements": candidate_state,
            "candidate_task_requirements": candidate_task,
            "failed_checks": sum(not check.passed for check in checks),
        },
        "checks": [check.as_dict() for check in checks],
    }


def _print_report(report: Mapping[str, Any]) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    print(f"{status} {report['gate']} gate")
    for check in report["checks"]:
        marker = "PASS" if check["passed"] else "FAIL"
        detail = f" ({check['detail']})" if check.get("detail") else ""
        print(
            f"[{marker}] {check['name']}: actual={check['actual']!r}, "
            f"expected={check['expected']!r}{detail}"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", required=True, choices=("dev", "lockbox", "paired150", "official750"))
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--router", type=Path)
    parser.add_argument("--state-bench-root", type=Path, default=DEFAULT_STATE_BENCH_ROOT)
    parser.add_argument("--runner", type=Path, default=DEFAULT_RUNNER)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text checklist.")
    parser.add_argument("--output", type=Path, help="Also write the JSON report to this path.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.config.resolve(strict=True) != DEFAULT_CONFIG.resolve(strict=True):
            raise EvaluationInputError(
                "promotion thresholds are frozen; --config must reference "
                f"{DEFAULT_CONFIG}"
            )
        config = json.loads(args.config.read_text(encoding="utf-8"))
        contract = config.get("benchmark_contract")
        if not isinstance(contract, Mapping):
            raise EvaluationInputError("configuration is missing benchmark_contract")
        domains = [str(domain) for domain in contract.get("domains", [])]
        candidate = load_dataset(args.candidate, domains)
        baseline = load_dataset(args.baseline, domains) if args.baseline else None
        if args.gate == "official750" and baseline is not None:
            raise EvaluationInputError("official750 is candidate-only; do not pass --baseline")
        artifact_policy = config.get("artifact_policy")
        if not isinstance(artifact_policy, Mapping):
            raise EvaluationInputError("configuration is missing artifact_policy")
        kind = (
            "optimizer80"
            if args.gate in set(artifact_policy.get("optimizer80_stages", []))
            else "full100"
        )
        memory_path = args.memory or (REPO_ROOT / str(artifact_policy[f"{kind}_memory"]))
        router_path = args.router or (REPO_ROOT / str(artifact_policy[f"{kind}_router"]))
        report = evaluate_gate(
            gate_name=args.gate,
            config=config,
            candidate=candidate,
            baseline=baseline,
        )
        execution_checks, candidate_stage = evaluate_execution_contract(
            gate_name=args.gate,
            config=config,
            candidate=candidate,
            baseline=baseline,
            memory_path=memory_path,
            router_path=router_path,
            state_bench_root=args.state_bench_root,
            runner_path=args.runner,
        )
        report["checks"].extend(check.as_dict() for check in execution_checks)
        report["passed"] = bool(report["passed"]) and all(
            check.passed for check in execution_checks
        )
        report["candidate_router_stage"] = candidate_stage
        report["memory_path"] = str(memory_path.resolve())
        report["router_path"] = str(router_path.resolve())
        report["state_bench_root"] = str(args.state_bench_root.resolve())
        report["runner_path"] = str(args.runner.resolve())
        report["summary"]["failed_checks"] = sum(
            not bool(check["passed"]) for check in report["checks"]
        )
    except (OSError, UnicodeError, json.JSONDecodeError, EvaluationInputError) as exc:
        print(f"evaluation input error: {exc}", file=sys.stderr)
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
