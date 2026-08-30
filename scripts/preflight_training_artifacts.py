"""Offline provenance preflight for selective-PWM training artifacts.

The preflight deliberately derives its expectations from only three local,
read-only inputs:

* ``STATE-Bench/datasets/train_task_trajectories/<domain>/*.json``;
* the committed ID-only ``configs/workflow_router_dev_ids.json``; and
* the official ``train_test.json`` split manifests (IDs only).

It neither imports an LLM client nor performs any network access.  A failure is
fatal because an artifact that cannot prove its train-only scope must never be
used to start an evaluation run.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

try:
    from scripts.evaluate_gate import (
        REQUIRED_EXCLUDED_SOURCES,
        SPLIT_SEED,
        EvaluationInputError,
        file_sha256,
        load_independent_optimizer_splits,
        load_train_inventory_evidence,
        train_content_manifest_sha256,
    )
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_gate import (  # type: ignore[no-redef]
        REQUIRED_EXCLUDED_SOURCES,
        SPLIT_SEED,
        EvaluationInputError,
        file_sha256,
        load_independent_optimizer_splits,
        load_train_inventory_evidence,
        train_content_manifest_sha256,
    )


DOMAINS = ("shopping_assistant", "travel", "customer_support")
DEV_MANIFEST_RELATIVE = Path("configs/workflow_router_dev_ids.json")
TRAIN_SPLIT_RELATIVE = Path("state_bench/domains/{domain}/splits/train_test.json")
EXPECTED_ROUTER_INPUTS = [
    "datasets/train_task_trajectories/<domain>/*.json",
    "v1 process_workflows.json",
]


class TrainingArtifactPreflightError(ValueError):
    """Raised when an artifact cannot prove its train-only provenance."""


def _load_object(path: Path, *, label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise TrainingArtifactPreflightError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise TrainingArtifactPreflightError(f"{label} must be a JSON object: {path}")
    return value


def _nested(value: Any, *keys: str, default: Any = None) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def _git(repository_root: Path, *args: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            check=True,
            capture_output=True,
            text=False,
            timeout=15,
        )
        # Git emits path output as UTF-8 on the supported Windows setup.  Do
        # not let Python's locale codec replace non-ASCII repository names,
        # otherwise a valid checkout such as ``AI科研`` fails the root check.
        return completed.stdout.decode("utf-8", errors="strict").strip()
    except (OSError, UnicodeError, subprocess.SubprocessError) as exc:
        raise TrainingArtifactPreflightError(
            f"cannot verify committed split manifest with git {' '.join(args)!r}: {exc}"
        ) from exc


def _require_committed_dev_manifest(repository_root: Path) -> tuple[Path, str]:
    root = repository_root.resolve()
    manifest = (root / DEV_MANIFEST_RELATIVE).resolve()
    try:
        relative = manifest.relative_to(root).as_posix()
    except ValueError as exc:
        raise TrainingArtifactPreflightError("development split manifest escapes repository root") from exc
    if relative != DEV_MANIFEST_RELATIVE.as_posix() or not manifest.is_file():
        raise TrainingArtifactPreflightError(
            f"missing development split manifest: {root / DEV_MANIFEST_RELATIVE}"
        )
    top_level = Path(_git(root, "rev-parse", "--show-toplevel")).resolve()
    if top_level != root:
        raise TrainingArtifactPreflightError(
            f"repository root mismatch: requested {root}, git reports {top_level}"
        )
    _git(root, "ls-files", "--error-unmatch", "--", relative)
    working_blob = _git(root, "hash-object", "--", relative)
    committed_blob = _git(root, "rev-parse", f"HEAD:{relative}")
    if not working_blob or working_blob != committed_blob:
        raise TrainingArtifactPreflightError(
            "configs/workflow_router_dev_ids.json is not identical to the committed HEAD version"
        )
    return manifest, file_sha256(manifest)


def _normal_strings(value: Any, *, label: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TrainingArtifactPreflightError(f"{label} must be a list of task-ID strings")
    if len(value) != len(set(value)):
        raise TrainingArtifactPreflightError(f"{label} contains duplicate task IDs")
    return list(value)


def _official_test_ids(state_bench_root: Path) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for domain in DOMAINS:
        path = state_bench_root / Path(str(TRAIN_SPLIT_RELATIVE).format(domain=domain))
        payload = _load_object(path, label=f"{domain} official train/test split")
        values = _normal_strings(
            _nested(payload, "splits", "test"), label=f"{domain} official test split"
        )
        if len(values) != 50:
            raise TrainingArtifactPreflightError(
                f"{domain} official test split must contain exactly 50 unique IDs"
            )
        result[domain] = set(values)
    return result


def _test_id_references(
    value: Any,
    official_test_ids: Mapping[str, set[str]],
) -> list[dict[str, Any]]:
    """Find complete official-test ID tokens in every JSON key and string value.

    STATE-Bench task IDs are filename-safe identifiers.  Alphanumerics,
    underscores, and hyphens therefore count as continuation characters.  The
    boundary rule catches an ID surrounded by prose/path punctuation but does
    not flag an ordinary substring inside a longer identifier.
    """

    domains_by_id: dict[str, set[str]] = {}
    for domain, task_ids in official_test_ids.items():
        for task_id in task_ids:
            domains_by_id.setdefault(str(task_id), set()).add(str(domain))
    if not domains_by_id:
        raise TrainingArtifactPreflightError("official test ID inventory is empty")
    alternatives = "|".join(
        re.escape(task_id)
        for task_id in sorted(domains_by_id, key=lambda item: (-len(item), item))
    )
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_-])(?:{alternatives})(?![A-Za-z0-9_-])"
    )
    found: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def scan_text(text: str, *, path: str, location: str) -> None:
        for match in pattern.finditer(text):
            task_id = match.group(0)
            key = (path, location, task_id)
            if key in seen:
                continue
            seen.add(key)
            found.append(
                {
                    "path": path,
                    "location": location,
                    "test_id": task_id,
                    "test_domains": sorted(domains_by_id[task_id]),
                }
            )

    def visit(current: Any, path: str) -> None:
        if isinstance(current, Mapping):
            for key, child in current.items():
                key_text = str(key)
                child_path = f"{path}[{json.dumps(key_text, ensure_ascii=False)}]"
                scan_text(key_text, path=child_path, location="key")
                visit(child, child_path)
        elif isinstance(current, list):
            for index, child in enumerate(current):
                visit(child, f"{path}[{index}]")
        elif isinstance(current, str):
            scan_text(current, path=path, location="value")

    visit(value, "$")
    return found


def _exact_string_set(value: Any, expected: set[str], *, label: str) -> None:
    values = _normal_strings(value, label=label)
    if set(values) != expected or len(values) != len(expected):
        raise TrainingArtifactPreflightError(
            f"{label} mismatch: expected {sorted(expected)}, got {sorted(values)}"
        )


def _validate_memory(
    memory: Mapping[str, Any],
    *,
    kind: str,
    allowed_ids: Mapping[str, set[str]],
    all_train_ids: Mapping[str, set[str]],
    splits: Mapping[str, Mapping[str, list[str]]],
    inventory_sha256: str,
    inventory_count: int,
    content_manifest_sha256: str,
    dev_manifest_sha256: str,
    dev_manifest_method: str,
) -> dict[str, dict[str, list[str]]]:
    expected_per_domain = 80 if kind == "optimizer80" else 100
    provenance = memory.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingArtifactPreflightError("memory.provenance is missing or not an object")
    expected_split = "optimizer" if kind == "optimizer80" else "all"
    expected_selector_hash = dev_manifest_sha256 if kind == "optimizer80" else None
    expected_selector_method = (
        dev_manifest_method if kind == "optimizer80" else "all_fixed_train_trajectories"
    )
    expected = {
        "task_split": expected_split,
        "task_manifest_sha256": expected_selector_hash,
        "train_inventory_sha256": inventory_sha256,
        "inventory_file_count": inventory_count,
        "trajectory_manifest_sha256": content_manifest_sha256,
        "read_content_manifest_sha256": content_manifest_sha256,
        "learning_input": "datasets/train_task_trajectories only",
        "split_seed": SPLIT_SEED,
    }
    for field, wanted in expected.items():
        if provenance.get(field) != wanted:
            raise TrainingArtifactPreflightError(
                f"memory.provenance.{field} mismatch: expected {wanted!r}, "
                f"got {provenance.get(field)!r}"
            )
    if provenance.get("task_manifest_method") != expected_selector_method:
        raise TrainingArtifactPreflightError(
            f"memory.provenance.task_manifest_method must equal {expected_selector_method!r}"
        )
    if set(map(str, provenance.get("excluded_sources") or [])) != set(
        REQUIRED_EXCLUDED_SOURCES
    ):
        raise TrainingArtifactPreflightError(
            "memory.provenance.excluded_sources does not match the frozen exclusion set"
        )
    selected_counts = provenance.get("selected_counts")
    expected_counts = {domain: expected_per_domain for domain in DOMAINS}
    if not isinstance(selected_counts, Mapping) or dict(selected_counts) != expected_counts:
        raise TrainingArtifactPreflightError(
            f"memory.provenance.selected_counts must equal {expected_counts}"
        )
    stats = memory.get("stats")
    if not isinstance(stats, Mapping) or set(stats) != set(DOMAINS):
        raise TrainingArtifactPreflightError("memory.stats must contain exactly the three domains")
    for domain in DOMAINS:
        if _nested(stats, domain, "trajectories") != expected_per_domain:
            raise TrainingArtifactPreflightError(
                f"memory.stats.{domain}.trajectories must equal {expected_per_domain}"
            )
    split_summary = provenance.get("split_summary")
    if not isinstance(split_summary, Mapping) or set(split_summary) != set(DOMAINS):
        raise TrainingArtifactPreflightError(
            "memory.provenance.split_summary must contain exactly the three domains"
        )
    for domain in DOMAINS:
        domain_summary = split_summary.get(domain)
        if not isinstance(domain_summary, Mapping):
            raise TrainingArtifactPreflightError(f"memory split summary missing {domain}")
        if kind == "optimizer80":
            if set(domain_summary) != {"dev", "lockbox", "optimizer"}:
                raise TrainingArtifactPreflightError(
                    f"optimizer80 memory split summary has invalid fields for {domain}"
                )
            for split in ("dev", "lockbox", "optimizer"):
                _exact_string_set(
                    domain_summary.get(split),
                    set(splits[domain][split]),
                    label=f"memory.provenance.split_summary.{domain}.{split}",
                )
        else:
            if set(domain_summary) != {"all"}:
                raise TrainingArtifactPreflightError(
                    f"full100 memory split summary has invalid fields for {domain}"
                )
            _exact_string_set(
                domain_summary.get("all"),
                all_train_ids[domain],
                label=f"memory.provenance.split_summary.{domain}.all",
            )

    cards = memory.get("cards")
    if not isinstance(cards, list) or not cards:
        raise TrainingArtifactPreflightError("memory.cards must be a non-empty list")
    overlap = {domain: {"dev": set(), "lockbox": set()} for domain in DOMAINS}
    for index, card in enumerate(cards):
        if not isinstance(card, Mapping):
            raise TrainingArtifactPreflightError(f"memory.cards[{index}] is not an object")
        domain = str(card.get("domain", ""))
        if domain not in DOMAINS:
            raise TrainingArtifactPreflightError(
                f"memory.cards[{index}] has unsupported domain {domain!r}"
            )
        sources = _normal_strings(
            card.get("source_tasks"), label=f"memory.cards[{index}].source_tasks"
        )
        if not sources:
            raise TrainingArtifactPreflightError(
                f"memory.cards[{index}].source_tasks must not be empty"
            )
        outside = set(sources) - allowed_ids[domain]
        if outside:
            raise TrainingArtifactPreflightError(
                f"memory card {card.get('id', index)!r} references disallowed task IDs: "
                f"{sorted(outside)}"
            )
        for split in ("dev", "lockbox"):
            overlap[domain][split].update(set(sources) & set(splits[domain][split]))
    return {
        domain: {split: sorted(values) for split, values in domain_values.items()}
        for domain, domain_values in overlap.items()
    }


def _validate_router(
    router: Mapping[str, Any],
    *,
    memory_path: Path,
    kind: str,
    splits: Mapping[str, Mapping[str, list[str]]],
    actual_source_overlap: Mapping[str, Mapping[str, list[str]]],
    inventory_sha256: str,
    content_manifest_sha256: str,
    selected_file_count: int,
    dev_manifest_sha256: str,
) -> None:
    expected_per_domain = 80 if kind == "optimizer80" else 100
    expected_training_split = "optimizer" if kind == "optimizer80" else "all"
    expected_lockbox = kind == "optimizer80"
    memory_hash = file_sha256(memory_path)
    if router.get("source_memory_sha256") != memory_hash:
        raise TrainingArtifactPreflightError(
            "router.source_memory_sha256 does not match the memory artifact bytes"
        )
    router_splits = router.get("splits")
    if not isinstance(router_splits, Mapping) or set(router_splits) != set(DOMAINS):
        raise TrainingArtifactPreflightError("router.splits must contain exactly the three domains")
    for domain in DOMAINS:
        domain_splits = router_splits.get(domain)
        if not isinstance(domain_splits, Mapping) or set(domain_splits) != {
            "dev",
            "lockbox",
            "optimizer",
        }:
            raise TrainingArtifactPreflightError(f"router.splits.{domain} has invalid fields")
        for split in ("dev", "lockbox", "optimizer"):
            values = _normal_strings(
                domain_splits.get(split), label=f"router.splits.{domain}.{split}"
            )
            if values != list(splits[domain][split]):
                raise TrainingArtifactPreflightError(
                    f"router.splits.{domain}.{split} does not match independent recomputation"
                )

    provenance = router.get("provenance")
    if not isinstance(provenance, Mapping):
        raise TrainingArtifactPreflightError("router.provenance is missing or not an object")
    expected = {
        "memory_sha256": memory_hash,
        "memory_training_split": expected_training_split,
        "lockbox_independent": expected_lockbox,
        "dev_manifest_sha256": dev_manifest_sha256,
        "train_file_count": selected_file_count,
        "train_manifest_sha256": content_manifest_sha256,
        "read_content_manifest_sha256": content_manifest_sha256,
        "train_inventory_sha256": inventory_sha256,
        "split_seed": SPLIT_SEED,
        "api_calls": 0,
    }
    for field, wanted in expected.items():
        if provenance.get(field) != wanted:
            raise TrainingArtifactPreflightError(
                f"router.provenance.{field} mismatch: expected {wanted!r}, "
                f"got {provenance.get(field)!r}"
            )
    if provenance.get("learning_inputs") != EXPECTED_ROUTER_INPUTS:
        raise TrainingArtifactPreflightError(
            "router.provenance.learning_inputs does not match the train-only contract"
        )
    if set(map(str, provenance.get("excluded_sources") or [])) != set(
        REQUIRED_EXCLUDED_SOURCES
    ):
        raise TrainingArtifactPreflightError(
            "router.provenance.excluded_sources does not match the frozen exclusion set"
        )
    expected_counts = {domain: expected_per_domain for domain in DOMAINS}
    counts = provenance.get("memory_trajectory_counts")
    if not isinstance(counts, Mapping) or dict(counts) != expected_counts:
        raise TrainingArtifactPreflightError(
            f"router.provenance.memory_trajectory_counts must equal {expected_counts}"
        )
    expected_overlap = (
        {domain: {"dev": [], "lockbox": []} for domain in DOMAINS}
        if kind == "optimizer80"
        else {
            domain: {
                split: list(actual_source_overlap[domain][split])
                for split in ("dev", "lockbox")
            }
            for domain in DOMAINS
        }
    )
    if provenance.get("source_task_overlap") != expected_overlap:
        raise TrainingArtifactPreflightError(
            "router.provenance.source_task_overlap does not match memory card sources"
        )


def preflight_training_artifacts(
    *,
    kind: str,
    memory_path: Path,
    router_path: Path,
    state_bench_root: Path,
    repository_root: Path,
) -> dict[str, Any]:
    """Validate a frozen optimizer80 or full100 artifact pair without side effects."""

    if kind not in {"optimizer80", "full100"}:
        raise TrainingArtifactPreflightError("kind must be optimizer80 or full100")
    memory_path = memory_path.resolve()
    router_path = router_path.resolve()
    state_bench_root = state_bench_root.resolve()
    repository_root = repository_root.resolve()
    memory = _load_object(memory_path, label="memory artifact")
    router = _load_object(router_path, label="router artifact")
    manifest_path, dev_manifest_sha256 = _require_committed_dev_manifest(repository_root)
    dev_manifest = _load_object(manifest_path, label="committed development split manifest")
    dev_manifest_method = str(dev_manifest.get("method") or "sha256_dev_manifest")
    try:
        inventory = load_train_inventory_evidence(state_bench_root, DOMAINS)
        splits, helper_manifest_sha256 = load_independent_optimizer_splits(
            state_bench_root=state_bench_root,
            repository_root=repository_root,
            domains=DOMAINS,
        )
    except EvaluationInputError as exc:
        raise TrainingArtifactPreflightError(str(exc)) from exc
    if helper_manifest_sha256 != dev_manifest_sha256:
        raise TrainingArtifactPreflightError(
            "development split manifest changed during preflight"
        )
    all_train_ids = {
        domain: set(map(str, inventory["files"][domain])) for domain in DOMAINS
    }
    official_test = _official_test_ids(state_bench_root)
    overlap = {
        domain: sorted(all_train_ids[domain] & official_test[domain]) for domain in DOMAINS
    }
    if any(overlap.values()):
        raise TrainingArtifactPreflightError(
            f"train trajectory IDs overlap the official test split: {overlap}"
        )
    leaked_test_ids = {
        "memory": _test_id_references(memory, official_test),
        "router": _test_id_references(router, official_test),
    }
    if any(leaked_test_ids.values()):
        first = {
            artifact: references[:10]
            for artifact, references in leaked_test_ids.items()
            if references
        }
        raise TrainingArtifactPreflightError(
            "memory/router contains complete official test ID token(s): "
            + json.dumps(first, ensure_ascii=False, sort_keys=True)
        )
    allowed_ids = {
        domain: (
            set(splits[domain]["optimizer"])
            if kind == "optimizer80"
            else all_train_ids[domain]
        )
        for domain in DOMAINS
    }
    try:
        content_manifest_sha256, selected_file_count = train_content_manifest_sha256(
            inventory["files"], allowed_ids
        )
    except EvaluationInputError as exc:
        raise TrainingArtifactPreflightError(str(exc)) from exc
    expected_file_count = 240 if kind == "optimizer80" else 300
    if inventory["file_count"] != 300 or selected_file_count != expected_file_count:
        raise TrainingArtifactPreflightError(
            f"unexpected train inventory/content count: inventory={inventory['file_count']}, "
            f"selected={selected_file_count}"
        )
    actual_overlap = _validate_memory(
        memory,
        kind=kind,
        allowed_ids=allowed_ids,
        all_train_ids=all_train_ids,
        splits=splits,
        inventory_sha256=str(inventory["inventory_sha256"]),
        inventory_count=int(inventory["file_count"]),
        content_manifest_sha256=content_manifest_sha256,
        dev_manifest_sha256=dev_manifest_sha256,
        dev_manifest_method=dev_manifest_method,
    )
    _validate_router(
        router,
        memory_path=memory_path,
        kind=kind,
        splits=splits,
        actual_source_overlap=actual_overlap,
        inventory_sha256=str(inventory["inventory_sha256"]),
        content_manifest_sha256=content_manifest_sha256,
        selected_file_count=selected_file_count,
        dev_manifest_sha256=dev_manifest_sha256,
    )
    return {
        "schema_version": "1.0.0",
        "passed": True,
        "kind": kind,
        "memory_sha256": file_sha256(memory_path),
        "router_sha256": file_sha256(router_path),
        "dev_manifest_sha256": dev_manifest_sha256,
        "train_inventory_sha256": inventory["inventory_sha256"],
        "train_content_manifest_sha256": content_manifest_sha256,
        "inventory_file_count": inventory["file_count"],
        "selected_file_count": selected_file_count,
        "selected_counts": {
            domain: len(allowed_ids[domain]) for domain in DOMAINS
        },
        "official_test_overlap": overlap,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", required=True, choices=("optimizer80", "full100"))
    parser.add_argument("--memory", required=True, type=Path)
    parser.add_argument("--router", required=True, type=Path)
    parser.add_argument("--state-bench-root", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = preflight_training_artifacts(
            kind=args.kind,
            memory_path=args.memory,
            router_path=args.router,
            state_bench_root=args.state_bench_root,
            repository_root=args.repository_root,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TrainingArtifactPreflightError) as exc:
        print(f"training artifact preflight error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
