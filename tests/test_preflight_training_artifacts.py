from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.preflight_training_artifacts import (  # noqa: E402
    DOMAINS,
    EXPECTED_ROUTER_INPUTS,
    REQUIRED_EXCLUDED_SOURCES,
    SPLIT_SEED,
    TrainingArtifactPreflightError,
    _git as artifact_git,
    file_sha256,
    load_independent_optimizer_splits,
    load_train_inventory_evidence,
    main,
    preflight_training_artifacts,
    train_content_manifest_sha256,
)


def test_git_path_output_is_decoded_as_utf8(tmp_path: Path) -> None:
    repository = tmp_path / "AI科研" / "repository"
    repository.mkdir(parents=True)
    _git(repository, "init")

    assert Path(artifact_git(repository, "rev-parse", "--show-toplevel")) == repository.resolve()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


def _fixture_roots(tmp_path: Path) -> dict[str, Any]:
    repository = tmp_path / "repository"
    state_bench = tmp_path / "STATE-Bench"
    repository.mkdir()
    train_ids: dict[str, list[str]] = {}
    dev_manifest: dict[str, Any] = {
        "seed": "fixture-seed",
        "method": "fixture deterministic development panel",
    }
    for domain in DOMAINS:
        ids = [f"{domain}-train-{index:03d}" for index in range(100)]
        train_ids[domain] = ids
        dev_manifest[domain] = ids[:10]
        for index, task_id in enumerate(ids):
            _write_json(
                state_bench
                / "datasets"
                / "train_task_trajectories"
                / domain
                / f"{task_id}.json",
                {"conversation": [], "fixture_index": index},
            )
        _write_json(
            state_bench
            / "state_bench"
            / "domains"
            / domain
            / "splits"
            / "train_test.json",
            {
                "splits": {
                    "train": ids,
                    "test": [f"{domain}-test-{index:03d}" for index in range(50)],
                }
            },
        )
    dev_path = repository / "configs" / "workflow_router_dev_ids.json"
    _write_json(dev_path, dev_manifest)
    _git(repository, "init")
    _git(repository, "config", "user.email", "preflight@example.invalid")
    _git(repository, "config", "user.name", "Preflight Fixture")
    _git(repository, "add", "configs/workflow_router_dev_ids.json")
    _git(repository, "commit", "-m", "freeze split manifest")
    return {
        "repository": repository,
        "state_bench": state_bench,
        "dev_path": dev_path,
        "train_ids": train_ids,
    }


def _build_pair(fixture: dict[str, Any], kind: str) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    repository = fixture["repository"]
    state_bench = fixture["state_bench"]
    inventory = load_train_inventory_evidence(state_bench, DOMAINS)
    splits, dev_sha256 = load_independent_optimizer_splits(
        state_bench_root=state_bench,
        repository_root=repository,
        domains=DOMAINS,
    )
    all_ids = {domain: set(inventory["files"][domain]) for domain in DOMAINS}
    allowed = {
        domain: set(splits[domain]["optimizer"]) if kind == "optimizer80" else all_ids[domain]
        for domain in DOMAINS
    }
    content_sha256, selected_count = train_content_manifest_sha256(inventory["files"], allowed)
    expected_per_domain = 80 if kind == "optimizer80" else 100
    cards = []
    overlap = {domain: {"dev": [], "lockbox": []} for domain in DOMAINS}
    for domain in DOMAINS:
        if kind == "optimizer80":
            sources = [splits[domain]["optimizer"][0]]
        else:
            sources = [
                splits[domain]["dev"][0],
                splits[domain]["lockbox"][0],
                splits[domain]["optimizer"][0],
            ]
            overlap[domain] = {
                "dev": [splits[domain]["dev"][0]],
                "lockbox": [splits[domain]["lockbox"][0]],
            }
        cards.append(
            {
                "id": f"{domain}:fixture:0",
                "domain": domain,
                "source_tasks": sources,
            }
        )
    split_summary = (
        {
            domain: {
                split: sorted(splits[domain][split])
                for split in ("dev", "lockbox", "optimizer")
            }
            for domain in DOMAINS
        }
        if kind == "optimizer80"
        else {domain: {"all": sorted(all_ids[domain])} for domain in DOMAINS}
    )
    memory = {
        "version": 1,
        "provenance": {
            "task_split": "optimizer" if kind == "optimizer80" else "all",
            "task_manifest_sha256": dev_sha256 if kind == "optimizer80" else None,
            "task_manifest_method": (
                "fixture deterministic development panel"
                if kind == "optimizer80"
                else "all_fixed_train_trajectories"
            ),
            "train_inventory_sha256": inventory["inventory_sha256"],
            "inventory_file_count": 300,
            "trajectory_manifest_sha256": content_sha256,
            "read_content_manifest_sha256": content_sha256,
            "split_seed": SPLIT_SEED,
            "selected_counts": {domain: expected_per_domain for domain in DOMAINS},
            "split_summary": split_summary,
            "learning_input": "datasets/train_task_trajectories only",
            "excluded_sources": sorted(REQUIRED_EXCLUDED_SOURCES),
        },
        "cards": cards,
        "stats": {
            domain: {"trajectories": expected_per_domain, "families": 1, "cards": 1}
            for domain in DOMAINS
        },
    }
    memory_path = repository / f"memory-{kind}.json"
    router_path = repository / f"router-{kind}.json"
    _write_json(memory_path, memory)
    router = {
        "schema_version": "2.0.0",
        "source_memory_sha256": file_sha256(memory_path),
        "splits": splits,
        "provenance": {
            "learning_inputs": EXPECTED_ROUTER_INPUTS,
            "excluded_sources": sorted(REQUIRED_EXCLUDED_SOURCES),
            "train_manifest_sha256": content_sha256,
            "train_inventory_sha256": inventory["inventory_sha256"],
            "read_content_manifest_sha256": content_sha256,
            "train_file_count": selected_count,
            "memory_sha256": file_sha256(memory_path),
            "dev_manifest_sha256": dev_sha256,
            "split_seed": SPLIT_SEED,
            "memory_training_split": "optimizer" if kind == "optimizer80" else "all",
            "memory_trajectory_counts": {
                domain: expected_per_domain for domain in DOMAINS
            },
            "lockbox_independent": kind == "optimizer80",
            "source_task_overlap": (
                {domain: {"dev": [], "lockbox": []} for domain in DOMAINS}
                if kind == "optimizer80"
                else overlap
            ),
            "api_calls": 0,
        },
    }
    _write_json(router_path, router)
    return memory_path, router_path, memory, router


@pytest.mark.parametrize("kind,selected", [("optimizer80", 240), ("full100", 300)])
def test_accepts_independently_bound_train_only_artifacts(
    tmp_path: Path, kind: str, selected: int
) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, _router_payload = _build_pair(fixture, kind)
    report = preflight_training_artifacts(
        kind=kind,
        memory_path=memory,
        router_path=router,
        state_bench_root=fixture["state_bench"],
        repository_root=fixture["repository"],
    )
    assert report["passed"] is True
    assert report["inventory_file_count"] == 300
    assert report["selected_file_count"] == selected
    assert all(not values for values in report["official_test_overlap"].values())


def test_rejects_changed_train_content_after_artifact_build(tmp_path: Path) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, _router_payload = _build_pair(fixture, "optimizer80")
    task_id = fixture["train_ids"][DOMAINS[0]][-1]
    _write_json(
        fixture["state_bench"]
        / "datasets"
        / "train_task_trajectories"
        / DOMAINS[0]
        / f"{task_id}.json",
        {"conversation": [{"role": "user", "content": "changed"}]},
    )
    with pytest.raises(TrainingArtifactPreflightError, match="trajectory_manifest_sha256"):
        preflight_training_artifacts(
            kind="optimizer80",
            memory_path=memory,
            router_path=router,
            state_bench_root=fixture["state_bench"],
            repository_root=fixture["repository"],
        )


def test_rejects_optimizer_card_using_held_out_train_id(tmp_path: Path) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, memory_payload, _router_payload = _build_pair(fixture, "optimizer80")
    memory_payload["cards"][0]["source_tasks"] = [
        fixture["train_ids"][DOMAINS[0]][0]
    ]
    _write_json(memory, memory_payload)
    with pytest.raises(TrainingArtifactPreflightError, match="disallowed task IDs"):
        preflight_training_artifacts(
            kind="optimizer80",
            memory_path=memory,
            router_path=router,
            state_bench_root=fixture["state_bench"],
            repository_root=fixture["repository"],
        )


def test_rejects_train_id_in_official_test_split(tmp_path: Path) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, _router_payload = _build_pair(fixture, "full100")
    domain = DOMAINS[0]
    split_path = (
        fixture["state_bench"]
        / "state_bench"
        / "domains"
        / domain
        / "splits"
        / "train_test.json"
    )
    split = json.loads(split_path.read_text(encoding="utf-8"))
    split["splits"]["test"][0] = fixture["train_ids"][domain][0]
    _write_json(split_path, split)
    with pytest.raises(TrainingArtifactPreflightError, match="overlap the official test split"):
        preflight_training_artifacts(
            kind="full100",
            memory_path=memory,
            router_path=router,
            state_bench_root=fixture["state_bench"],
            repository_root=fixture["repository"],
        )


@pytest.mark.parametrize("artifact_location", ["memory_value", "router_key"])
def test_rejects_complete_cross_domain_test_id_in_any_json_string_or_key(
    tmp_path: Path, artifact_location: str
) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, memory_payload, router_payload = _build_pair(fixture, "full100")
    leaked_id = f"{DOMAINS[1]}-test-000"
    if artifact_location == "memory_value":
        # The card belongs to shopping while the leaked ID belongs to travel;
        # scanning must be global rather than scoped to the card's domain.
        memory_payload["cards"][0]["note"] = f"foreign held-out task: {leaked_id}."
        _write_json(memory, memory_payload)
    else:
        router_payload[leaked_id] = {"note": "leaked through a field name"}
        _write_json(router, router_payload)
    with pytest.raises(TrainingArtifactPreflightError, match=leaked_id):
        preflight_training_artifacts(
            kind="full100",
            memory_path=memory,
            router_path=router,
            state_bench_root=fixture["state_bench"],
            repository_root=fixture["repository"],
        )


def test_longer_identifiers_containing_test_id_substrings_do_not_false_positive(
    tmp_path: Path,
) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, memory_payload, router_payload = _build_pair(fixture, "full100")
    test_id = f"{DOMAINS[2]}-test-000"
    memory_payload["safe_note"] = (
        f"prefix{test_id}suffix {test_id}_suffix prefix-{test_id}"
    )
    _write_json(memory, memory_payload)
    memory_hash = file_sha256(memory)
    router_payload["source_memory_sha256"] = memory_hash
    router_payload["provenance"]["memory_sha256"] = memory_hash
    _write_json(router, router_payload)
    report = preflight_training_artifacts(
        kind="full100",
        memory_path=memory,
        router_path=router,
        state_bench_root=fixture["state_bench"],
        repository_root=fixture["repository"],
    )
    assert report["passed"] is True


def test_rejects_router_provenance_or_memory_hash_tampering(tmp_path: Path) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, router_payload = _build_pair(fixture, "full100")
    router_payload["provenance"]["train_file_count"] = 299
    _write_json(router, router_payload)
    with pytest.raises(TrainingArtifactPreflightError, match="train_file_count"):
        preflight_training_artifacts(
            kind="full100",
            memory_path=memory,
            router_path=router,
            state_bench_root=fixture["state_bench"],
            repository_root=fixture["repository"],
        )


def test_cli_success_emits_a_deterministic_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, _router_payload = _build_pair(fixture, "full100")
    code = main(
        [
            "--kind",
            "full100",
            "--memory",
            str(memory),
            "--router",
            str(router),
            "--state-bench-root",
            str(fixture["state_bench"]),
            "--repository-root",
            str(fixture["repository"]),
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["passed"] is True
    assert report["kind"] == "full100"
    assert report["selected_file_count"] == 300


def test_cli_fails_closed_when_dev_manifest_differs_from_head(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixture = _fixture_roots(tmp_path)
    memory, router, _memory_payload, _router_payload = _build_pair(fixture, "optimizer80")
    changed = json.loads(fixture["dev_path"].read_text(encoding="utf-8"))
    changed["method"] = "uncommitted replacement"
    _write_json(fixture["dev_path"], changed)
    code = main(
        [
            "--kind",
            "optimizer80",
            "--memory",
            str(memory),
            "--router",
            str(router),
            "--state-bench-root",
            str(fixture["state_bench"]),
            "--repository-root",
            str(fixture["repository"]),
        ]
    )
    assert code == 2
    assert "not identical to the committed HEAD version" in capsys.readouterr().err
