from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_lattice_pilot import analyze


def test_analyzer_recovers_antagonistic_pair(tmp_path: Path) -> None:
    completion = {0: False, 1: True, 2: True, 3: False, 4: False, 5: False, 6: False, 7: False}
    for mask in range(8):
        run_dir = tmp_path / f"mask{mask}" / "run1"
        run_dir.mkdir(parents=True)
        record = {
            "task_id": "train-task",
            "task_completion_pass": completion[mask],
            "state_requirements_met": True,
            "task_requirements_met": completion[mask],
            "ux_score": 4.0,
            "total_tokens": 100 + mask,
        }
        (run_dir / "train-task.json").write_text(json.dumps(record), encoding="utf-8")

    report = analyze(tmp_path)
    assert report["complete_blocks"] == 1
    assert report["mobius"]["completion"]["3"]["mean"] == -2.0
    assert report["mobius"]["completion"]["3"]["nonzero_rate"] == 1.0
    assert report["interaction_mass"]["completion"] > 0.0
    assert report["complete_tasks"] == 1
    assert report["runs_per_task"] == [1]
    assert report["mobius_on_task_means"]["completion"]["3"]["mean"] == -2.0
    assert report["within_arm_stochasticity"]["completion"]["0"][
        "max_task_range"
    ] == 0.0
    assert report["contrast_vs_empty"]["completion"]["1"]["mean"] == 1.0
    assert report["contrast_vs_empty"]["completion"]["1"]["wins"] == 1
