#!/usr/bin/env bash
# Fresh train-only confirmation of fixed rank-3 against original full PWM.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bench_root="${STATE_BENCH_ROOT:-/root/Agentic/STATE-Bench}"
python_bin="${STATE_BENCH_PYTHON:-${bench_root}/.venv/bin/python}"
output_root="${CONFIRM_OUTPUT_ROOT:-/root/autodl-tmp/fixed_rank3_confirmation_v1}"
workers="${WORKERS:-3}"
salt="fixed-rank3-confirmation-v1"

unset STATE_BENCH_AGENT_BASE_URL STATE_BENCH_AGENT_API_KEY STATE_BENCH_AGENT_MODEL
unset STATE_BENCH_AGENT_MAX_TOKENS STATE_BENCH_AGENT_TIMEOUT_SECONDS
unset STATE_BENCH_AGENT_MAX_RETRIES STATE_BENCH_EVAL_ENDPOINT
unset STATE_BENCH_EVAL_DEPLOYMENTS STATE_BENCH_EVAL_API_KEY

export PYTHONPATH="${repo_root}:${bench_root}"
export STATE_BENCH_MEMORY_PATH="${bench_root}/artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"
export STATE_BENCH_MEMORY_MODE=hybrid
export PYTHONUTF8=1

tasks="$(${python_bin} - "${salt}" <<'PY'
import hashlib
import sys
from state_bench.protocol import load_default_protocol, load_split_task_ids

salt = sys.argv[1]
protocol = load_default_protocol()
task_ids = load_split_task_ids("shopping_assistant", "train", protocol.split_version)
ordered = sorted(
    task_ids,
    key=lambda task_id: hashlib.sha256(f"{salt}:{task_id}".encode()).digest(),
)
print(",".join(ordered[:50]))
PY
)"

run_arm() {
  local arm="$1"
  local agent_class="$2"
  local arm_root="${output_root}/${arm}/shopping_assistant"
  mkdir -p "${arm_root}/run1"
  for attempt in 1 2 3; do
    local missing
    missing="$(${python_bin} - "${arm_root}/run1" "${tasks}" <<'PY'
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
expected = sys.argv[2].split(",")
present = {path.stem for path in run_dir.glob("*.json")}
print(",".join(task_id for task_id in expected if task_id not in present))
PY
)"
    if [[ -z "${missing}" ]]; then
      break
    fi
    echo "arm=${arm} attempt=${attempt} remaining=$(awk -F, '{print NF}' <<<"${missing}")"
    "${python_bin}" -m state_bench.scripts.run_batch \
      --domain shopping_assistant \
      --tasks "${missing}" \
      --output-dir "${arm_root}" \
      --num-runs 1 \
      --num-workers "${workers}" \
      --agent-class "${agent_class}" \
      --agent-client-class OpenCodeLLMClient \
      --agent-model-name gpt-5.4 \
      --retrieve-learnings-top-k 3 \
      --score-reasoning-effort high
  done
}

cd "${repo_root}"
run_arm baseline ProcessWorkflowMemoryAgent
run_arm candidate FixedRank3PWMAgent

"${python_bin}" - "${output_root}" "${tasks}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
from statistics import mean

root = Path(sys.argv[1])
task_ids = sys.argv[2].split(",")
summary = {
    "schema_version": "fixed_rank3_confirmation_v1",
    "split": "train",
    "selection": "salted_sha256_first_50",
    "task_manifest_sha256": hashlib.sha256("\n".join(task_ids).encode()).hexdigest(),
    "arms": {},
}
records_by_arm = {}
for arm in ("baseline", "candidate"):
    records = {}
    for path in (root / arm / "shopping_assistant" / "run1").glob("*.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record.get("task_id", ""))] = record
    missing = sorted(set(task_ids) - set(records))
    unscored = sorted(
        task_id
        for task_id, record in records.items()
        if any(record.get(key) is None for key in (
            "task_completion_pass",
            "state_requirements_met",
            "task_requirements_met",
            "ux_score",
        ))
    )
    if missing or unscored:
        raise SystemExit(f"{arm} incomplete: missing={missing}, unscored={unscored}")
    selected = [records[task_id] for task_id in task_ids]
    records_by_arm[arm] = selected
    summary["arms"][arm] = {
        "n": len(selected),
        "completion": sum(bool(item["task_completion_pass"]) for item in selected),
        "state": sum(bool(item["state_requirements_met"]) for item in selected),
        "task": sum(bool(item["task_requirements_met"]) for item in selected),
        "mean_ux": mean(float(item["ux_score"]) for item in selected),
        "mean_total_tokens": mean(float(item.get("total_tokens") or 0) for item in selected),
    }

baseline = summary["arms"]["baseline"]
candidate = summary["arms"]["candidate"]
summary["delta"] = {
    metric: candidate[metric] - baseline[metric]
    for metric in ("completion", "state", "task", "mean_ux", "mean_total_tokens")
}
summary["gate"] = {
    "minimum_completion_gain": 3,
    "require_state_nondecrease": True,
    "require_task_nondecrease": True,
    "maximum_ux_drop": 0.05,
}
summary["passed"] = bool(
    summary["delta"]["completion"] >= 3
    and summary["delta"]["state"] >= 0
    and summary["delta"]["task"] >= 0
    and summary["delta"]["mean_ux"] >= -0.05
)
payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
(root / "confirmation_summary.json").write_text(payload, encoding="utf-8")
print(payload)
PY
