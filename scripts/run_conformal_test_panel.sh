#!/usr/bin/env bash
# Run one clean 50-task test pass in each domain after train-only certification.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bench_root="${STATE_BENCH_ROOT:-/root/Agentic/STATE-Bench}"
python_bin="${STATE_BENCH_PYTHON:-${bench_root}/.venv/bin/python}"
output_root="${PANEL_OUTPUT_ROOT:-/root/autodl-tmp/conformal_lattice_test_panel_v1}"
router_path="${STATE_BENCH_CONFORMAL_ROUTER_PATH:-${repo_root}/artifacts/conformal_lattice_router/router.json}"
workers="${WORKERS:-3}"

unset STATE_BENCH_AGENT_BASE_URL STATE_BENCH_AGENT_API_KEY STATE_BENCH_AGENT_MODEL
unset STATE_BENCH_AGENT_MAX_TOKENS STATE_BENCH_AGENT_TIMEOUT_SECONDS
unset STATE_BENCH_AGENT_MAX_RETRIES STATE_BENCH_EVAL_ENDPOINT
unset STATE_BENCH_EVAL_DEPLOYMENTS STATE_BENCH_EVAL_API_KEY

export PYTHONPATH="${repo_root}:${bench_root}"
export STATE_BENCH_MEMORY_PATH="${bench_root}/artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"
export STATE_BENCH_MEMORY_MODE=hybrid
export STATE_BENCH_CONFORMAL_ROUTER_PATH="${router_path}"
export PYTHONUTF8=1

"${python_bin}" - "${router_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
artifact = json.loads(path.read_text(encoding="utf-8"))
if artifact.get("schema_version") != "conformal_lattice_router_v1":
    raise SystemExit("invalid conformal router artifact")
if not artifact.get("deployment_enabled"):
    raise SystemExit("router did not pass train-only certification; refusing test evaluation")
print("train-only certificate accepted")
PY

cd "${repo_root}"
mkdir -p "${output_root}"
for domain in shopping_assistant travel customer_support; do
  domain_root="${output_root}/${domain}"
  mkdir -p "${domain_root}/run1"
  for attempt in 1 2 3; do
    missing="$(${python_bin} - "${domain_root}/run1" "${domain}" <<'PY'
import sys
from pathlib import Path
from state_bench.protocol import load_default_protocol, load_split_task_ids

run_dir = Path(sys.argv[1])
domain = sys.argv[2]
protocol = load_default_protocol()
expected = load_split_task_ids(domain, "test", protocol.split_version)
present = {path.stem for path in run_dir.glob("*.json")}
print(",".join(task_id for task_id in expected if task_id not in present))
PY
)"
    if [[ -z "${missing}" ]]; then
      break
    fi
    echo "domain=${domain} attempt=${attempt} remaining=$(awk -F, '{print NF}' <<<"${missing}")"
    "${python_bin}" -m state_bench.scripts.run_batch \
      --domain "${domain}" \
      --split test \
      --tasks "${missing}" \
      --output-dir "${domain_root}" \
      --num-runs 1 \
      --num-workers "${workers}" \
      --agent-class ConformalLatticeRouterAgent \
      --agent-client-class OpenCodeLLMClient \
      --agent-model-name gpt-5.4 \
      --retrieve-learnings-top-k 3 \
      --score-reasoning-effort high
  done
done

"${python_bin}" - "${output_root}" <<'PY'
import json
import sys
from pathlib import Path
from statistics import mean
from state_bench.protocol import load_default_protocol, load_split_task_ids

root = Path(sys.argv[1])
protocol = load_default_protocol()
summary = {
    "protocol": protocol.protocol_id,
    "split": "test",
    "num_runs": 1,
    "domains": {},
}
for domain in ("shopping_assistant", "travel", "customer_support"):
    run_dir = root / domain / "run1"
    expected = set(load_split_task_ids(domain, "test", protocol.split_version))
    records = {}
    for path in sorted(run_dir.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        records[str(record.get("task_id", ""))] = record
    missing = sorted(expected - set(records))
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
        raise SystemExit(f"{domain} incomplete: missing={missing}, unscored={unscored}")
    selected = [records[task_id] for task_id in sorted(expected)]
    summary["domains"][domain] = {
        "n": len(selected),
        "completion": sum(bool(item["task_completion_pass"]) for item in selected),
        "state": sum(bool(item["state_requirements_met"]) for item in selected),
        "task": sum(bool(item["task_requirements_met"]) for item in selected),
        "mean_ux": mean(float(item["ux_score"]) for item in selected),
        "mean_total_tokens": mean(float(item.get("total_tokens") or 0) for item in selected),
    }
summary["overall"] = {
    "n": sum(item["n"] for item in summary["domains"].values()),
    "completion": sum(item["completion"] for item in summary["domains"].values()),
    "state": sum(item["state"] for item in summary["domains"].values()),
    "task": sum(item["task"] for item in summary["domains"].values()),
    "macro_ux": mean(item["mean_ux"] for item in summary["domains"].values()),
}
payload = json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
(root / "panel_summary.json").write_text(payload, encoding="utf-8")
print(payload)
PY
