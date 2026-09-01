#!/usr/bin/env bash
# Collect one complete train-only counterfactual lattice for the conformal router.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bench_root="${STATE_BENCH_ROOT:-/root/Agentic/STATE-Bench}"
python_bin="${STATE_BENCH_PYTHON:-${bench_root}/.venv/bin/python}"
output_root="${LATTICE_OUTPUT_ROOT:-/root/autodl-tmp/conformal_lattice_train_v1}"
workers="${WORKERS:-3}"

unset STATE_BENCH_AGENT_BASE_URL STATE_BENCH_AGENT_API_KEY STATE_BENCH_AGENT_MODEL
unset STATE_BENCH_AGENT_MAX_TOKENS STATE_BENCH_AGENT_TIMEOUT_SECONDS
unset STATE_BENCH_AGENT_MAX_RETRIES STATE_BENCH_EVAL_ENDPOINT
unset STATE_BENCH_EVAL_DEPLOYMENTS STATE_BENCH_EVAL_API_KEY

export PYTHONPATH="${repo_root}:${bench_root}"
export STATE_BENCH_MEMORY_PATH="${bench_root}/artifacts/statebench_cross_domain_pwm/memory/process_workflows.json"
export STATE_BENCH_MEMORY_MODE=hybrid
export PYTHONUTF8=1

cd "${repo_root}"
mkdir -p "${output_root}"

for mask in 0 1 2 3 4 5 6 7; do
  arm_root="${output_root}/shopping_assistant/mask${mask}"
  mkdir -p "${arm_root}/run1"
  export STATE_BENCH_LATTICE_MASK="${mask}"
  for attempt in 1 2 3; do
    missing="$(${python_bin} - "${arm_root}/run1" <<'PY'
import sys
from pathlib import Path
from state_bench.protocol import load_default_protocol, load_split_task_ids

run_dir = Path(sys.argv[1])
protocol = load_default_protocol()
expected = load_split_task_ids("shopping_assistant", "train", protocol.split_version)
present = {path.stem for path in run_dir.glob("*.json")}
print(",".join(task_id for task_id in expected if task_id not in present))
PY
)"
    if [[ -z "${missing}" ]]; then
      break
    fi
    echo "mask=${mask} attempt=${attempt} remaining=$(awk -F, '{print NF}' <<<"${missing}")"
    "${python_bin}" -m state_bench.scripts.run_batch \
      --domain shopping_assistant \
      --tasks "${missing}" \
      --output-dir "${arm_root}" \
      --num-runs 1 \
      --num-workers "${workers}" \
      --agent-class LatticeSubsetPWMAgent \
      --agent-client-class OpenCodeLLMClient \
      --agent-model-name gpt-5.4 \
      --retrieve-learnings-top-k 3 \
      --score-reasoning-effort high
  done

  "${python_bin}" - "${arm_root}/run1" "${mask}" <<'PY'
import json
import sys
from pathlib import Path
from state_bench.protocol import load_default_protocol, load_split_task_ids

run_dir = Path(sys.argv[1])
mask = sys.argv[2]
protocol = load_default_protocol()
expected = set(load_split_task_ids("shopping_assistant", "train", protocol.split_version))
paths = list(run_dir.glob("*.json"))
present = {path.stem for path in paths}
missing = sorted(expected - present)
unscored = []
for path in paths:
    record = json.loads(path.read_text(encoding="utf-8"))
    if any(record.get(key) is None for key in (
        "task_completion_pass", "state_requirements_met", "task_requirements_met", "ux_score"
    )):
        unscored.append(path.stem)
if missing or unscored:
    raise SystemExit(f"mask {mask} incomplete: missing={missing} unscored={sorted(unscored)}")
print(f"mask {mask}: 100/100 complete and scored")
PY
done

echo "complete lattice: ${output_root}"
