#!/usr/bin/env bash
# Continue the certified pipeline after the detached train-lattice collector exits.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bench_root="${STATE_BENCH_ROOT:-/root/Agentic/STATE-Bench}"
python_bin="${STATE_BENCH_PYTHON:-${bench_root}/.venv/bin/python}"
grid_pid_file="${GRID_PID_FILE:-/root/autodl-tmp/conformal_lattice_train_v1.pid}"
lattice_root="${LATTICE_OUTPUT_ROOT:-/root/autodl-tmp/conformal_lattice_train_v1}"
router_path="${ROUTER_OUTPUT_PATH:-${repo_root}/artifacts/conformal_lattice_router/router.json}"
panel_root="${PANEL_OUTPUT_ROOT:-/root/autodl-tmp/conformal_lattice_test_panel_v1}"

if [[ ! -f "${grid_pid_file}" ]]; then
  echo "missing grid pid file: ${grid_pid_file}" >&2
  exit 1
fi
grid_pid="$(tr -d '[:space:]' <"${grid_pid_file}")"
while kill -0 "${grid_pid}" 2>/dev/null; do
  completed="$(find "${lattice_root}" -type f -name '*.json' 2>/dev/null | wc -l)"
  echo "waiting for train lattice: ${completed}/800"
  sleep 30
done

if [[ "$(find "${lattice_root}" -type f -name '*.json' 2>/dev/null | wc -l)" -ne 800 ]]; then
  echo "train lattice process exited without 800 trajectories" >&2
  exit 1
fi

cd "${repo_root}"
export PYTHONPATH="${repo_root}:${bench_root}"
"${python_bin}" scripts/train_conformal_lattice_router.py \
  --lattice-root "${lattice_root}" \
  --memory-path "${bench_root}/artifacts/statebench_cross_domain_pwm/memory/process_workflows.json" \
  --output "${router_path}"

deployment_enabled="$(${python_bin} - "${router_path}" <<'PY'
import json
import sys
from pathlib import Path
print(str(bool(json.loads(Path(sys.argv[1]).read_text())["deployment_enabled"])).lower())
PY
)"
if [[ "${deployment_enabled}" != "true" ]]; then
  echo "train-only certificate failed; test panel intentionally not started"
  exit 2
fi

PANEL_OUTPUT_ROOT="${panel_root}" \
STATE_BENCH_CONFORMAL_ROUTER_PATH="${router_path}" \
bash scripts/run_conformal_test_panel.sh
