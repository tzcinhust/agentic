#!/usr/bin/env bash
# Run the prospective train confirmation, then test only after the fixed gate passes.
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bench_root="${STATE_BENCH_ROOT:-/root/Agentic/STATE-Bench}"
python_bin="${STATE_BENCH_PYTHON:-${bench_root}/.venv/bin/python}"
confirmation_root="${CONFIRM_OUTPUT_ROOT:-/root/autodl-tmp/fixed_rank3_confirmation_v1}"
panel_root="${PANEL_OUTPUT_ROOT:-/root/autodl-tmp/fixed_rank3_test_panel_v1}"

cd "${repo_root}"
CONFIRM_OUTPUT_ROOT="${confirmation_root}" bash scripts/run_fixed_rank3_confirmation.sh

passed="$(${python_bin} - "${confirmation_root}/confirmation_summary.json" <<'PY'
import json
import sys
from pathlib import Path
print(str(bool(json.loads(Path(sys.argv[1]).read_text())["passed"])).lower())
PY
)"
if [[ "${passed}" != "true" ]]; then
  echo "fixed rank-3 confirmation failed; test panel intentionally not started"
  exit 2
fi

PANEL_OUTPUT_ROOT="${panel_root}" bash scripts/run_fixed_rank3_test_panel.sh
