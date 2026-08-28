#!/usr/bin/env bash
# Launch one arm on one or more domains, with the flags that make runs comparable.
#
# Every arm in this study has to be run identically or the pairing in
# scripts/compare_arms.py is meaningless, and the flags that matter are easy to
# get wrong by hand: PYTHONUTF8=1 (without it TaskDefinition.load dies on gbk),
# --agent-client-class (without it the loader rejects a non-StateBenchAgent
# class), and --split test (without it --tasks silently spans all 150).
#
# The relay is a single shared upstream, so domains run in sequence rather than
# in parallel; overlapping batches just trade rate-limit errors for wall time.
#
# Usage:
#   ARM=a4 AGENT=GatedLedgerAgent scripts/run_arm.sh shopping_assistant
#   ARM=base AGENT=ProcessWorkflowMemoryAgent RUNS=1 RUN_START=2 scripts/run_arm.sh travel
#
# Env:
#   ARM        output prefix, giving outputs/<ARM>_<domain>/run<N>   (required)
#   AGENT      --agent-class                                          (required)
#   RUNS       replicates                                    (default 2)
#   RUN_START  first run index, for topping up an arm         (default 1)
#   WORKERS    parallel task workers                          (default 8)
#   SPLIT      train | test                                   (default test)
set -euo pipefail

: "${ARM:?set ARM, e.g. ARM=a4}"
: "${AGENT:?set AGENT, e.g. AGENT=GatedLedgerAgent}"
RUNS="${RUNS:-2}"
RUN_START="${RUN_START:-1}"
WORKERS="${WORKERS:-8}"
# Development measurement belongs on train. Defaulted to test because every arm
# comparison so far reports test, but a failure analysis that reads test is a
# failure analysis that has fitted to it.
SPLIT="${SPLIT:-test}"

cd "$(dirname "$0")/.."
set -a; . ./.env; set +a

# The OpenAI SDK honors the host's system proxy configuration. On Windows that
# can route the locked simulator/judge calls to the localhost eval shim through
# the proxy, which returns an empty HTTP 502 before the shim sees the request.
# Keep loopback traffic local while leaving every upstream request untouched.
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

# The locked eval client hardcodes endpoint + "/openai/v1/", so it can only reach
# the relay through the shim. 401 means the upstream answered and rejected an
# unauthenticated probe, which is the liveness signal; 000 means nothing listening.
code=$(curl -s -o /dev/null -w "%{http_code}" "${STATE_BENCH_EVAL_ENDPOINT}/openai/v1/models" --max-time 8 || true)
if [ "$code" = "000" ]; then
  echo "shim not answering at ${STATE_BENCH_EVAL_ENDPOINT} — start it with:" >&2
  echo '  SHIM_UPSTREAM="https://ai.novacode.top/v1" SHIM_PORT=8765 python tools/eval_shim.py' >&2
  exit 1
fi

for domain in "$@"; do
  out="outputs/${ARM}_${domain}"
  echo "##################### ${ARM} / ${AGENT} / ${domain} #####################"
  PYTHONUTF8=1 uv run python -m state_bench.scripts.run_batch \
    --domain "$domain" \
    --split "$SPLIT" \
    --output-dir "$out" \
    --num-runs "$RUNS" \
    --num-runs-idx-start "$RUN_START" \
    --num-workers "$WORKERS" \
    --agent-class "$AGENT" \
    --agent-client-class OpenCodeLLMClient \
    --agent-model-name "$STATE_BENCH_AGENT_MODEL"
done
