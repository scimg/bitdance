#!/usr/bin/env bash
# Run every (benchmark, arm) cell. Resumable at question granularity via the JSONL, so
# a reap mid-cell costs a model reload rather than the scored questions.
#
# Cells are attempted in passes and a cell that fails is SKIPPED, not fatal. The reason
# is memory, not tidiness: bf16 needs 22.3 GiB resident, while the CPU-staged quantized
# arms need ~9 GiB, and the node's regression job leaves ~19 GiB free for most of its
# cycle. Aborting the driver on the first bf16 OOM would stall the five quantized arms
# behind a cell that simply cannot run yet. Passes repeat until nothing progresses.
set -u
cd /work
export HF_HOME=/work/hf_cache
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REPO="${REPO:-google/gemma-4-12B-it}"
ARMS="${ARMS:-fp4 w4a6 w6a6_tc w4a4_tc fp6 bf16}"
GPQA_LIMIT="${GPQA_LIMIT:-198}"
MMLU_LIMIT="${MMLU_LIMIT:-2000}"
PASSES="${PASSES:-6}"
RES=/work/kernel/e2e_results
mkdir -p "$RES"
base=$(basename "$REPO")

cell_done() {   # cell_done <bench> <arm> <limit>
  local f="$RES/${base}__$1__$2.jsonl"
  [ -f "$f" ] || return 1
  [ "$(wc -l < "$f")" -ge "$3" ]
}

for pass in $(seq 1 "$PASSES"); do
  remaining=0
  progressed=0
  for arm in $ARMS; do
    for spec in "gpqa:$GPQA_LIMIT" "mmlu_pro:$MMLU_LIMIT"; do
      bench="${spec%%:*}"; lim="${spec##*:}"
      if cell_done "$bench" "$arm" "$lim"; then continue; fi
      before=$(wc -l < "$RES/${base}__${bench}__${arm}.jsonl" 2>/dev/null || echo 0)
      echo "=============================================================="
      echo "=== pass $pass | $REPO | $bench | arm=$arm | limit=$lim"
      echo "=============================================================="
      # bf16 at 31B is 60 GiB and has to span cards; the quantized arms are 16-27 GiB
      # and stay on one, where they are faster for not crossing the interconnect.
      dm=()
      [ "$arm" = bf16 ] && [ -n "${BF16_DEVICE_MAP:-}" ] && dm=(--device-map "$BF16_DEVICE_MAP")
      python3 -m kernel.eval_bench --bench "$bench" --arm "$arm" \
        --repo "$REPO" --limit "$lim" "${dm[@]}"
      rc=$?
      after=$(wc -l < "$RES/${base}__${bench}__${arm}.jsonl" 2>/dev/null || echo 0)
      [ "$after" -gt "$before" ] && progressed=1
      if [ $rc -ne 0 ]; then
        echo "!!! cell $bench/$arm exited rc=$rc (scored $before -> $after); skipping"
      fi
      cell_done "$bench" "$arm" "$lim" || remaining=$((remaining + 1))
    done
  done
  echo "=== pass $pass complete: $remaining cell(s) still incomplete ==="
  [ "$remaining" -eq 0 ] && break
  if [ "$progressed" -eq 0 ]; then
    echo "=== pass $pass made no progress at all; the card is unavailable ==="
    exit 75      # tells supervise.sh this is environmental, not a code failure
  fi
done

echo "=== driver finished ==="
python3 -m kernel.eval_analyze --repo "$REPO" || true
