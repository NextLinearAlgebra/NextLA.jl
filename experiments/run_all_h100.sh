#!/usr/bin/env bash
# Everything that needs a rented H100, in one go. Writes a transcript to
# h100_results_<timestamp>.txt so nothing is lost if the session drops.
#
#   bash experiments/run_all_h100.sh
#   PROBE_N=16384 bash experiments/run_all_h100.sh      # bigger sweep
#
# Run from the repo root. Expects `--project=experiments` to resolve NextLA+CUDA.

set -u
PROJ="${PROJ:-experiments}"
OUT="h100_results_$(date +%Y%m%d_%H%M%S).txt"
export PROBE_N="${PROBE_N:-8192}"
export PROBE_BM="${PROBE_BM:-256}"
export PROBE_RMAX="${PROBE_RMAX:-128}"

exec > >(tee -a "$OUT") 2>&1
echo "### run_all_h100  N=$PROBE_N BM=$PROBE_BM RMAX=$PROBE_RMAX  $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ---------------------------------------------------------------------------
# 1. ALIGNMENT — one variant per process. A misaligned pointer hard-faults on
#    tensor-core kernels and the CUDA error is sticky, so a shared process
#    would report every later row as a false failure.
#    offA/offC = 1 breaks 16-byte alignment for every dtype (2/4/8 bytes < 16).
# ---------------------------------------------------------------------------
echo
echo "=============================================================="
echo "### PHASE align  (expect FAULT_MISALIGNED rows — that is the result)"
echo "=============================================================="
for spec in \
  "Float16       default FP16/computeFP32" \
  "Core.BFloat16 default BF16/computeFP32" \
  "Float32       tf32    FP32/TF32-tensor" \
  "Float32       default FP32/computeFP32" \
  "Float64       default FP64/computeFP64"
do
  set -- $spec; STYPE=$1; MD=$2; LBL=$3
  for oo in "0 0" "1 0" "0 1"; do
    set -- $oo
    S="$STYPE" OFFA="$1" OFFC="$2" MODE="$MD" LABEL="$LBL" \
      julia --project="$PROJ" experiments/align_one.jl 2>&1 \
      | grep -E "^RESULT" || echo "RESULT $LBL offA=$1 offC=$2 PROCESS_DIED"
  done
done
echo
echo ">> Read: offA/offC=0 must be OK. If offA=1 or offC=1 FAULTs, 16-byte"
echo ">> alignment is a HARD requirement for that dtype and the guard in"
echo ">> src/gemm_grouped.jl is load-bearing. Compare across dtypes: the"
echo ">> boundary should track tensor-core kernel selection, not element size."

# ---------------------------------------------------------------------------
# 2. Everything else — safe to share one process.
# ---------------------------------------------------------------------------
for ph in ranks descriptor fold rowsperrun plan mixedout overhead; do
  PROBE_PHASE="$ph" julia --project="$PROJ" experiments/h100_audit_probe.jl
done

# ---------------------------------------------------------------------------
# 3. Plan-cost scaling is host-only and the most expensive to discover late.
#    Already covered by `plan`, but repeat at the big grid you actually care
#    about if PROBE_N was raised.
# ---------------------------------------------------------------------------
echo
echo "### transcript written to $OUT"
