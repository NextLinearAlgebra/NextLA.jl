#!/bin/bash
# Full-2.5D × all-variants × all-precisions × LA on/off OOM-bounded sweep.
#
# Per qr_schur_xpartition.tex §A.3 / Conflux SC'21 mathematics:
#   - M (fast-memory budget) extracted from device totalGlobalMem × NEXTLA_FASTMEM_FRAC.
#   - c, Px, Py, Pz, b all derived at runtime via derived_schedule.hpp.
#   - cuSOLVERMp baseline at every N (NVIDIA reference).
#
# Variants  : scqr3 (passes=3), cqr2 (passes=2), householder, givens, qdwh.
# Precisions: fp64, fp64mp, fp64mp_tf32, fp32full.
# LA        : on (default) and off (--no-la ablation).
# N         : 2048 → OOM, geometric-ish ladder.
#
# Env vars:
#   NP                   number of H200 GPUs (default 8)
#   SIZES                override N ladder (space-separated)
#   NEXTLA_FASTMEM_FRAC  fraction of HBM counted as "fast memory" (default 0.72)
#   NEXTLA_BENCH_RUNS    number of timed runs per combo (default 5)
#
# Usage:
#   NP=8 ./run_full25d_oom_sweep.sh

set -u
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH:-}
export NEXTLA_FASTMEM_FRAC=${NEXTLA_FASTMEM_FRAC:-0.72}

NP=${NP:-8}
SIZES=${SIZES:-"2048 4096 6144 8192 12288 16384 24576 32768 49152 65536 98304 131072"}
LA_MODES="la no-la"
MATRIX_MODES="fp64 fp64mp fp64mp_tf32 fp32full"
RUNS=${NEXTLA_BENCH_RUNS:-5}

MPIRUN="mpirun --map-by :OVERSUBSCRIBE -np $NP"
HEAD () { printf "\n========================================================================\n  %s\n========================================================================\n" "$*"; }

# Compute cuSOLVERMp 2D grid (Px*Py = NP). Pure 2D (NVIDIA baseline has no replication).
case $NP in
  1)  CUMP_PX=1; CUMP_PY=1 ;;
  2)  CUMP_PX=2; CUMP_PY=1 ;;
  4)  CUMP_PX=2; CUMP_PY=2 ;;
  8)  CUMP_PX=4; CUMP_PY=2 ;;
  *)  CUMP_PX=$NP; CUMP_PY=1 ;;
esac

run_or_oom () {
    # Run a command; capture exit code. Return 0 on success, 137 (or non-zero)
    # on OOM/abort. The sweep treats any non-zero exit at a given N as
    # "this combo failed; continue with other combos at this N, but flag the N".
    local cmd="$1"
    eval "$cmd"
    local rc=$?
    return $rc
}

# Header.
HEAD "Full-2.5D OOM-bounded sweep — NP=$NP"
printf "  date          : %s\n" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
printf "  host          : %s\n" "$(hostname)"
printf "  GPUs          : NP=%d, CUMP grid=[%d, %d]\n" $NP $CUMP_PX $CUMP_PY
printf "  Sizes         : $SIZES\n"
printf "  Matrix modes  : $MATRIX_MODES\n"
printf "  LA modes      : $LA_MODES\n"
printf "  Runs/combo    : $RUNS\n"
printf "  NEXTLA_FASTMEM_FRAC = %s\n" "$NEXTLA_FASTMEM_FRAC"

OOM_REACHED=0
LAST_SUCCESSFUL_N=0

for N in $SIZES; do
    HEAD "N = $N"

    # ---------- 1. cuSOLVERMp baseline (apples-to-apples NVIDIA reference) ----------
    if [ $OOM_REACHED -eq 0 ]; then
        printf "\n--- cuSOLVERMp baseline (grid=%dx%d, NP=%d) ---\n" $CUMP_PX $CUMP_PY $NP
        if run_or_oom "$MPIRUN ./cusolverMp_geqrf_bench $N 256 256 $CUMP_PX $CUMP_PY"; then
            : # success
        else
            printf "\n[OOM-FLAG] cuSOLVERMp failed at N=$N — likely OOM. Continuing variants at this N anyway.\n"
        fi
    fi

    # ---------- 2. Path s: sCQR3 (passes=3) and CQR2 (passes=2) ----------
    for PASSES in 3 2; do
        VTAG=$([ $PASSES -eq 3 ] && echo scqr3 || echo cqr2)
        for MAT in $MATRIX_MODES; do
            for LA in $LA_MODES; do
                LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
                printf "\n--- variant=%s matrix=%s la=%s N=%d ---\n" $VTAG $MAT $LA $N
                run_or_oom "$MPIRUN ./scqr3_full25d_bench --N=$N --passes=$PASSES --matrix=$MAT $LAFLAG" || \
                    printf "[FAIL] %s %s %s @ N=%d\n" $VTAG $MAT $LA $N
            done
        done
    done

    # ---------- 3. Path h: Householder ----------
    for MAT in $MATRIX_MODES; do
        for LA in $LA_MODES; do
            LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
            printf "\n--- variant=householder matrix=%s la=%s N=%d ---\n" $MAT $LA $N
            run_or_oom "$MPIRUN ./householder_2p5d_bench --N=$N --matrix=$MAT $LAFLAG" || \
                printf "[FAIL] householder %s %s @ N=%d\n" $MAT $LA $N
        done
    done

    # ---------- 4. Path g: Givens (tournament-parallel panel kernel) ----------
    # Givens uses a smaller panel size by default; capped sweep at N ≤ 32768
    # to keep wall-clock reasonable (the tournament kernel is Θ(b log m)).
    if [ $N -le 32768 ]; then
        for MAT in $MATRIX_MODES; do
            for LA in $LA_MODES; do
                LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
                printf "\n--- variant=givens matrix=%s la=%s N=%d ---\n" $MAT $LA $N
                run_or_oom "$MPIRUN ./givens_2p5d_bench --N=$N --matrix=$MAT $LAFLAG" || \
                    printf "[FAIL] givens %s %s @ N=%d\n" $MAT $LA $N
            done
        done
    else
        printf "[SKIP] givens at N=%d (capped; panel kernel cost grows as N b log m)\n" $N
    fi

    # ---------- 5. Path q: QDWH (6 Halley iters × 2n×n inner QR) ----------
    # QDWH operates on a 2N × N stacked matrix → OOMs ~ half the N at which
    # other variants OOM.  Cap at N ≤ 49152 by default.
    if [ $N -le 49152 ]; then
        for MAT in $MATRIX_MODES; do
            for LA in $LA_MODES; do
                LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
                printf "\n--- variant=qdwh matrix=%s la=%s N=%d ---\n" $MAT $LA $N
                run_or_oom "$MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6 --matrix=$MAT $LAFLAG" || \
                    printf "[FAIL] qdwh %s %s @ N=%d\n" $MAT $LA $N
            done
        done
    else
        printf "[SKIP] qdwh at N=%d (2N×N stack would OOM)\n" $N
    fi

    LAST_SUCCESSFUL_N=$N
done

HEAD "DONE  (last attempted N = $LAST_SUCCESSFUL_N)"
