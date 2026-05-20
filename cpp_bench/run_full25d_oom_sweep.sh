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

set +e   # Allow failed combos to skip without aborting the sweep.
cd /home/ftome_local/comparative-bench/NextLA.jl/cpp_bench
export PATH=/home/ftome_local/miniforge3/bin:$PATH
export LD_LIBRARY_PATH=/home/ftome_local/miniforge3/lib:${LD_LIBRARY_PATH:-}
export NEXTLA_FASTMEM_FRAC=${NEXTLA_FASTMEM_FRAC:-0.72}

NP=${NP:-8}
# Per-NP size ladder.  2-GPU runs have ~2x per-rank memory pressure vs 4-GPU,
# so we cap N earlier.  4 and 8 GPU sweeps go to OOM territory.
if [ -z "${SIZES:-}" ]; then
  case $NP in
    2) SIZES="2048 4096 6144 8192 12288 16384 24576 32768 49152 65536" ;;
    4) SIZES="2048 4096 6144 8192 12288 16384 24576 32768 49152 65536 98304" ;;
    8) SIZES="2048 4096 6144 8192 12288 16384 24576 32768 49152 65536 98304 131072" ;;
    *) SIZES="2048 4096 6144 8192 12288 16384 24576 32768" ;;
  esac
fi
LA_MODES="la no-la"
MATRIX_MODES="fp64 fp64mp fp64mp_tf32 fp32full"
RUNS=${NEXTLA_BENCH_RUNS:-5}

# Per-NP variant caps (manuscript-driven: givens has Θ(b·log m) single-block
# panel, qdwh works on 2N×N stacked input so OOMs at ~half the other variants).
GIVENS_CAP=${GIVENS_CAP:-32768}
QDWH_CAP=${QDWH_CAP:-49152}

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
    # cuSolverMp is finicky about small grids:
    #   - NP=2: library returns INTERNAL_ERROR (error 7) at every N tested.
    #     Skip cuSolverMp entirely for NP=2; report NA for the column.
    #   - NP=4: works at N >= 8192 with MB=NB=512.
    #   - NP=8: works at N >= 4096 with MB=NB=512 (small N) or 1024 (larger).
    # The threshold and MB are NP-dependent below.
    CUMP_MIN_N=4096
    case $NP in
      2) CUMP_MIN_N=999999 ;;   # disable: library returns error 7 at NP=2 (all N tested)
      4) CUMP_MIN_N=999999 ;;   # disable: library returns error 7 at NP=4 (all N tested)
      8) CUMP_MIN_N=4096 ;;     # works at NP=8 grid 4x2 for N >= 4096
    esac
    if [ $N -lt $CUMP_MIN_N ]; then
        printf "\n--- cuSOLVERMp baseline SKIPPED at N=%d (NP=%d minimum-N=%d for library happiness) ---\n" \
               $N $NP $CUMP_MIN_N
    else
        if [ $N -le 8192 ]; then CUMP_MB=512
        else                     CUMP_MB=1024
        fi
        if [ $OOM_REACHED -eq 0 ]; then
            printf "\n--- cuSOLVERMp baseline (grid=%dx%d, NP=%d, MB=NB=%d matched to derived b) ---\n" $CUMP_PX $CUMP_PY $NP $CUMP_MB
            if run_or_oom "$MPIRUN ./cusolverMp_geqrf_bench $N $CUMP_MB $CUMP_MB $CUMP_PX $CUMP_PY"; then
                : # success
            else
                printf "\n[BASELINE-FAIL] cuSOLVERMp failed at N=$N (library v0.8.0 in this env returns INTERNAL_ERROR; baseline column will be NA at this row). Continuing variants.\n"
            fi
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
    # Givens uses a smaller panel size by default; capped at GIVENS_CAP
    # (default 32768) to keep wall-clock reasonable (Θ(b log m) single-block).
    if [ $N -le $GIVENS_CAP ]; then
        for MAT in $MATRIX_MODES; do
            for LA in $LA_MODES; do
                LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
                printf "\n--- variant=givens matrix=%s la=%s N=%d ---\n" $MAT $LA $N
                run_or_oom "$MPIRUN ./givens_2p5d_bench --N=$N --matrix=$MAT $LAFLAG" || \
                    printf "[FAIL] givens %s %s @ N=%d\n" $MAT $LA $N
            done
        done
    else
        printf "[SKIP] givens at N=%d (above GIVENS_CAP=%d)\n" $N $GIVENS_CAP
    fi

    # ---------- 5. Path q: QDWH (6 Halley iters × 2n×n inner QR) ----------
    # QDWH operates on a 2N × N stacked matrix → OOMs ~ half the N at which
    # other variants OOM.  Cap at N ≤ QDWH_CAP (default 49152).
    if [ $N -le $QDWH_CAP ]; then
        for MAT in $MATRIX_MODES; do
            for LA in $LA_MODES; do
                LAFLAG=$([ "$LA" = "la" ] && echo --la || echo --no-la)
                printf "\n--- variant=qdwh matrix=%s la=%s N=%d ---\n" $MAT $LA $N
                run_or_oom "$MPIRUN ./qdwh_2p5d_bench --N=$N --iters=6 --matrix=$MAT $LAFLAG" || \
                    printf "[FAIL] qdwh %s %s @ N=%d\n" $MAT $LA $N
            done
        done
    else
        printf "[SKIP] qdwh at N=%d (above QDWH_CAP=%d)\n" $N $QDWH_CAP
    fi

    LAST_SUCCESSFUL_N=$N
done

HEAD "DONE  (last attempted N = $LAST_SUCCESSFUL_N)"
