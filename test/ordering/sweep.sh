#!/bin/bash
# Run inside an existing Slurm allocation, from the repository root.
set -euo pipefail
[[ -f test/ordering/report.jl ]] || { echo 'Run from the NextLA.jl repository root.' >&2; exit 1; }
command -v julia >/dev/null || { echo 'Load Julia first: module load julia'; exit 1; }
command -v timeout >/dev/null || { echo 'GNU timeout is required.'; exit 1; }
export STILES_CORES="${STILES_CORES:-${SLURM_CPUS_PER_TASK:-1}}"
export OMP_NUM_THREADS="$STILES_CORES"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1 ORDERING_BLAS_THREADS=1
export ORDERING_REPEATS="${ORDERING_REPEATS:-3}"
export ORDERING_METHODS="${ORDERING_METHODS:-amd,rcm,scotch,cholmod,stiles,natural,random}"
catalog_dir="${MATRIX_DIR:-$HOME/orcd/scratch/nextla-matrices}"
legacy_dir="${LEGACY_MATRIX_DIR:-$HOME/orcd/scratch/nextla-legacy-matrices}"
root="${ORDERING_SWEEP_OUTPUT:-$PWD/benchmark-results/ordering-sweep-${SLURM_JOB_ID:-local}-$(date +%Y%m%d-%H%M%S)}"
cap="${ORDERING_MATRIX_TIMEOUT:-300}"
[[ "$cap" =~ ^[1-9][0-9]*$ ]] || { echo 'ORDERING_MATRIX_TIMEOUT must be positive.'; exit 1; }
mkdir -p "$root"
if [[ -n "${ORDERING_SWEEP_MATRICES:-}" ]]; then
    IFS=',' read -r -a matrices <<< "$ORDERING_SWEEP_MATRICES"
else
    # Prioritize requested inputs, then the remaining catalog. Animal2 is opt-in:
    # its roughly 503 million lower-triangle entries make it a much larger run.
    matrices=(inla_graph_animal1 bodyy4 bodyy5 bodyy6)
    while IFS=',' read -r name rest; do
        name="${name//\"/}"
        [[ "$name" == name || "$name" == inla_graph_animal1 ]] && continue
        matrices+=("$name")
    done < test/catalog/matrices.csv
    matrices+=(nasasrb crankseg_2)
fi
[[ -f "$root/status.csv" ]] || echo 'matrix,run_status,report' > "$root/status.csv"
cat > "$root/index.html" <<'HTML'
<!doctype html><meta charset="utf-8"><title>Ordering sweep</title>
<style>body{font:17px system-ui;max-width:1000px;margin:40px auto}li{margin:12px}</style>
<h1>Float64 ordering sweep</h1>
<p>Each matrix report contains before/after patterns, factor fill, recursive boundaries, and measurements. Run status describes process completion; inspect each report for individual method failures and accuracy.</p>
<p><a href="status.csv">Run status CSV</a>. Missing inputs are skipped; no downloads or shifts are applied. A timeout can leave a partial report. Matrices run in priority order, so an allocation cutoff may leave later cases unattempted.</p><ul>
HTML
echo "Reports: $root/index.html"
for name in "${matrices[@]}"; do
    [[ "$name" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "Invalid matrix name: $name"; exit 1; }
    limit="$cap"
    if [[ -n "${SLURM_JOB_END_TIME:-}" ]]; then
        left=$((SLURM_JOB_END_TIME-$(date +%s)-90))
        (( left > 30 )) || { echo 'Allocation reserve reached; saved reports are retained.'; break; }
        (( limit <= left )) || limit="$left"
    fi
    input_dir=''
    for dir in "$catalog_dir" "$legacy_dir" "$PWD"; do
        if [[ -f "$dir/$name.mtx" ]]; then input_dir="$dir"; break; fi
    done
    if [[ -z "$input_dir" ]]; then
        echo "$name: MISSING (searched catalog, legacy, and repository directories)"
        echo "$name,MISSING," >> "$root/status.csv"
        printf '<li>%s: missing input</li>\n' "$name" >> "$root/index.html"
        continue
    fi
    folder="$root/$name"
    mkdir -p "$folder"
    if [[ -f "$folder/.finished" ]]; then
        status='PREVIOUSLY_FINISHED'
    else
        echo "[$(date -Is)] $name; timeout=${limit}s; input=$input_dir"
        if MATRIX_DIR="$input_dir" ORDERING_MATRICES="$name" ORDERING_OUTPUT="$folder" \
            timeout --signal=TERM --kill-after=10s "${limit}s" \
            julia --startup-file=no --project=test/ordering test/ordering/report.jl \
            2>&1 | tee "$folder/run.log"; then
            status='FINISHED_INSPECT_REPORT'
            touch "$folder/.finished"
        else
            code=$?
            status="STOPPED_EXIT_$code"
        fi
    fi
    echo "$name,$status,$name/index.html" >> "$root/status.csv"
    if [[ -f "$folder/index.html" ]]; then
        printf '<li><a href="%s/index.html">%s</a>: %s</li>\n' "$name" "$name" "$status" >> "$root/index.html"
    else
        printf '<li>%s: %s; <a href="%s/run.log">log</a></li>\n' "$name" "$status" "$name" >> "$root/index.html"
    fi
done
echo '</ul>' >> "$root/index.html"
echo "Saved: $root/index.html"
