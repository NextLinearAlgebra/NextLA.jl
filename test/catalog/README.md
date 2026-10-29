# Expanded real-matrix Cholesky survey

Twenty **new**, successfully downloaded catalog matrices are selected in `matrices.csv`;
the exact names are in `download-list.txt`. They span 6 catalog groups, n=3,948 to 36,000, and stored
lower-triangle densities from 0.030% to 22.10%. The first wave deliberately avoids the
much larger dense allocations needed for the remaining thermal/circuit/bone groups.
One FP64 dense matrix at n=36,000 is 9.66 GiB; the worker conservatively budgets five
such matrices before GPU allocation. This estimate is not a guarantee against OOM.

The original selection contained 30 matrices. The ORCD download log on 2026-10-29
confirmed that 20 succeeded and 10 catalog URLs returned HTTP 404. Those 10 are
preserved with their catalog metadata and reason in `matrices-unavailable.csv`.
They are excluded from default downloads and benchmarks. Download failures are
not numerical failures. No substitute matrices have been added.

The old six inputs are not repeated. No diagonal shifts or normalization are applied.
Use the actual `.mtx` downloads from
<https://esmail-abdulfattah.github.io/sTiles/suits/index.html>, since the catalog may
have modified an original SuiteSparse matrix to make it SPD.

## Where to put downloads

Downloads on your laptop are fine temporarily. Extract archives, then transfer the
20 `<name>.mtx` files to one dedicated **ORCD-accessible** directory. The worker
expects flat filenames, for example `~/orcd/scratch/nextla-matrices/nd3k.mtx`.
It does not read zip/gzip archives or recursively search extracted subdirectories.
Set `MATRIX_DIR` to that directory. Local Windows Downloads cannot be read by an
ORCD job. Matrix data and results should not be committed to Git.

Download the selected files **directly on ORCD** without clicking each catalog page:

```bash
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
julia test/catalog/download.jl
```

The downloader retains completed files, checks the MatrixMarket header, dimensions,
stored-entry count and actual number of entry lines, and logs source URLs and local
SHA-256 fingerprints in `download-status.csv`. These fingerprints record what was
downloaded; the catalog does not supply trusted checksums for comparison. Partial
downloads stay under `.part` names and are retried from the beginning. Rerunning
skips valid completed files. `--check-only` validates files without downloading.
`DOWNLOAD_TIMEOUT_SECONDS` defaults to 600 per attempt and `DOWNLOAD_RETRIES` to 3.
Missing/404 catalog files are recorded as FAILED, not replaced with possibly different
SuiteSparse originals. Successful downloads remain usable even if another URL fails.
With the updated default manifest, rerunning should report EXISTING for the 20
valid files already on ORCD; the 10 known broken links are not requested again.
Use `julia test/catalog/download.jl --check-only` to validate without network access.
If the server is repaired later, the unavailable set can be explicitly retried with
`MATRIX_MANIFEST=test/catalog/matrices-unavailable.csv julia test/catalog/download.jl`.

## Set up once (before spending allocation time on the full survey)

From the repository on ORCD, using the same Julia version as your previous runs:

```bash
julia --startup-file=no test/catalog/setup.jl
```

This creates a separate environment in `test/catalog`. It installs CUDA,
KernelAbstractions, MatrixMarket, StochasticRounding and the Julia sTiles wrapper.
It leaves the root `Project.toml` unchanged. Preserve the generated Manifest with
your experiment records (it is Git-ignored).

```bash
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
julia --project=test/catalog test/catalog/run.jl --check
```

`--check` only checks file presence and lists missing names. It needs no GPU and
does not read all matrix entries. Fix missing files before the full run.

## Run the selected matrices

For a two-hour interactive allocation, request CPU cores explicitly from a login node:

```bash
salloc --ntasks=1 --cpus-per-task=14 \
  --gres=gpu:h200:1 --mem=256G \
  -p mit_normal_gpu -t 02:00:00 \
  --account=mit_amf_advanced_gpu \
  --qos=mit_amf_advanced_gpu
```

Inside the compute allocation, run from the full repository containing `src/` and
`test/`. The separate `nextla-catalog-download` folder used for downloading alone
does not contain all solver sources. Load the same Julia module used for setup.
The following 6,000-second sweep budget leaves 20 minutes within a two-hour job:

```bash
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
export STILES_CORES="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$STILES_CORES"
# Avoid nested BLAS thread pools competing with sTiles' parallelism.
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export GPU_TIMEOUT_SECONDS=600
export STILES_TIMEOUT_SECONDS=300
export SWEEP_TIMEOUT_SECONDS=6000
export RESULTS_DIR="$PWD/benchmark-results/catalog-wave1"

julia --project=test/catalog test/sparse_cholesky_benchmarking.jl
```

The existing benchmark path forwards to the catalog runner. It no longer repeats
the synthetic dense experiment. Use a separate output directory for each distinct
CPU/core configuration. Old one-core sTiles results and new 14-core results must
remain separately labeled.

Defaults retain M1/M2/M4/M6/M7 (k FP16 levels followed by FP32 leaves), Pure F32/F64,
cuSOLVER F32/F64, and sTiles F64, with five measured repetitions after GPU warm-up.
The sTiles wrapper is warmed on a tiny separate matrix. It performs fresh analysis
and factorization for each full-size repetition, and closes each native factor.

Optional controls:

```bash
# Run only chosen new matrices, in manifest order:
export REAL_MATRICES="bcsstk15,nd3k,inla_graph_lidense"
# Run only one backend (default all):
export METHODS=gpu                # or stiles
# Restrict GPU schedules (default all nine):
export GPU_METHODS="M1,M2,M4,M6,M7,cuSOLVER_F32,cuSOLVER_F64"
```

Baselines are still needed **on each new matrix** to compute its speedup. The
existing six-matrix baseline data cannot supply timings for a different matrix.

## Timeouts, checkpoints and resuming

Each matrix/backend runs in a separate process. On ORCD, GNU `timeout` terminates
the worker process group, followed by SIGKILL after five seconds if necessary.
Limits include startup, compilation, loading, all methods/repetitions and accuracy:

- GPU: 600 s for all selected GPU methods on one matrix.
- sTiles: 300 s for all sTiles repetitions and solve checks on one matrix.
- Sweep default: 10,800 s (3 h) per invocation for a 4 h allocation. The two-hour
  example above explicitly sets 6,000 s (100 minutes); choose a budget shorter than
  the actual remaining allocation time.

These are **wall-time budgets**, not per-factorization timing thresholds. If GPU M7
hangs after M1-M6 completed, their rows survive. Unfinished methods are marked
TIMEOUT; they were not necessarily the cause of the timeout. The next backend or
matrix can still run. The global budget also caps the next worker's allowed time.
When the **global** budget stops a worker, its case summary says SWEEP_LIMIT and
unfinished methods remain eligible for resume, rather than being marked slow.

Results are checkpointed atomically after each method. `results.csv` is regenerated
after each worker, with per-case CSVs in `cases/`, native output in `logs/`, backend
wall times in `cases-summary.csv`, and settings/environment/source/input hashes in
`settings.toml`. Interrupting the supervisor terminates its active worker.

To continue after the sweep budget or allocation ends, use the same environment,
settings and `RESULTS_DIR` and rerun the command. Completed/failed/timed-out/missing
rows are not retried automatically. Inputs, settings or source changes require a
new results directory. To retry a failed case with a higher timeout, use a new
directory plus `REAL_MATRICES`/`METHODS`; preserve the failed observations.

## Accuracy and interpretation

Eight fixed Gaussian probes estimate
`norm((L*L' - A)*Omega) / norm(A*Omega)` and the corresponding input-storage error.
These are randomized estimates, not exact Frobenius residuals. Inputs/probes are
identical across schedules. An independent block decoder applies every stored
scale in FP64; decoded factors use only the lower triangle.

Four known-solution right-hand sides give forward error, equation residual and
Frobenius-scaled backward error (maximum across the four). GPU solve checks use
FP64 triangular arithmetic on the decoded factors, with no refinement. sTiles
uses its native solve. Its factor/quantization residual fields remain blank.
`OK` means finite measurements completed, **not** that a chosen accuracy tolerance
was met. Missing values, failures and memory/time skips are not zero errors.

Mixed timing includes container construction and factorization. Pure/cuSOLVER
timing covers factorization. All exclude input conversion/transfer, resetting,
decoding and accuracy. sTiles analysis and factor are reported separately; the
phase pair is from the repetition with the smallest combined time. All timing
comparisons retain these scope and CPU/GPU distinctions.

`max_abs`, `min_abs_nonzero`, density, diagonal-dominance diagnostics, scaled-block
counts and scale maxima support trend plots. No dense condition number is computed.
The schema includes matrix groups, methods, timings, residuals and explicit status
so results can be plotted without parsing terminal text.

Generate the heatmap, accuracy/speedup comparisons (separate cuSOLVER and sTiles
baselines), speedup/dimension plot and quantization/factor-error plot with:

```bash
julia test/catalog/setup.jl --plots
julia --project=test/catalog test/catalog/plot.jl "$RESULTS_DIR/results.csv"
```

Run setup before beginning a results directory: adding packages changes the
environment fingerprint. Plots go into that run's `plots/` directory. sTiles
speedups use numeric factor time; its analysis is separately available in the CSV.
Errors/timeouts are labeled in the heatmap and omitted from finite scatter points.

## Validation

```bash
julia --project=test/catalog test/catalog/runtests.jl
julia --project=test/catalog test/catalog/runner_integration_tests.jl
julia --project=test/catalog test/catalog/integration_tests.jl
```

Tests exercise CSV/checkpoints, selection, scale-aware decoding, known-solution
metrics, and stopping a process blocked in a native call. GPU and sTiles integration
must also be checked on the allocated node. The CUDA source loader uses this
branch's corrected `cholesky_tree_subdiv.jl`, not the old embedded fallback.
