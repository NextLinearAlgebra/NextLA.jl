# Explore sparse Cholesky orderings

**Correction:** the original current-API sTiles adapter used the native export
in the wrong direction. It now uses `invperm(sTiles.permutation(F))` for
`A[p,p]`. A native-release check on a scrambled 225-node grid gives 1,780 factor
entries natively and after inversion, versus 5,483 with the old adapter.
Original real-matrix sTiles rows must be rerun and are withheld by the summary
generator unless the metadata records the corrected adapter. The separate legacy
`ordering(...).iperm` path is unchanged. Other ordering measurements are unaffected.
Run `julia --project=test/ordering test/ordering/stiles_direction_check.jl`
on ORCD to check against the native factor count. Use a fresh sweep output folder
and `ORDERING_METHODS=stiles` to rerun only these cases; old `.finished` markers
would otherwise skip them.

For an unattended corrected rerun, submit `test/ordering/rerun_stiles.sbatch`
from the repository root. It checks the native permutation convention first,
then runs only sTiles on the 25-matrix list, with 600 seconds per matrix and
three repetitions. It uses the existing Julia environment; set
`ORDERING_SETUP=1` before submission only if setup has not been done.
Results resume under `benchmark-results/ordering-stiles-corrected`, independent
of the submitting terminal. Logs go to `slurm-stiles-corrected-<jobid>.out`.
Submit only one job at a time to this output folder. Set `ORDERING_STILES_OUTPUT`
to a different absolute directory for an independent run.

The file defaults to the CPU `mit_normal` partition, 14 cores, 256 GiB and four
hours. If needed, override the partition/account/QOS with your permitted
configuration. For the user's previously working H200 allocation settings:

```bash
sbatch --partition=mit_normal_gpu --gres=gpu:h200:1 \
  --account=mit_amf_advanced_gpu --qos=mit_amf_advanced_gpu \
  test/ordering/rerun_stiles.sbatch
```

After `sbatch` prints `Submitted batch job <id>`, it is safe to log out.
The script's ordering/factorization computations still run on CPU.

For a single consolidated analysis of the November 5 sweep, use
`summarize.py INPUT_DIRECTORY OUTPUT_DIRECTORY` with Python, matplotlib and numpy.
It creates standalone HTML and PDF documents with overall takeaways, ranked
figures with bold takeaways, and all-matrix evidence. Example:

```bash
python -m pip install matplotlib numpy
python test/ordering/summarize.py /path/to/ordering-many-27991088 /path/to/summary
```

To incorporate the corrected rerun without recomputing other methods:

```bash
python test/ordering/summarize.py /path/to/ordering-many-27991088 /path/to/summary \
  --corrected-stiles /path/to/ordering-stiles-node-28042904
python test/ordering/summarize_tests.py
```

The merge requires matching input SHA-256 hashes, dimensions and nonzero counts,
plus the corrected-permutation marker and an OK status for every new sTiles row.
It replaces only sTiles metrics and uses the matching rerun figures/timing/block
files. Source downloads remain unchanged; superseded measurements are archived
separately. The report labels the rerun cohort and notes that its timing is from
a different allocation, even when CPU class and software settings match.

The HTML embeds every figure and can be shared as one file. The output also
includes Markdown, individual exportable figures, all metrics in one CSV, and
source CSV fingerprints. This is a curated analysis for this specific sweep;
its narrative examples require animal1, bodyy5 and 83o4NNNo and include a check
against the recorded bodyy5 timing ratio. Review/update the narrative for new
experiments rather than treating it as an unrestricted automatic conclusion
generator. No matrix factorization or GPU allocation is needed to summarize.

Start with the scrambled 24×24 grid demo (576 unknowns). It is deliberately SPD,
small, and initially scrambled to make the visual changes easy to recognize.
This is synthetic demonstration data, not a result for your real catalog.

```bash
julia test/ordering/setup.jl --stiles
julia --project=test/ordering test/ordering/runtests.jl
julia --project=test/ordering test/ordering/report.jl --demo
```

Open the printed `index.html` path in a browser. Everything is local and works
offline after dependencies have been installed. SVG pictures can be opened on
their own, zoomed, and saved. No plotting package, GPU, or web server is needed.
Omit `--stiles` during setup if you do not want that optional package; its report
row will explicitly say FAILED if it cannot load. It never silently substitutes
another method. The project is isolated from NextLA's root and catalog projects.

For the same `.mtx` files used by `test/sparse_cholesky_benchmarking.jl` (which
delegates to `test/catalog/run.jl`), run on the machine holding the matrices:

```bash
export MATRIX_DIR=/path/to/your/matrices
export ORDERING_MATRICES=bcsstk15,bodyy5
export STILES_CORES=8
julia --project=test/ordering test/ordering/report.jl
```

PowerShell equivalent:

```powershell
$env:MATRIX_DIR = 'C:\path\to\matrices'
$env:ORDERING_MATRICES = 'bcsstk15,bodyy5'
$env:STILES_CORES = '8'
julia --project=test/ordering test/ordering/report.jl
```

Use the catalog's actual SPD downloads; no averaging, normalization, or diagonal
shift is applied. A triangular input is mirrored once. A missing, nonsymmetric,
or indefinite input is reported as a failure. A failed candidate does not erase
the successful candidates. Output is checkpointed after each candidate. Natural
and random ordering can create very large factors on real matrices: start with
one smaller matrix. This standalone exploratory runner does not impose native
call timeouts or process/memory limits; use a scheduler job limit for long runs.

## Run on MIT ORCD

From the full repository on ORCD:

```bash
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
export ORDERING_MATRICES=bcsstk15
sbatch test/ordering/run.sbatch
```

The script requests a CPU job on `mit_normal` (8 cores, 32 GiB, one hour),
installs dependencies, runs the checks, and produces the report under
`benchmark-results/ordering-<jobid>`. It starts with one smaller catalog matrix.
It has bounded setup/test/report stages and checkpoints completed candidates.
Change memory/time and matrix selection as your experiment grows. If your default
account cannot use this CPU partition, pass an appropriate CPU account to
`sbatch --account=...`; the existing catalog script's advanced GPU account/QOS
are not assumed to authorize CPU jobs. Copy the entire output folder back to
your laptop and open `index.html` there.

This stage uses CPU sparse Cholesky even if executed on an H200 node. An H200
allocation becomes useful for the next experiment: load these permutations,
apply `A[p,p]` before transfer, and compare **Float64 NextLA** on the GPU under
identical recursion settings. Only then vary the precision schedule and measure
error. The current catalog GPU worker does not consume these permutation CSVs;
that GPU experiment needs explicit integration and is not measured by this report.

ORCD references: [partitions](https://orcd-docs.mit.edu/running-jobs/overview/),
[GPU resource requests](https://orcd-docs.mit.edu/running-jobs/requesting-resources/).

If you already requested an interactive H200 allocation with `salloc`, wait for
it to be granted, then enter its compute node with `srun --pty bash` if your
shell is still on a login node. Do not submit `run.sbatch` inside that allocation.
Run the commands directly on the compute node instead:

```bash
cd ~/NextLA.jl
git switch vicki-development
git pull --ff-only origin vicki-development
module load julia
export STILES_CORES="${SLURM_CPUS_PER_TASK:-14}"
export OMP_NUM_THREADS="$STILES_CORES"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
julia --startup-file=no test/ordering/setup.jl --stiles
julia --startup-file=no --project=test/ordering test/ordering/runtests.jl
export ORDERING_OUTPUT="$PWD/benchmark-results/ordering-demo-${SLURM_JOB_ID}"
julia --startup-file=no --project=test/ordering test/ordering/report.jl --demo
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
ls "$MATRIX_DIR/bcsstk15.mtx"
export ORDERING_MATRICES=bcsstk15
export ORDERING_OUTPUT="$PWD/benchmark-results/ordering-${SLURM_JOB_ID}"
julia --startup-file=no --project=test/ordering test/ordering/report.jl
```

The H200 is idle during these CPU ordering tests; its presence does not change
which solver this report measures. Use CPU jobs for later ordering-only runs.

## Compared methods

## Sweep all available catalog and legacy inputs

Inside an existing compute allocation, after setup:

```bash
module load julia
export MATRIX_DIR="$HOME/orcd/scratch/nextla-matrices"
export LEGACY_MATRIX_DIR="$HOME/orcd/scratch/nextla-legacy-matrices"
export ORDERING_SWEEP_OUTPUT="$PWD/benchmark-results/ordering-many-$SLURM_JOB_ID"
bash test/ordering/sweep.sh
```

The sweep searches both directories and the repository root for each file.
It prioritizes animal1 and bodyy4/5/6, then tries the remaining catalog inputs,
nasasrb, and crankseg_2 (25 matrices total). This is a coverage list, not a claim
that all these inputs have passed every previous numerical test. Missing inputs
are recorded and skipped. AMD/RCM/SCOTCH/CHOLMOD/sTiles run before natural/random
so expensive baselines do not prevent obtaining the main ordering comparisons.
Each matrix gets a separate report and log; the top-level `index.html` links them.
The default is three repetitions and a 300-second process limit per matrix.
Individual method results are checkpointed. Timeouts may leave partial reports.
The sweep stops 90 seconds before the allocation end when Slurm supplies that
time. To resume, reuse `ORDERING_SWEEP_OUTPUT`; finished matrix processes are
skipped, while interrupted cases restart. Inspect reports for method-level
failures even when a process is marked finished.

For just the requested smaller inputs:

```bash
export ORDERING_SWEEP_MATRICES=inla_graph_animal1,bodyy4,bodyy5,bodyy6
bash test/ordering/sweep.sh
```

Animal2 is opt-in because the catalog lists about 503 million lower-triangle
nonzeros. Run it separately with more time/memory available; the timeout below
is a cap, not an estimate of completion time:

```bash
export ORDERING_SWEEP_MATRICES=inla_graph_animal2
export ORDERING_MATRIX_TIMEOUT=1800
export ORDERING_METHODS=amd,rcm,scotch,cholmod,stiles
export ORDERING_SWEEP_OUTPUT="$PWD/benchmark-results/ordering-animal2-$SLURM_JOB_ID"
bash test/ordering/sweep.sh
```

Its original matrix still appears in every before/after comparison even when
natural-order factorization is omitted. All results remain CPU Float64. A later
GPU/mixed-precision experiment is needed to establish effects on NextLA.

## Method reference

| Method | Implementation | Intended purpose |
|---|---|---|
| natural | Identity permutation | Original-order baseline |
| random | Seeded Julia `randperm` | Control showing why order matters |
| amd | `AMD.amd(A)` | Approximate minimum degree / fill reduction |
| rcm | Deterministic Julia BFS, lowest-degree roots, reversed | Bandwidth reduction; not an optimizer for fill |
| scotch | Scotch.jl graph + native `SCOTCH_graphOrder` | SCOTCH default ordering, including nested dissection |
| cholmod | Julia `cholesky(Symmetric(A)).p` | CHOLMOD's default fill-reducing choice |
| stiles | Legacy `ordering(...).iperm`, or current `analyze` + inverse of `permutation` | sTiles' selected ordering |

Scotch.jl currently exposes graph construction but no high-level graph ordering
function. `methods.jl` contains one small adapter to its generated native binding.
It removes diagonal self-edges, keeps symmetric adjacency, uses 1-based SCOTCH
indices, validates direct/inverse permutations, and uses a private seeded context
with SCOTCH's deterministic and fixed-seed options. The saved `p` maps
**new index → original index**, suitable for `A[p,p]`.
Its native API is an integration point that may need updating for future versions.

RCM here uses minimum-degree component roots, rather than a pseudo-peripheral
search. Results may differ from other RCM implementations. CHOLMOD default is
not labeled AMD: its actual default choice belongs to CHOLMOD. Neither sTiles'
selected permutation nor SCOTCH's default is promised to be the fastest order
for this CHOLMOD experiment.

Every candidate factors the same sparse Float64 input after symmetric
permutation. Explicit identity `perm` prevents CHOLMOD from replacing it. The
script checks the permutation round trip and validates the Cholesky action and
solve. These are comparisons of **orderings using one solver**, not comparisons
of solver speeds. `F \ b[p]` returns a permuted solution; `x[p] = y` restores it.

## Read the report

Each method has three pictures with identical coordinates and scales:

1. Original input (the same before picture for all methods).
2. Reordered input. Its nonzero count is unchanged.
3. Lower Cholesky factor. Orange pixels mark fill absent from the reordered input.

The matrix is indexed from the top left. For n > 600, each pixel aggregates a
matrix region; any nonzero lights it up. Orange takes priority in a mixed pixel.
These are occupancy plots, not a depiction of exact density or magnitude.
Factor zeros are dropped before storage/pattern counts; these are numerical
factor counts, not a separate symbolic-pattern estimate.

Compare `nnz_L` / `fill_ratio` for storage, and factor/solve times for performance.
`column_count_square_sum` is a structural work proxy, not an exact flop count.
`bandwidth` is max |i-j| over input nonzeros; `profile` sums each column's distance
to its first nonzero above the diagonal (implicit diagonal included).
No single score is best for every objective. Timing uses warm-up and repeated
complete symbolic + numeric factorizations; minimum and median are reported.
`setup_s` is a separate warmed call whose scope differs by candidate: it includes
a full factorization for CHOLMOD and tile layout for current sTiles. Do not use
that column alone to claim an ordering-algorithm speedup. Compilation, threading,
and hardware still matter. Forward error also depends on matrix conditioning;
the status uses residual thresholds, not a universal forward-error bound.

Red boundaries follow the split rule in `src/symmmixedprec.jl`: half for a
power-of-two block, otherwise the largest lower power of two. The block CSV
lists occupancy, Frobenius norm, and maximum absolute value of each lower
off-diagonal region. Boundary depth is configurable. This is the storage
hierarchy's geometry, not a full kernel schedule or execution trace.

Reordering changes **which values enter which blocks**, not the recursion's
sizes. This can inform a later precision experiment, but does not establish
precision error or speedup: Schur updates change values and introduce fill.
Dense NextLA kernels also retain dense cubic work even when a reordered matrix
looks sparse. Keep this report in Float64 first, then use promising permutations
in a separate NextLA experiment with controlled precision schedules.

## Settings and artifacts

| Environment variable | Default |
|---|---|
| `ORDERING_MATRICES` | `ORDERING_MATRIX`, otherwise `bodyy5` |
| `MATRIX_DIR` | Repository root |
| `ORDERING_METHODS` | `natural,random,amd,rcm,scotch,cholmod,stiles` |
| `ORDERING_REPEATS` | 5 |
| `ORDERING_PROBES` | 4 |
| `ORDERING_SEED` | 20261105 |
| `ORDERING_DEPTH` | 3 (0 hides recursion overlay) |
| `ORDERING_BLAS_THREADS` | 1 |
| `STILES_CORES` | 1 |
| `ORDERING_OUTPUT` | `test/ordering/results/<local timestamp>` |

Results contain HTML, SVG, a metrics CSV, per-method permutations and block CSVs,
input SHA-256 hashes for real matrices, hardware/run settings, and copies of the
active project and manifest when present. Preserve the whole folder when sharing.
Generated results and the local Manifest are Git-ignored.

Source references: [AMD.jl](https://github.com/JuliaSmoothOptimizers/AMD.jl),
[Scotch.jl](https://github.com/Keluaa/Scotch.jl),
[SCOTCH manual](https://gitlab.inria.fr/scotch/scotch/tree/master/doc),
[sTiles Julia implementation](https://github.com/esmail-abdulfattah/sTiles/blob/main/julia/src/factor.jl).
