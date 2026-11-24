# NextLA tests

`test/runtests.jl` runs the tests with
[ParallelTestRunner.jl](https://github.com/JuliaTesting/ParallelTestRunner.jl):
each test runs in its own Julia process, several at once, and a test that
fails does not stop the others. A summary with the time and memory of each
test is printed at the end.

## Running

Run the tests from the NextLA folder with Julia 1.10 or newer. Everything
after `--` goes to the test runner. A shortcut keeps the commands short
(bash or zsh; ParallelTestRunner's docs suggest a similar one):

```bash
alias nlatest='julia --project -e "using Pkg; Pkg.test(test_args=ARGS)" --'
```

The first run takes longer: Pkg sets up and precompiles the test environment.

### Choosing tests

Every test is named by its path without `.jl`, e.g. `dense/lapack/qr/geqrt`.
A name selects every test that starts with it; `!` leaves tests out.

```bash
nlatest                                   # everything: dense, mxp, tlr
nlatest --list                            # the test names; × marks those that failed last time
nlatest mxp                               # one part: dense, mxp or tlr
nlatest dense/lapack/qr                   # one folder
nlatest mxp/blas/trsm/rectrxm_mixed       # one file
nlatest dense/lapack/qr/geqrt mxp/core    # several
nlatest '!tlr'                            # everything except tlr
```

`--list` remembers the last run for each Julia version and each `--backends`
choice, so after a failure you can rerun just the tests it marks.

### GPU backends

Each test runs on the CPU and on each GPU backend that is installed in the
test environment and working (CUDA, AMDGPU, oneAPI, Metal), except where a
routine exists only for some backends: then its test runs only there, e.g.
`dense/blas/trsm/unified_rectrxm` on GPUs only. No GPU package is
a test dependency, so a plain run installs none and runs on the CPU. To test
on a GPU, add its package to the test environment once (this edits
`test/Project.toml`; do not commit that change):

```bash
julia --project=test -e 'using Pkg; Pkg.add("CUDA")'    # or AMDGPU, oneAPI, Metal
```

To run on the GPU, define `nlatest` with `--backends`; it then applies to every
test (the list is comma-separated: `CPU`, `CUDA`, `AMDGPU`, `oneAPI`, `Metal`):

```bash
alias nlatest='julia --project -e "using Pkg; Pkg.test(test_args=ARGS)" -- --backends=CUDA'
nlatest                        # every test, CUDA only
nlatest mxp                    # the mixed-precision tests, CUDA only
```

Use `--backends=CPU,CUDA` for both, or `--backends=CPU` for the CPU alone (no
GPU package is loaded). `--backends` can be given only once, so to change it,
define `nlatest` again; without `--backends`, as at the top, every usable
backend runs.

### Speed and output

```bash
nlatest --jobs=4                            # 4 tests at once
nlatest --jobs=1                            # one test at a time
JULIA_NUM_THREADS=4 nlatest --jobs=4 mxp    # 4 tests at once, 4 threads each
nlatest --quickfail                         # stop at the first failure
nlatest --verbose                           # more output (start and set-up times)
```

### Another Julia version

With [juliaup](https://github.com/JuliaLang/juliaup), define `nlatest` with the
version (`+1.10`, `+release`, `+nightly`, ...) and, if needed, `--backends`:

```bash
alias nlatest='julia +1.10 --project -e "using Pkg; Pkg.test(test_args=ARGS)" -- --backends=CUDA'
nlatest mxp                    # the mixed-precision tests on Julia 1.10, CUDA only
```

### From the Julia REPL

Start Julia with `julia --project` in the NextLA folder (with juliaup,
`julia +1.10 --project` for another version). `Pkg.test` takes the same
arguments as `nlatest`, written between backticks; this is also the way on
Windows:

```julia
using Pkg
Pkg.test("NextLA")                                                   # everything
Pkg.test("NextLA"; test_args=`--list`)                               # the test names
Pkg.test("NextLA"; test_args=`mxp`)                                  # one part
Pkg.test("NextLA"; test_args=`dense/lapack/qr/geqrt mxp/core`)       # several
Pkg.test("NextLA"; test_args=`'!tlr'`)                               # everything except tlr
Pkg.test("NextLA"; test_args=`--backends=CUDA mxp`)                  # mxp on CUDA only
Pkg.test("NextLA"; test_args=`--jobs=4 mxp`)                         # 4 tests at once
```

### Results

Each test file runs in its own Julia process, so a failing file does not stop
the others. The summary lists every file with its time and memory, and the
command ends with an error if any test fails.

## Test files

### Dense (`test/dense`)

One test per file, at the same path as the source it tests in `src/dense`,
e.g. `dense/lapack/qr/geqrt` or `dense/blas/gemm/recgemm`; `--list` shows them
all. Three test a source of another name: `dense/blas/trsm/unified_rectrxm`
(`rectrxm.jl`), `dense/lapack/cholesky/cholesky_blocked` (`potrf_blocked.jl`)
and `dense/lapack/lu/lu_factor` (`lu_base.jl`, `lu_tiled.jl`). Their shared
helpers (LAPACK reference wrappers, backends, the LU test matrices) are loaded
by `test/setup/dense.jl`.

### Mixed precision (`test/mxp`)

One test per source file, at the same path as the source in `src/mxp`:

| Test name | Tests |
|---|---|
| `mxp/core/fullmixedprec` | `FullMixedPrec` |
| `mxp/core/symmmixedprec` | `SymmMixedPrec` |
| `mxp/core/trimixedprec` | `TriMixedPrec` |
| `mxp/core/tiledtrimixedprec` | `TiledTriMixedPrec` |
| `mxp/core/transposedmixedprec` | the lazy `transpose`/`adjoint` of a container |
| `mxp/core/blocks` | the argument checks all constructors share, and the block helpers |
| `mxp/core/adaptive` | `adaptive_precisions` |
| `mxp/core/quantize` | `quantize`, `dequantize` |
| `mxp/blas/gemm/recgemm_mixedprec` | `recgemm!` |
| `mxp/blas/trsm/rectrxm_mixed` | `unified_rectrxm!` with mixed-precision containers |
| `mxp/blas/syrk/recsyrk_mixedprec` | `recsyrk!` |
| `mxp/lapack/lu/reclu_mixedprec` | `lu_recursive!` |
| `mxp/lapack/cholesky/recpotrf_mixedprec` | `potrf_recursive!` |

`mxp/core` runs all the container tests, `mxp/blas` the BLAS routines, and so on.

The helpers shared by these tests (error bounds, test matrices, reference
results) are in `test/setup/mxp.jl`; the comment at its top explains the
error bounds.

### Tile low-rank (`test/tlr`)

One test per source file or folder, at the same path as the source in `src/tlr`:

| Test name | Tests |
|---|---|
| `tlr/core/tlr_matrix` | `TLRMatrix` and the `TileLowRank` exports |
| `tlr/core/compressed_ftlr_matrix` | `CompressedFTLRMatrix`: layout, packing, `rank_multiple` |
| `tlr/core/compression/norms` | the batched tile norms and the orthogonalization type |
| `tlr/core/compression/ara` | adaptive randomized approximation: convergence, basis, truncation |
| `tlr/core/compression/compress` | dense to `TLRMatrix` / `CompressedFTLRMatrix`, the compression workspace |
| `tlr/blas/gemm/dense_accumulation` | `gemm!` into a dense `C` with TLR operands |
| `tlr/blas/gemm/compressed_accumulation` | `gemm` of two `CompressedFTLRMatrix` into a compressed result |

The tests opt in with `using NextLA.TileLowRank`, as users do. Their helpers
(tile fixtures, reconstruction, the GEMM drivers) are in `test/setup/tlr.jl`.
The GPU tests run on the backends the TLR code is written for: CUDA for the
batched SVD, grouped GEMM and TF32 paths, and AMDGPU where the source has its
own path.
