# Handoff: bigger-GPU benchmarking (Julia + CUDA/MPI)

This document serves **two** audiences:

1. **Julia / NextLA.jl** — benchmark and tune `NextLA.geqrf_2p5d!` on
   server-class GPUs (A100, H100, MI250, etc.); laptop vs server regime
   notes below.
2. **CUDA/MPI `cpp_bench`** — Conflux-aligned multi-process QR benches,
   sweeps, vendor **`METRICS`**, and the executive dashboard. That work is
   driven by the plan file  
   [`/home/ftome_local/.cursor/plans/conflux-faithful_4-gpu_benches_bc7310ed.plan.md`](/home/ftome_local/.cursor/plans/conflux-faithful_4-gpu_benches_bc7310ed.plan.md)  
   (same content may appear under `.cursor/plans/` on another machine).

**Julia track** starts at [Why the bigger GPU matters](#why-the-bigger-gpu-matters).  
**`cpp_bench` track** is in the [appendix](#appendix-conflux-cuda-mpi-cpp-bench-track).

---

## Why the bigger GPU matters

The paper [qr_schur_xpartition.pdf](/home/felipetome/ECHO/qr_schur_xpartition.pdf)
defines the 2.5D replication factor as

```
c = ⌊PM/N²⌋,    admissible block size:  c ≤ b ≤ N/√P₁,    P₁ = P/c
```

On the RTX 4060 Laptop the per-SM shared-memory model gives
`P=24, M≈12800` FP64 words ⇒ `PM ≈ 3.1×10⁵`. For any `N ≥ 600` this
forces `c=1` and we collapse to plain 2D block-cyclic, where sCQR3-2.5D
has **no flop or bandwidth advantage** over Householder QR (cuSOLVER).
That is exactly what we measured: ~0.63× of cuSOLVER FP64 across
N ∈ {1000, 2000, 4000} — a constant ratio, no asymptotic win.

On an A100 (108 SMs, ~166 KB smem/SM) the same model gives
`PM ≈ 2.2×10⁶` FP64 words; on H100 it is larger still. The 2.5D regime
(`c ≥ 2`) should engage for N up to ~1500–2000 on those cards, and the
paper's `Θ(N²√c/√P)` bandwidth scaling should kick in.

Also: server cards have FP64 throughput close to FP32 (no 1:64 cripple),
so FP64 targets like "1000×1000 in 1.5 ms" are physically reachable
where they are not on a 4060 Laptop (FP64 peak 234 GF/s ⇒ 2.85 ms floor).

## What was done in the previous session

### Algorithm-level

- **`ortho` keyword** added to `geqrf_2p5d!`:
  - `:fast` (default) — single trailing projection per panel. Gives
    per-panel `O(u)` orthogonality (Fukaya 2018 Theorem 3.4) and
    inter-panel `O(κ·u)`. ~2N³/3 trailing flops.
  - `:safe` — adds Fix B (Björck 1967 §2.2 double trailing projection)
    and Fix C (double Gram-Schmidt pre-projection of the panel against
    accumulated Q). Global `O(u)` orthogonality up to κ ≈ u⁻¹. Costs
    roughly 5× the trailing-update flops of `:fast`.

  Use `:safe` only when you need κ-independent orthogonality up to the
  Fukaya bound on a single panel — i.e. when the matrix is genuinely
  ill-conditioned (κ ≳ 10⁶).

- **Vendor kernels everywhere**:
  - BLAS/cuBLAS `mul!` dispatch in `_geqrf_qta!`, `_geqrf_apply!`,
    and `scqr3_gram!` for `T <: BlasFloat`.
  - `_scqr3_potrf!` dispatches to `LAPACK.potrf!` on CPU and
    `CUDA.CUSOLVER.potrf!` on CUDA (see [ext/cudaext.jl](NextLA.jl/ext/cudaext.jl)).
  - TRSM via `rdiv!(view, UpperTriangular(G))` — cuBLAS DTRSM on GPU.

- **Per-panel scratch cleanup in `scqr3!`**: removed `Matrix{T}(I,b,b)`
  host allocation (replaced by device-side `_scqr3_fill_diag!`),
  removed per-iter `copyto!(Racc, Rwrk)` (swap-buffer instead),
  stripped redundant `KernelAbstractions.synchronize(be)` calls,
  fused trace-pack → workgroup-reduce → diag-shift into an on-device
  chain that no longer round-trips the trace value through host memory
  (`scqr3_shift_diag_from_trace_kernel!`).

### Tests & scripts

- [`test/geqrf_2p5d.jl`](NextLA.jl/test/geqrf_2p5d.jl) ill-conditioned
  sweep now exercises both modes: `ortho=:safe` against the
  κ-independent Fukaya bound, `ortho=:fast` against a κ·u-scaled tolerance.
- [`scripts/bench_gpu_vs_cusolver.jl`](NextLA.jl/scripts/bench_gpu_vs_cusolver.jl)
  sweeps b across the admissible window for FP64+FP32 at N ∈ {1000,
  2000, 4000} and reports `c`, GF/s, % of peak, and speedup vs
  `CUDA.CUSOLVER.geqrf!`. Set `BENCH_SIZES=4000,8000,16000` to extend.
- [`scripts/validate_ortho_modes.jl`](NextLA.jl/scripts/validate_ortho_modes.jl)
  CPU correctness checks across κ ∈ {1, 10², 10⁶, 10¹⁰, 10¹⁴} for both
  modes and all four BLAS element types.
- [`scripts/bench_and_diagnose.jl`](NextLA.jl/scripts/bench_and_diagnose.jl)
  inter-panel orthogonality diagnostics (per-step drift, off-diagonal
  Gram norm).

## What to do on the bigger GPU

### 1. Confirm the regime actually activates

Run this snippet first — it tells you whether 2.5D is even active for
the sizes you care about:

```julia
using NextLA, KernelAbstractions, CUDA
be = CUDABackend()
for N in (1000, 2000, 4000, 8000)
    p = compute_params(be, Float64, N)
    println("N=$N  c=$(p.c)  b=$(p.b)  b_min=$(p.b_min)  b_max=$(p.b_max)  Px=$(p.Px)")
end
```

If every line prints `c=1`, the cudaext probe model is still too
small. The probe in [ext/cudaext.jl](NextLA.jl/ext/cudaext.jl) uses
`MAX_SHARED_MEMORY_PER_MULTIPROCESSOR`. On A100/H100 that is
~166–228 KB/SM, which should give `c ≥ 2` for N up to ~1500. If you
want a larger admissible-b range, consider letting `probe_device` use
the L2 cache or a tunable fraction of global memory; the paper's `M`
is "fast memory per processor" and the choice is implementation-defined.

### 2. Re-run the benchmark

```bash
BENCH_SIZES=1000,2000,4000,8000,16000 \
    julia --project=NextLA.jl scripts/bench_gpu_vs_cusolver.jl
```

Look for:
- **`c` in the per-row output** — once `c ≥ 2`, the 2.5D advantage is
  expected to show up. The paper predicts `Θ(N²√c/√P)` bandwidth vs
  `Θ(N²/√P)` for c=1.
- **% of peak** — cuSOLVER on A100 FP64 should hit 30–60% of FP64 peak
  for N ≥ 4K. NextLA should track within ~30%.
- **Crossover N** — the size at which `:fast` matches or beats
  cuSOLVER. On A100 this is likely N ≳ 4K; on H100 possibly smaller.

### 3. Tune b inside the admissible window

The default `b` in `compute_params` is `clamp(⌊√M⌋, c, N/√P₁)` — the
X-partition cube side from §3.4. The latency-optimal `b★ = Θ(√(γP log
P/β))` from §3.9 requires hardware α/β parameters that we do not have
in code. The benchmark sweeps b across {64, 128, 192, 256, 384, 512,
640, 768}; record which b minimizes time for each (N, T) pair and
update the default heuristic if a different formula fits better.

### 4. Validate the FP32 / mixed-precision path

We did **not** explore Tensor Core paths. On A100/H100 the FP64 Tensor
Cores or BF16/TF32 GEMM with FP32 accumulate would change the picture
significantly. cuSOLVER `geqrf!` already uses TF32 on A100+ for FP32
inputs; matching that requires either:

- Routing the trailing GEMM through `CUDA.CUBLAS.gemmEx` with
  appropriate compute type, or
- Using `LinearAlgebra.mul!` with a `MathMode` setter on the CUDA
  stream (see CUDA.jl docs).

The current code goes through Julia's `mul!`, which dispatches to
cuBLAS GEMM with the array element type. This means FP64 uses the FP64
ALU, FP32 uses the FP32 ALU, and Tensor Cores are off. Enabling them
is a future direction.

### 5. Larger-N orthogonality regression

`ortho=:safe` was verified on CPU at 128×64. On GPU at N=4K nothing
exercises it. Run:

```julia
A = CUDA.randn(Float64, 4096, 4096)
# perturb to ill-condition
A_safe = copy(A); R = CUDA.zeros(Float64, 4096, 4096); tau = CUDA.zeros(Float64, 4096)
NextLA.geqrf_2p5d!(4096, 4096, A_safe, R, tau; b=128, ortho=:safe)
CUDA.synchronize()
Q = A_safe; G = Q' * Q - I
@show norm(G)  # expect ≲ 50·m·b·u ≈ 1e-10 for FP64 at this size
```

## Code map

```
NextLA.jl/
├── Project.toml            ─ CUDA is a weakdep; cudaext is the extension
├── ext/cudaext.jl          ─ probe_device, CUSOLVER POTRF override
├── src/
│   ├── xpartition.jl       ─ DeviceParams, compute_params (paper §3.4 aligned)
│   ├── scqr3.jl            ─ Panel factorization: gram, shift, potrf, TRSM (3-pass)
│   └── geqrf_2p5d.jl       ─ Outer loop with ortho=:fast|:safe gating
└── test/geqrf_2p5d.jl      ─ Correctness across imat conditioning cases

scripts/
├── bench_gpu_vs_cusolver.jl    ─ b sweep + cuSOLVER comparison (FP64/FP32, multi-N)
├── validate_ortho_modes.jl     ─ κ ∈ {1, 1e2, 1e6, 1e10, 1e14} on CPU
└── bench_and_diagnose.jl       ─ Inter-panel orthogonality diagnostics
```

## Known issues and gotchas

- **Pre-existing `compute_params` clamping bug for tiny N on multi-core
  CPU**: When `P·M / N² ≫ N`, the fixpoint clamp produces `c_val = N`
  which forces `b_min = N`, blocking small `b` overrides. Manifests for
  e.g. `compute_params(CPU(), Float64, 32; b=8)` with 16+ threads.
  Workaround: pass `c=1` explicitly when calling from tests or scripts
  with small N. The validate_ortho_modes.jl wrapper test hits this.
- **`ortho=:safe` allocates** `W2_buf` and `W_pre_buf` (sized for the
  worst-case panel) even on `:fast` paths. Future cleanup: skip those
  allocs in `:fast` mode. See lines ~338–342 of `geqrf_2p5d.jl`.
- **`partials_buf`** is allocated only when `p.c > 1`. On the bigger
  GPU you will start exercising the `c > 1` reduce path
  (`panel_allreduce!`, `_geqrf_qta_partitioned!`) which has been
  untested on hardware at scale. Validate against the c=1 result before
  trusting timings.
- **`compute_params` default `b`** is `clamp(⌊√M⌋, c, N/√P₁)`. With M
  derived from per-SM smem, this stays small (e.g. b=113 for FP64 on
  4060). On A100 you'll get b ~ 145, on H100 ~ 168. If the benchmark
  shows a different optimum, override `b=` explicitly or update the
  heuristic in `compute_params`.

## Quick start

```bash
# 1. Sanity check the install
julia --project=NextLA.jl -e 'using NextLA, CUDA; @show CUDA.functional()'

# 2. Correctness validation (CPU, fast)
julia --project=NextLA.jl scripts/validate_ortho_modes.jl

# 3. GPU benchmark sweep
BENCH_SIZES=1000,2000,4000,8000 \
    julia --project=NextLA.jl scripts/bench_gpu_vs_cusolver.jl
```

## Expected outcomes on A100/H100

These are predictions based on the paper's scaling laws, not measured:

- **N=1000 FP64**: NextLA `:fast` 0.7–0.9× cuSOLVER (similar regime to
  the 4060 since c is still small).
- **N=4000 FP64**: NextLA `:fast` 0.9–1.1× cuSOLVER (crossover region).
- **N=8000–16000 FP64**: NextLA `:fast` 1.1–1.5× cuSOLVER if `c ≥ 2`
  (paper's bandwidth advantage manifests).
- **FP32 / mixed precision**: cuSOLVER will likely still win on
  Ampere/Hopper because it uses TF32 Tensor Cores. To compete, enable
  TF32 path on the cuBLAS GEMM calls (see step 4 above).

If actual measurements diverge significantly from these predictions,
the most likely cause is M being chosen too conservatively (per-SM
smem instead of a larger fast-memory proxy). Adjust `probe_device` in
`ext/cudaext.jl` and re-run.

---

## Appendix: Conflux CUDA MPI cpp bench track

This appendix aligns with the **Conflux-faithful multi-GPU benches** plan:  
[`/home/ftome_local/.cursor/plans/conflux-faithful_4-gpu_benches_bc7310ed.plan.md`](/home/ftome_local/.cursor/plans/conflux-faithful_4-gpu_benches_bc7310ed.plan.md)  
(overview, YAML todos, central `DerivedSchedule` table, path table, validation rules, success metrics, scoped batches A–F).  
The **living dashboard** is [`NextLA.jl/cpp_bench/EXECUTIVE_SUMMARY.md`](NextLA.jl/cpp_bench/EXECUTIVE_SUMMARY.md).

### Plan todos — implemented in tree

| Todo id | Intent | Primary files |
|--------|--------|----------------|
| `pdf-pin` | SC’21 / PDF cross-links in comments | `derived_schedule.hpp`, bench headers (as pinned in-session) |
| `derived-params` | `DerivedSchedule`, rectangular `p_r×p_c`, `b_in_window`, `--M=` / `--smoke`, `--strict-derived-grid` | `cpp_bench/derived_schedule.hpp`, Path (s) + callers |
| `layout-engine` | Conflux-style layout checks; slab vs block-cyclic | `block_cyclic.hpp`, `layout_verify.hpp`, `--layout=` on benches |
| `comm-factory` | MPI_Comm_split + NCCL for panel / row / trailing groups | `comm_groups.cuh` |
| `path-s-blockcyclic` | Path (s) block-cyclic; TF32 trailing where supported | `scqr3_full25d_bench.cu`, `scqr3_block_cyclic.inl` |
| `paths-hgq` | (h)(g)(q) params, MP/TF32, QDWH tri-mode + QDWH BC | `householder_*.cu`, `givens_*.cu`, `qdwh_2p5d_bench.cu`, `qdwh_fp32full.inl`, `qdwh_block_cyclic.inl` |
| `validate-ci` | Smoke `mpirun -np P` (no GitHub CUDA CI per plan) | `run_smoke_validation.sh` |
| `cusolvermp-fp32-baseline` | libcusolverMp FP64 + FP32 | `cusolverMp_geqrf_bench.cpp`, `build.sh` |
| `perf-vs-cusolvermp` | Sweeps, parse aggregates + strict floors, postprocess gates, N_max probe | `run_full25d_sweep.sh`, `run_derived_sweep.sh`, `parse_bench_log.py`, `postprocess_sweep.py`, `n_max_gpu_probe.py` |
| `scripts-docs` | Scripts + TeX-facing column story | sweep scripts, `fuse_bench_metrics.py`, `fused_scqr3_metrics.py` |
| `exec-summary` | Executive sections (1)(2)(3) + P=4 table | `EXECUTIVE_SUMMARY.md`, `gen_executive_dashboard.py` |

### Scoped batch A–F — status vs plan

- **A — Derived schedule and CLI** — **Done:** `derived_schedule.hpp` (TeX-aligned `c_mem` / `c_cap`, downward `c` search, balanced rectangular `p_r×p_c=P₁`, `b_in_window` / `default_block_b`, `format_derived_schedule`, degenerate 1D schedule printouts for `--M=` on h/g/q). **`--M=`** required for faithful derived Path **(s)** unless **`--smoke`** (scripts set the hatch). **`--strict-derived-grid`:** fail fast instead of legacy `1×P×1` when no grid fits.
- **B — Shared block-cyclic (s)(h)(g)(q)** — **Updated (parity):** Path **(s)** slab + BC (`scqr3_block_cyclic.inl`, shared `nextla_mp_trail.hpp`). Paths **(h)(g):** **`--layout=blockcyclic`** with **`--no-la`** / **`--la`** matching Path **(s)** vocabulary. Path **(q):** **`--layout=blockcyclic --px=2 --py=2 --pz=1`** with **`qdwh_block_cyclic.inl`**: same **`--matrix=`** surface as slab (**`fp64`**, **`fp64mp`**, **`fp64mp_tf32`**, **`fp32full`**); inner stacked QR remains **dense / rank-0 panelized** (not full distributed CQR2 across ranks — see comments in **`qdwh_block_cyclic.inl`**). **`fp32full` BC** rejects **`--la`** (same as slab **`qdwh_fp32full`**).
- **C — TF32 Phase Q6** — **Done** as **`--matrix=fp64mp_tf32`** when `CUBLAS_COMPUTE_32F_FAST_TF32` exists; optional rank-0 TF32 banner in `bench_vendor_metrics.hpp`. **`SKIP_TF32_SMOKE=1`** in `run_smoke_validation.sh` skips TF32 smoke lines on older toolkits.
- **D — QDWH `fp32full`** — **Done:** `qdwh_fp32full.inl` + entry from `qdwh_2p5d_bench.cu`; **`METRICS`** `matrix=fp32full`; lookahead rules documented in bench.
- **E — Success metrics tooling** — **Done:** `bench_vendor_metrics.hpp`: **`NEXTLA_VENDOR_METRICS_TABLE`** (rows **`N P fp64_ms fp32_ms`**) matched by **`nextla_read_vendor_ms_for_np(N, mpi_np)`**; env vars **override** table when set. Legacy **`NEXTLA_VENDOR_METRICS_PATH`** / **`NEXTLA_VENDOR_*`** still supported. **`cusolverMp_geqrf_bench`** prints **`VENDOR_TABLE_ROW`** for **`capture_vendor_table.sh`**. **`parse_bench_log.py`:** `--aggregate`, `--strict-perf`, **`--strict-perf-geom`** (exit 3), **`--strict-perf-min`** (exit 4); 3D + 1D `METRICS` regexes. **`postprocess_sweep.py`:** `--require-geom`, `--require-min`, **`--gate-scqr3-fp64-min-geom`**, **`--gate-hgq-min-geom`**. **`n_max_gpu_probe.py`** for OOM-style **`N`** growth. Fusion scripts remain optional (Path **(s)** orchestration).
- **F — Executive summary** — **Done:** `EXECUTIVE_SUMMARY.md` (1)(2)(3); section 3 table from serial smoke; example deduped log `NextLA.jl/logs/p4_smoke_serial_20260518_vendor50_25.log` when vendor ms were synthetic.

**Regression note:** In `scqr3_block_cyclic.inl`, every **`ncclAllReduce`** must pass **`p1.nccl_p1`** before the stream (e.g. `stream_bc`). **`build.sh`:** **`LIBS`** includes **`-L${CUDA_HOME}/lib64`**; **`cusolverMp_geqrf_bench`** links with **`-lcusolverMp`** or is omitted with a stderr note if missing.

### cpp_bench code map

```
NextLA.jl/cpp_bench/
├── derived_schedule.hpp, comm_groups.cuh, block_cyclic.hpp, layout_verify.hpp
├── bench_vendor_metrics.hpp
├── scqr3_full25d_bench.cu, scqr3_block_cyclic.inl
├── householder_2p5d_bench.cu, householder_block_cyclic.inl
├── givens_2p5d_bench.cu, givens_block_cyclic.inl
├── qdwh_2p5d_bench.cu, qdwh_fp32full.inl, qdwh_block_cyclic.inl
├── cusolverMp_geqrf_bench.cpp, capture_vendor_table.sh
├── build.sh, run_smoke_validation.sh, run_full25d_sweep.sh, run_derived_sweep.sh
├── parse_bench_log.py, postprocess_sweep.py, gen_executive_dashboard.py
├── n_max_gpu_probe.py, aggregate_speedups.py, fuse_bench_metrics.py, fused_scqr3_metrics.py
└── EXECUTIVE_SUMMARY.md

NextLA.jl/logs/          # Dated sweep/smoke archives; METRICS extracts for dashboard
```

### Instructions for future agents (cpp_bench)

1. Read this appendix, the **plan** link above, and **`cpp_bench/EXECUTIVE_SUMMARY.md`** (especially “what is left” and the P=4 dashboard).
2. Set **`MPI_INCLUDE`** (dir with **`mpi.h`**), **`CUDA_HOME`**, and **`LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH`** before **`mpirun`**.
3. From **`NextLA.jl/cpp_bench`**, run **`./build.sh`**. The four CUDA/MPI benches should build; **`cusolverMp_geqrf_bench`** requires **`libcusolverMp`** (often under conda).
4. Run **`./run_smoke_validation.sh`**. Use **`SKIP_TF32_SMOKE=1`** if TF32 is undefined. Set **`SMOKE_LOG`** to a **fresh** path; avoid two concurrent smokes appending the same file (interleaved **`METRICS`**).
5. **Real cuSOLVERMp vendor medians:** run **`./capture_vendor_table.sh [out.txt]`** (needs **`cusolverMp_geqrf_bench`**). Then **`export NEXTLA_VENDOR_METRICS_TABLE=$PWD/out.txt`** (or rely on **`run_smoke_validation.sh`**, which picks up **`./vendor_metrics_table.txt`** if present). Table rows are **`N P vendor_fp64_ms vendor_fp32_ms`** with **`P = Px*Py`** matching the **`mpirun -np`** size used for our benches. **`NEXTLA_VENDOR_FP64_MS` / `NEXTLA_VENDOR_FP32_MS`** still **override** table entries when set.
6. Sweeps: **`run_full25d_sweep.sh`**, **`run_derived_sweep.sh`**; honor **`N_MAX`** / **`n_max_gpu_probe.py`** for largest **`N`** in **`S`**.
7. Parse: **`python3 parse_bench_log.py <log> --aggregate`**; add **`--strict-perf`**, **`--strict-perf-geom`**, **`--strict-perf-min`** as needed. Pipe **`--json`** into **`postprocess_sweep.py`** and use path gates when automating.
8. After major runs, update **`EXECUTIVE_SUMMARY.md`** section 3 and drop a canonical log under **`NextLA.jl/logs/`** with date and machine notes in a header comment.
9. On a **4-GPU** node, prefer a clean process–GPU map (avoid **`OVERSUBSCRIBE`** unless intentional); record CUDA driver, NCCL, and MPI versions in the log header.
10. **TeX / policy:** Path **(s)** default lookahead is **on** (`--no-la` ablates). Paths **(h)(g)(q)** now match: **default LA on**, **`--no-la` / `--no-lookahead`** supported. **QDWH `fp32full`:** inner-QR lookahead is not implemented — **`qdwh_2p5d_bench` main()** turns default LA off for that mode (rank-0 note once); **explicit `--la` + fp32full** still **`MPI_Abort`** with stderr (slab + BC).
11. **Still out of scope:** GitHub Actions for CUDA benches. **QDWH BC** inner QR remains rank-0–centric; **`METRICS`** print the requested **`matrix=`** mode.

### Parity smoke matrix (manual `mpirun`, small `N`, `P=4`)

Use after **`./build.sh`** from **`NextLA.jl/cpp_bench`**. Pick **`N`** divisible by **`P`** and by block size **`b`** (defaults apply when **`--b=0`**). For block-cyclic use **`--layout=blockcyclic --px=2 --py=2 --pz=1`** (and the same **`--matrix=`** / **`--la`** flags as slab). Path **(s)** adds **`--passes=2`** or **`3`** (other paths ignore it).

| Cell | Path (s) `scqr3_full25d_bench` | Path (h) `householder_2p5d_bench` | Path (g) `givens_2p5d_bench` | Path (q) `qdwh_2p5d_bench` |
|------|-------------------------------|-----------------------------------|------------------------------|----------------------------|
| slab `fp64` + default LA | `--N=1024 --matrix=fp64` | `--N=1024 --matrix=fp64` | `--N=1024 --matrix=fp64` | `--N=1024 --matrix=fp64` |
| slab `fp64` + `--no-la` | `--N=1024 --matrix=fp64 --no-la` | `--N=1024 --matrix=fp64 --no-la` | `--N=1024 --matrix=fp64 --no-la` | `--N=1024 --matrix=fp64 --no-la` |
| slab `fp64mp` + LA | `--N=1024 --matrix=fp64mp` | `--N=1024 --matrix=fp64mp` | `--N=1024 --matrix=fp64mp` | `--N=1024 --matrix=fp64mp` |
| slab `fp64mp_tf32` + LA | `--N=1024 --matrix=fp64mp_tf32` | `--N=1024 --matrix=fp64mp_tf32` | `--N=1024 --matrix=fp64mp_tf32` | `--N=1024 --matrix=fp64mp_tf32` |
| slab `fp32full` | `--N=1024 --matrix=fp32full --no-la` | `--N=1024 --matrix=fp32full` | `--N=1024 --matrix=fp32full` | `--N=1024 --matrix=fp32full` |
| BC same modes | add `--layout=blockcyclic --px=2 --py=2 --pz=1` | same | same | same (`fp32full`: do not pass explicit `--la`) |

Example (one line per path, slab `fp64mp` with default LA on):  
`mpirun -np 4 --bind-to none ./scqr3_full25d_bench --N=1024 --passes=2 --matrix=fp64mp`  
`mpirun -np 4 --bind-to none ./householder_2p5d_bench --N=1024 --matrix=fp64mp`  
`mpirun -np 4 --bind-to none ./givens_2p5d_bench --N=1024 --matrix=fp64mp`  
`mpirun -np 4 --bind-to none ./qdwh_2p5d_bench --N=1024 --matrix=fp64mp`

### Relation to the Julia track

`geqrf_2p5d!` (Julia) and **`cpp_bench`** (CUDA/MPI) are different codepaths; label which stack produced each timing when comparing to cuSOLVER or cuSOLVERMp.
