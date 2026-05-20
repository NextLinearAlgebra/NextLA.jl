# Executive summary — Conflux-aligned multi-GPU QR benches

**Date:** 2026-05-18 (refresh after each milestone)

## 1. What was done

### Scheduling and grids

- **`derived_schedule.hpp`:** TeX-aligned `c_mem` / `c_cap`, downward search for replication `c`, **rectangular** first-layer **`p_r × p_c = P₁`** via **`balanced_factors_p1`**, generalized **`b_in_window` / `default_block_b`**, **`format_derived_schedule`**, optional **`--strict-derived-grid`**. **`compute_degenerate_1d_schedule`** for **(h)(g)(q)** rank-0 printouts when **`--M=`** is set.

### Path **(s)** — `scqr3_full25d_bench.cu`

- **`--matrix=fp64|fp64mp|fp64mp_tf32|fp32full`**: TF32 trailing uses **`cublasGemmEx`** + **`CUBLAS_COMPUTE_32F_FAST_TF32`** when the CUDA toolkit exposes it; compile-time guard + runtime abort for **`fp64mp_tf32`** without TF32 support.
- **Grid resolution:** explicit **`--px/--py/--pz`**, or **`--M=`** derived schedule, or **`--smoke`** for the legacy cbrt heuristic (required when omitting all three).
- **`--layout=slab|blockcyclic`**, **`comm_groups.cuh`**, **`METRICS`** with **`layout=`** when applicable; **`bench_vendor_metrics.hpp`** injects **`vendor_fp64_ms` / `vendor_fp32_ms`** from the environment (no Python fusion).
- **NCCL:** `scqr3_block_cyclic.inl` **`ncclAllReduce`** calls pass **`p1.nccl_p1`** and dedicated CUDA streams (e.g. **`stream_bc`**) so host/device collectives stay well-formed.

### Paths **(h)** and **(g)** — `householder_2p5d_bench.cu`, `givens_2p5d_bench.cu`

- **`--matrix=fp64|fp64mp|fp64mp_tf32|fp32full`**, **`--la` / `--ir`**, **`METRICS`** with **`c=`** for 1D slab replication.
- **`--layout=blockcyclic`** with **`--px/--py/--pz`**: shared block-cyclic experiment branch (**`householder_block_cyclic.inl`**, **`givens_block_cyclic.inl`**) aligned with Path **(s)** **`Pz=1`** style (host panel + replicated LAPACK + device trailing).
- Optional rank-0 **TF32 trailing banner** when TF32 is active (see vendor header).

### Path **(q)** — `qdwh_2p5d_bench.cu`

- **`--matrix=fp64|fp64mp|fp64mp_tf32`** for stacked inner QR with TF32 on inner **`Sgemm`** when **`fp64mp_tf32`** and the toolkit supports it.
- **`--matrix=fp32full`:** **`qdwh_fp32full.inl`** float outer + float inner (**`run_qdwh_fp32full_main`**); **`--layout=blockcyclic`** is rejected (**`MPI_Abort`** with message to use slab paths).
- **`METRICS`** and TF32 banner wired like **(h)/(g)**.

### Vendor, parsing, sweeps

- **`cusolverMp_geqrf_bench.cpp`:** FP64 + FP32 baselines (link **`libcusolverMp`** when present; **`build.sh`** adds **`-L${CUDA_HOME}/lib64`** for CUDA math libs).
- **`parse_bench_log.py`:** **`--strict-perf-geom`** (exit 3) and **`--strict-perf-min`** (exit 4) on aggregate groups; **`--aggregate`** for geom_mean / min_speedup per bench×matrix×grid.
- **`postprocess_sweep.py`:** **`--require-geom`**, **`--require-min`**, **`--gate-scqr3-fp64-min-geom`**, **`--gate-hgq-min-geom`** (exits 5/6 as documented in the script).
- **`run_smoke_validation.sh`:** tri-mode **(s)(h)(g)**, **(q)** including **`fp32full`**; **block-cyclic** smoke for **(h)(g)**; optional **`SKIP_TF32_SMOKE=1`** when the toolkit lacks TF32 (avoids **`MPI_ABORT`** on **`fp64mp_tf32`**).

## 2. What is left

- **Replace synthetic vendor medians** in **`METRICS`** with measured **`cusolverMp`** (or other) times via **`NEXTLA_VENDOR_FP64_MS` / `NEXTLA_VENDOR_FP32_MS`** or **`NEXTLA_VENDOR_METRICS_PATH`**, then re-run sweeps so aggregate gates reflect real vendor baselines.
- **`fp64mp_tf32`** smoke and sweeps on a CUDA 11+ image where **`CUBLAS_COMPUTE_32F_FAST_TF32`** is defined.
- **QDWH block-cyclic** (still intentionally out of scope; use slab for **(q)**).
- **CI:** no GitHub Actions for these CUDA benches (by design for this batch).
- **Policy gap vs TeX “lookahead default on”:** Path **(s)** matches; **(h)(g)(q)** default **lookahead off** unless **`--la`**.

## 3. Current state on 4 GPUs (P = 4)

Measurements below are from a **single serial** smoke run (**`run_smoke_validation.sh`**), **`mpirun -np 4`**, grids **`[2,2,1]`** for 2.5D paths, **`SKIP_TF32_SMOKE=1`** on this runner (no TF32 symbol). **Vendor columns are synthetic** (**`NEXTLA_VENDOR_FP64_MS=50`**, **`NEXTLA_VENDOR_FP32_MS=25`**) so **`METRICS`** prints five numeric fields without Python fusion; speedups vs those placeholders are **not** claims against cuSOLVER/cuSOLVERMp.

**Canonical METRICS-only log (deduplicated):** `NextLA.jl/logs/p4_smoke_serial_20260518_vendor50_25.log`  
**Raw tee log (may contain interleaved lines if multiple smokes overlap):** `NextLA.jl/cpp_bench/smoke_metrics.log`

| Path / variant | Primary smoke config | Median ours ms | vs FP64 vendor (50 ms) | vs FP32 vendor (25 ms) | Orthogonality / stability (from stdout) |
|----------------|----------------------|----------------|-------------------------|-------------------------|------------------------------------------|
| **(s)** `scqr3_full25d` slab, **fp64**, passes=2 | N=2048, b=363, 2×2×1 | 8.07 | 6.19× | 3.10× | max|diag(Q'Q)−I| ≈ 2.9e−15 |
| **(s)** slab **fp64mp**, passes=2 | same | 8.16 | 6.13× | 3.06× | ≈ 3.3e−15 |
| **(s)** slab **fp32full**, passes=2 | same | 7.07 | 7.07× | **3.53×** (fp32 vendor) | max|diag(Q'Q)−I| ≈ 2.3e−6 |
| **(s)** slab **fp64** derived+LA, passes=2 | same | 8.03 | 6.23× | 3.11× | ≈ 2.9e−15 |
| **(s)** slab **fp64**, passes=1 (smoke grid) | same | 4.39 | 11.4× | 5.69× | ≈ 3.1e−6 (single-pass) |
| **(h)** `householder_2p5d` slab **fp64 / fp64mp / fp32full** | N=2048, b=363, c=4 | 16.92 / 17.00 / 15.22 | 2.95× / 2.94× / 3.28× | 1.48× / 1.47× / **1.64×** | Q orthog. ≈ 1.3e−15 (fp64/mp); fp32 ≈ 7e−7 |
| **(h)** block-cyclic **fp64** | N=2048, b=512, 2×2×1 | **1.84e4** | 0.0027× | 0.0014× | (timing only in smoke; path is experimental) |
| **(g)** `givens_2p5d` slab **fp64 / fp64mp / fp32full** | N=2048, b=64, c=4 | 4.88e3 / 4.88e3 / 4.31e3 | 0.010× / 0.010× / 0.010× | 0.005× / 0.005× / 0.005× | ≈ 1.6e−14 (fp64/mp); fp32 ≈ 8.6e−6 |
| **(g)** block-cyclic **fp64** | N=2048, b=512, 2×2×1 | **3.14e4** | 0.0016× | 0.00079× | (timing only in smoke) |
| **(q)** `qdwh_2p5d` **fp64 / fp64mp / fp32full** | N=1024, b=256, iters=2 | 7.58 / 7.45 / 11.13 | 6.59× / 6.71× / 4.49× | 3.30× / 3.35× / **2.25×** | max|diag(U'U)−I| ≈ 2.5e−1 (smoke settings; not a tight polar residual target) |

**Aggregate speedups** (same synthetic vendor, **`parse_bench_log.py --aggregate`** on `p4_smoke_serial_20260518_vendor50_25.log`): **(s)** `fp64` slab 2×2×1 geom_mean **7.605×** (n=3: passes 2, derived+LA passes 2, smoke passes=1), min **6.195×**; **`fp64mp`** geom **6.130×**; **`fp32full`** geom **3.535×**. **(h)** slab: **fp64** **2.954×**, **`fp64mp`** **2.942×**, **`fp32full`** **1.642×**; block-cyclic **fp64** **0.00271×**. **(g)** slab **fp64**/**fp64mp**/**fp32full** geom **0.01025×** / **0.01025×** / **0.005806×**; block-cyclic **fp64** **0.001590×**. **(q)** (one row each): **fp64** **6.594×**, **`fp64mp`** **6.715×**, **`fp32full`** **2.247×**.

**Gates:** With placeholder vendor timings, **`--strict-perf`** / aggregate floors are useful only after swapping in **real** vendor medians from **`cusolverMp_geqrf_bench`** (or recorded vendor lines). **`SKIP_TF32_SMOKE=1`** documents TF32 gaps without aborting the script.

**Tooling chain:** `run_*_sweep.sh` → archive stdout → `python3 parse_bench_log.py log --json | python3 postprocess_sweep.py --json-out agg.json` → `python3 gen_executive_dashboard.py agg.json`.

**Next run:** `./build.sh` (set **`MPI_INCLUDE`** if needed), **`export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${CUDA_HOME}/lib64:…"`** before **`mpirun`**, then **`./run_smoke_validation.sh`** (omit **`SKIP_TF32_SMOKE`** when TF32 is available).
