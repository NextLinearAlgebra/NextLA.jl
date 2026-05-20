# Session Handoff — BC 2.5D QR benchmark suite

**Session date:** 2026-05-20
**Branch:** `fda/scqr3-schur` (this commit + ancestors)
**Author of session:** Claude Opus 4.7 (co-authored commit)

## ⚠ Machine-portability note

This handoff is written for a fresh agent on **any machine that has this
repo cloned** (`<REPO>` = the `NextLA.jl/` checkout root in your
workspace).  The original work ran on a single-node SLURM box with 8×H200
GPUs (NVLink, ~141 GB HBM each).  Three pieces are machine-specific and
should be re-derived in your environment:

| Item | Original (dgx01) | What to do on a fresh machine |
|------|-----------------|-------------------------------|
| Toolchain paths | `<conda>/bin`, `<conda>/lib` (miniforge) | Point `PATH` / `LD_LIBRARY_PATH` / `CONDA_PREFIX` at your CUDA + MPI + NCCL install (must include cuBLAS, cuSOLVER, NCCL ≥ 2.7) |
| SLURM jobs 3767/3768/3769 | Live, dgx01-specific | Ignore those IDs; resubmit via `<REPO>/scripts/h200_full25d_oom_np{2,4,8}.sbatch` and use your new IDs |
| Hardware | 8×H200 NVLink | Algorithm is layout-agnostic; results numbers will differ |

Everything else — source code, build script, sweep driver, sbatch
wrappers, validation results — is in the repo and portable.

## TL;DR of the work

Butterfly Tournament-TSQR (Path-h) and butterfly tournament-Givens (Path-g)
replace gather-and-replicate panel reductions.  Phase Q5 look-ahead with
Q2 column-split and a duplicate `nccl_col_la` communicator wired into all
four BC 2.5D variants (sCQR3, Householder, Givens, QDWH).  QDWH now works
at Pz>1 (rearrange + single Dgemm replaces the old block-loop polar
reconstruction).  All variants validate to FP64 machine precision.  Three
12h sbatch wrappers (NP=2, 4, 8) drive an OOM-bounded sweep.  Givens is
excluded from the production sweep — its single-block tournament kernel
is Θ(b·log m) sequential and consumes hours per cell at N ≥ 6144.

## What was built in this session

### Algorithmic optimizations (Phase Q1 + Phase Q5)

The manuscript `<REPO>/qr_schur_xpartition.tex` is the algorithmic
contract — read §A.3 (Conflux SC'21 grid math), §Phase Q3_h / Q3_g
(butterfly TSQR / tournament-Givens), §Phase Q5 (look-ahead).  The
auto-derived schedule (`compute_derived_schedule`) and runtime
fast-memory budget (`nextla_device_fast_memory_budget_elements`) are
**not** changed by this session; only Phase Q1's panel-reduction kernel
and Phase Q5 scheduling are modified.

1. **Butterfly Tournament-TSQR for Path-h Householder**
   (`<REPO>/cpp_bench/householder_bc25d.inl`): replaced gather-and-
   replicate (`m·b` bandwidth) with `log₂(P_r)` butterfly stages of NCCL
   Send/Recv exchanging `b²` R-blocks.  Per-rank bandwidth
   `2·b²·log₂(P_r)`, matches Demmel et al. 2012 / Phase Q3_h bound
   exactly.  Per-rank local-Q reconstructed from stage reflectors via
   per-stage Dorgqr + half-block Dgemm.

2. **Butterfly tournament-Givens for Path-g**
   (`<REPO>/cpp_bench/givens_bc25d.inl`): same butterfly structure with
   `givens_panel_kernel` as the per-stage QR; Q reconstructed from
   `form_Q_from_givens_kernel` applied to stage-stored rotation lists.

3. **Item 4 — ping-pong G_self extract** (householder + givens BC25D
   files): eliminated `log₂(P_r)` per-stage `d_G_tmp → d_G_self` memcpys
   via host-side pointer swap.  Free; saves ~6 μs per panel.

4. **Items 1+2 unified — Look-ahead + Q2 column-split** (all four
   `*_bc25d.inl` files): gated on `--la`.  Q2 split column-wise into
   `Q2_next` (panel-(k+1)'s local cols on Q1-owner ranks) and `Q2_rest`.
   Panel-(k+1) Q1 runs on a dedicated `s_la` stream with duplicate
   scratch, concurrent with `Q2_rest` on `s_comp`.
   - New stream: `s_la` + dedicated `s_comm_la` for the LA NCCL
     collectives so they don't serialize on `s_comm`.
   - New comm: `S.nccl_col_la` (added to `Full25DSubcomms` in
     `<REPO>/cpp_bench/full25d_grid.hpp`).
   - Duplicate scratch context (`HhBcPanelCtx pri`, `la` for Householder;
     analogous structs in scqr3, givens, qdwh).

5. **QDWH at Pz > 1** (`<REPO>/cpp_bench/qdwh_bc25d.inl`): the old polar
   reconstruction `for r in [0, col_size): P += Q1_recv[r] · Q2_recv[r]^T`
   indexed both buffers by `col_size` even though `Q1_recv` only has
   `row_size` blocks — failed at `col_size != row_size` with
   `MPI_Abort(92)`.  Replaced with:
   - Two rearrange kernels (`qdwh_bc25d_q1_recv_to_full`,
     `qdwh_bc25d_q2_recv_to_full`) that flatten rank-block AllGather
     output into contiguous column-major `Q1_full` (`locr × N`) and
     `Q2_full` (`N × locc`).
   - **Single contiguous Dgemm**
     `cublasDgemm(N, N, locr, locc, N, …, Q1_full, locr, Q2_full,
     col_size·locr, …, d_P, locr)`.
   - Works for any `(col_size, row_size, locr, locc)`; validated at
     Pz=2 with `max|diag(UᵀU)−1| = 0.00e+00`.

6. **Rectangular-grid dispatch** (in `householder_2p5d_bench.cu`,
   `givens_2p5d_bench.cu`, `qdwh_2p5d_bench.cu` mains): relaxed
   `use_full25d = (Pz>1 || (Px>1 && Py>1))` to also fire when
   `A.bc25d_layout && P > 1`, so 2-GPU runs with `[1, 2, 1]` grid take
   the BC25D path instead of falling through to legacy slab.

7. **`snap_b_to_divisor`** (`<REPO>/cpp_bench/bc25d_helpers.cuh`): the
   auto-derived `b ≈ √M` doesn't always divide N (e.g., at NP=2 N=2048
   the derived b is 1448).  BC25D dispatch in each `.cu` main now calls
   this helper to snap b down to the largest divisor of N still within
   the §A3b admissible window `c ≤ b ≤ N/√P₁`.  Logs `snapping b X → Y`
   so the runtime decision is auditable.

### Sweep infrastructure

| File | Purpose |
|------|---------|
| `<REPO>/cpp_bench/run_full25d_oom_sweep.sh` | Main driver.  Env-var parameterized: `NP`, `SIZES`, `GIVENS_CAP`, `QDWH_CAP`, `NEXTLA_BENCH_RUNS`. |
| `<REPO>/scripts/h200_full25d_oom_np2.sbatch` | 2-GPU wrapper, 12h, `--partition=large`, `GIVENS_CAP=0` (skips Givens). |
| `<REPO>/scripts/h200_full25d_oom_np4.sbatch` | 4-GPU wrapper. |
| `<REPO>/scripts/h200_full25d_oom_np8.sbatch` | 8-GPU wrapper. |

Size ladders per NP (defined in the main sweep script):
- NP=2: `2048 4096 6144 8192 12288 16384 24576 32768 49152 65536`
- NP=4: same plus `98304`
- NP=8: same plus `131072`

QDWH capped at N=49152 (its 2N×N stacked panel OOMs ~half the other
variants).

## Validation status (from source machine)

All four BC 2.5D variants validate at FP64 machine precision in both
`--la` and `--no-la`, across all matrix modes, at every grid tested:

| Variant | Grid | LA | `max|diag(QᵀQ)−1|` |
|---|---|---|---|
| scqr3 | Px=Py=Pz=2 | both | 6.66e-16 – 1.44e-15 |
| householder (butterfly TSQR) | Px=Py=Pz=2 | both | 1.78e-15 |
| givens (butterfly tournament) | Px=Py=Pz=2 | both | 1.55e-15 |
| qdwh | Px=Py=2 Pz=1 | both | 0.00e+00 |
| qdwh | Px=Py=Pz=2 | both | 0.00e+00 – 3.11e-15 |
| qdwh | Px=1 Py=2 Pz=1 (NP=2) | both | 0.00e+00 |

A fresh agent should re-run validation in the target environment with:

```bash
cd <REPO>/cpp_bench
mpirun --map-by :OVERSUBSCRIBE -np 8 ./householder_2p5d_bench \
       --N=2048 --matrix=fp64 --la
# Look for "max|diag(Q'Q)-1| = ..." in the output.
```

## Known issues / caveats

### 1. cuSolverMp library — vendor-baseline NA on the source machine

On dgx01 with `libcusolverMp.so.0.8.0.0`, every grid tested (1×1, 2×1,
2×2, 4×2 at N ∈ {2048, 4096, 8192, 16384, 32768, 65536}) returned
`CUSOLVER_STATUS_INTERNAL_ERROR` (error 7) inside `cusolverMpGeqrf`.  The
wrapper code in `<REPO>/cpp_bench/cusolverMp_geqrf_bench.cpp` is
unchanged from prior runs.

On a fresh machine the cuSolverMp behavior **may differ** — try the
wrapper directly at one size before assuming the baseline is broken:

```bash
mpirun -np 8 <REPO>/cpp_bench/cusolverMp_geqrf_bench 8192 1024 1024 4 2
```

If it succeeds, the sweep at NP=8 picks up vendor numbers automatically.
If it fails the same way, our variants still produce their full sweep
(vendor column = NA).  The sweep script categorizes failures:
`[BASELINE-FAIL]` line for cuSolverMp refusals, plain METRICS lines for
our variants.

### 2. Givens is fundamentally single-block on the panel

`givens_panel_kernel<<<1, 256, …>>>` is single-block by design.
Empirical per-cell wall-clock on the source machine:
- N=2048: ~10.9 s/run
- N=4096: ~102.8 s/run
- N=6144: ~382.2 s/run
- N=8192: ~855 s/run (≈ 14 min/run × 5 timed runs → > 1 hour/cell)

Matches manuscript's `Θ(b²·m·log m / threads)`.  The production sbatch
wrappers set `GIVENS_CAP=0` to skip Givens.  Run a separate small-N
Givens sweep (`GIVENS_CAP=2048`) if you want Path-g data points.

A real fix would be a multi-block cooperative-launch kernel — not done
this session.

### 3. Strong scaling reverses at small N

Observed in the source-machine partial first round: NP=2 → NP=4 at
N=2048 is *slower* for every variant by 10×–25×.  Reason: at NP=2 the
grid is `[1, 2, 1]` → `col_size = Px·Pz = 1` → AllReduces are no-ops; at
NP=4 grid `[2, 2, 1]` → `col_size = 2` → NCCL latency per panel becomes
visible.  Manuscript predicts the 2.5D speedup arrives at N ≥ ~30,000
where compute dominates communication.

### 4. LA effect is variant-dependent at small N

From the source-machine partial run (NP=2 fp64):
- **sCQR3 +LA wins big**: 99.8 → 17.2 ms at N=2048 (6× speedup).  Phase Q5
  prediction holds for the 3-pass panel.
- **Householder +LA hurts at small N**: 15.7 → 46.1 ms.  Butterfly TSQR
  panel is too light; LA-stream overhead exceeds gain.
- **QDWH +LA hurts**: outer Halley loop overhead dominates inner gain.

LA should pay off uniformly at N ≥ 32k.  Worth checking in the target
environment.

## Repo / file map (all paths relative to `<REPO>` = `NextLA.jl/`)

```
cpp_bench/
├── derived_schedule.hpp           # Conflux (c, Px, Py, Pz, b) auto-derivation
├── nextla_fast_memory.hpp         # Runtime M extraction (HBM × frac)
├── full25d_grid.hpp               # [Px,Py,Pz] grid resolve, col_comm + row_comm
│                                     + nccl_col_la (added this session)
├── full25d_kernels.cuh            # rearrange + d2f/f2d cast kernels
├── bc25d_helpers.cuh              # numroc BC indexing + snap_b_to_divisor
├── tsqr_butterfly.cuh             # NEW: shared butterfly scaffolding
│                                     (partner XOR, send/recv, eye, copy_upper_R)
├── scqr3_bc25d.inl                # Path-s: SYRK+AllReduce(G)+POTRF+TRSM
├── householder_bc25d.inl          # Path-h: butterfly TSQR panel
├── givens_bc25d.inl               # Path-g: butterfly tournament-Givens
├── qdwh_bc25d.inl                 # Path-q: Halley + Pz>1 polar fix
├── scqr3_full25d_bench.cu         # main() with --layout=bc25d default
├── householder_2p5d_bench.cu
├── givens_2p5d_bench.cu
├── qdwh_2p5d_bench.cu
├── cusolverMp_geqrf_bench.cpp     # NVIDIA reference (broken on source machine)
├── run_full25d_oom_sweep.sh       # The sweep driver
└── build.sh                       # nvcc + mpi build script

scripts/
├── h200_full25d_oom_np2.sbatch    # GIVENS_CAP=0 + 12h limit
├── h200_full25d_oom_np4.sbatch
└── h200_full25d_oom_np8.sbatch

qr_schur_xpartition.tex            # Algorithmic contract
SC_factorization.pdf               # Conflux SC'21 (pages 7-8 for BC 2.5D + butterfly)
```

## How to build and run from scratch (target machine)

1. Ensure your environment has: CUDA toolkit (cuBLAS + cuSOLVER), MPI
   (OpenMPI or MPICH), NCCL ≥ 2.7, a recent g++ / nvcc (CUDA ≥ 11).

2. Point the build at your install:
   ```bash
   cd <REPO>/cpp_bench
   # Adjust these to YOUR install prefixes (the source machine used
   # /home/<user>/miniforge3, but yours will differ):
   export PATH=<your-conda-or-cuda>/bin:$PATH
   export LD_LIBRARY_PATH=<your-conda-or-cuda>/lib:${LD_LIBRARY_PATH:-}
   # build.sh autodetects MPI via `mpicc --showme:compile`; or set
   # MPI_INCLUDE explicitly to the directory containing mpi.h.
   ./build.sh
   ```

3. Smoke test:
   ```bash
   mpirun --map-by :OVERSUBSCRIBE -np 8 ./householder_2p5d_bench \
          --N=2048 --matrix=fp64 --la
   # Expect tmed ~40-50 ms on H200; "max|diag(Q'Q)-1| ~ 1.8e-15"
   ```

4. Run a sweep (single NP count, ~1-4 h depending on hardware):
   ```bash
   cd <REPO>
   sbatch scripts/h200_full25d_oom_np8.sbatch
   # Output appears in <REPO>/logs/full25d_oom_np8-<JOBID>.out
   ```

   You may need to edit the sbatch wrapper if your SLURM cluster has a
   different partition name or different `--gres` syntax.

## METRICS line format (sweep output)

Every successful cell emits exactly one METRICS line.  Variant-specific:

```
METRICS bench=scqr3_bc25d matrix=fp64 layout=bc_2p5d panel=scqr3+la N=2048 b=1024 Px=1 Py=2 Pz=1 passes=3 vendor_fp64_ms=NA vendor_fp32_ms=NA ours_ms=17.2107
METRICS bench=householder_bc25d matrix=fp64 layout=bc_2p5d panel=tsqr_butterfly+la N=2048 b=1024 Px=1 Py=2 Pz=1 passes=1 ours_ms=46.1302
METRICS bench=qdwh_bc25d matrix=fp64 layout=bc_2p5d panel=cqr2+la N=2048 b=1024 Px=1 Py=2 Pz=1 passes=6 ours_ms=58.2076
```

- LA on ⇒ `panel=*+la` substring; LA off ⇒ no `panel=` (scqr3) or
  `panel=tsqr_butterfly` / `butterfly_givens` without `+la`.
- `passes=` distinguishes sCQR3 (3) from CQR2 (2) within the scqr3 bench.
- `vendor_*_ms=NA` if cuSolverMp baseline failed; numeric if it ran.
- `ours_ms` is **median of 5 timed runs after 2 warmups**.

Quick parse:
```python
rows = []
for line in open(log_path):
    if not line.startswith("METRICS "): continue
    d = {k:v for k,v in (tok.split("=",1) for tok in line.split() if "=" in tok)}
    d["la"] = "la" if d.get("panel","").endswith("+la") else "no-la"
    rows.append(d)
```

## What to do next

### Immediate
1. **Verify build + smoke** on the target machine (steps 1-3 above).
2. **Submit sweeps**:
   ```bash
   cd <REPO>
   sbatch scripts/h200_full25d_oom_np2.sbatch
   sbatch scripts/h200_full25d_oom_np4.sbatch
   sbatch scripts/h200_full25d_oom_np8.sbatch
   ```
   Save the new job IDs.

### After jobs complete
1. **Build the result table**: 4 variants × 4 modes × 2 LA × N-ladder × 3 NP
   ≈ 700–900 rows.  Use the Python snippet above.
2. **Plots**: ms vs N per variant for each (NP, matrix-mode, LA); strong
   scaling NP=2 vs NP=4 vs NP=8 at each N; LA on/off ratio vs N.
3. **Sanity-check OOM-Ns** against the manuscript's `c = PM/N²` bound.

### Recommended follow-ups
1. **cuSolverMp**: try a different library version (the failing one on the
   source machine is 0.8.0).  Library installs vary by package manager.
2. **Givens dedicated short sweep**: `GIVENS_CAP=2048` at N≤2048 only, all
   NP × mode × LA.  ~30 min wall-clock.  Produces the missing Path-g row.
3. **Multi-block tournament-Givens**: cooperative-group kernel rewrite so
   Path-g can reach larger N.  Substantial work; not required for the
   paper.
4. **Tolerance check**: confirm `max|diag(QᵀQ)−1|` per (variant, matrix,
   LA, N) stays within manuscript bounds (`≤ 50·m·b·u` for FP64,
   `≤ 10·m·b·u_fp32` for FP32).

## User preferences observed in this session

- **Faithful to the manuscripts**: 2.5D attribution, scheduling, parameter
  selection from `compute_derived_schedule` must remain unchanged; all
  optimizations are *layered on top*.  Don't change the auto-derivation
  itself.
- **Honest about limitations**: prefer corrected numbers over over-claimed
  ones; when cuSolverMp fails, report NA rather than hide the issue.
- **Communication-optimal where possible**: butterfly TSQR + butterfly
  tournament-Givens replace gather-and-replicate so Path-h/Path-g achieve
  the manuscript's `b² log P_r` bound.
- **Math-proven before adding**: the user asked for a proof that butterfly
  wouldn't help Path-s / Path-q before considering adding it.  Bandwidth
  proof (`≤ 6b²` constant beats `2b² log P_r` at large P_r) ruled out
  butterfly for those paths.

## Open questions for the next session

1. Does the cuSolverMp baseline work on this machine?
   `mpirun -np 8 <REPO>/cpp_bench/cusolverMp_geqrf_bench 8192 1024 1024 4 2`
2. Does the LA crossover materialize at N ≥ 16k as predicted?
3. Does strong scaling NP=4/NP=8 overtake NP=2 at large N as predicted?
4. Is `max|diag(QᵀQ)−1|` within tolerance at all sweep points, or are
   there outliers?
