# NextLA sCQR3/QR Phase Plan (Function-by-Function)

## Scope
- Implement file-by-file with tests after each file.
- Skip Schur until QR is solid.
- Order: xpartition -> scqr3 -> geqrf_2p5d -> tests (after each file).

---

## File: src/xpartition.jl

### DeviceParams{T}
- [X] Define fields for P, M, b, c, P1, Px, Py, Pz, TILE_DIM, b_min, b_max, AI_target.
- [X] Add a simple constructor or `compute_params` fills all fields.

### probe_device(backend, ::Type{T})
- [x] Implement backend-agnostic fallback: P = max(1, Threads.nthreads()), M = 16384 / sizeof(T).
- [x] Add optional dispatch stubs for CUDA/AMDGPU/oneAPI/Metal (query SM count + shared mem) with safe defaults.
- [x] Keep signature generic for any KernelAbstractions backend.

### compute_params(backend, ::Type{T}, N; b=nothing, c=nothing)
- [X] Compute c = floor(P*M / N^2), clamp to >= 1.
- [X] Set P1 = P ÷ c, Px = Py = isqrt(P1), Pz = c.
- [X] Set TILE_DIM = floor(Int, sqrt(M)).
- [X] Set b_min = c, b_max = N ÷ Px.
- [X] Select b if not provided; validate if provided.
- [X] Emit warning when falling back to c = 1.

### panel_cu_set(k, params)
- [X] Return iterable of (r, j_k, z) for r in 0:Px-1, z in 0:c-1.

### block_owner(IJ, params)
- [X] Map (I,J) to (I mod Px, J mod Py, 0:c-1).

### workgroup_reduce!(out, src; op=+, N=256)
- [X] Launch workgroup_reduce_kernel!.
- [X] Validate N power-of-two and length(src) <= N.

### panel_allreduce!(G, partials, params)
- [X] Compute panel size and power-of-two tree size.
- [X] Launch panel_allreduce_kernel!; no-op when c == 1.

### verify_budget(params)
- [X] Host-side asserts for paper constraints.
- [X] Launch verify_budget_kernel! in debug builds (optional).

### @kernel workgroup_reduce_kernel!(out, src, op, ::Val{N})
- [X] Load into local buffer with bounds checks.
- [X] Tree-reduce with @synchronize between steps.
- [X] Write out[1].

### @kernel panel_allreduce_kernel!(G, partial, ::Val{TREE})
- [X] Tree-reduce panel G across TREE replicas.
- [X] Ensure deterministic partner order.

### @kernel verify_budget_kernel!(ok, P, M, N, b, c)
- [X] Write ok[1] based on equality and bounds.

#### Tests after xpartition
- File: test/xpartition.jl
- [X] compute_params window checks
- [X] reduction determinism (run twice)
- [X] verify_budget sanity

---

## File: src/scqr3.jl

### scqr3!(m, b, A_panel, R, G, info; params)
- [X] Three-iteration loop: Gram -> reduce -> shift -> chol -> TRSM.
- [X] Throw PosDefException on failure.
- [X] Accumulate R product.

### scqr3!(A_panel, R; params=nothing)
- [X] Allocate G, info, params if needed.
- [X] Call low-level scqr3!.

### @kernel scqr3_gram_kernel!(G_local, A_panel, m, b, ::Val{TILE})
- [X] Tile loads with @localmem and @synchronize.
- [X] Complex-safe conjugation pattern.

### @kernel scqr3_cholesky_kernel!(G, b)
- [X] Serial Cholesky, write info on failure.

### @kernel scqr3_trsm_kernel!(A_panel, R, ::Val{TILE})
- [X] Right-side TRSM kernel (or delegate to trsm!).

#### Tests after scqr3
- File: test/scqr3.jl
- [X] Orthogonality and residual checks
- [X] PosDefException on rank-deficient panel
- [X] Determinism

---

## File: src/geqrf_2p5d.jl

### geqrf_2p5d!(m, n, A, R_acc, tau; params=nothing, b=nothing)
- [ ] Outer loop over panels.
- [ ] Call scqr3! per panel.
- [ ] Compute W = Q' * A_trailing.
- [ ] Reduce W across replicas (if c > 1).
- [ ] Apply trailing update A -= Q*W.

### geqrf_2p5d!(A)
- [ ] Allocate params, R_acc, tau, scratch.
- [ ] Call low-level geqrf_2p5d!.

### @kernel qta_tile_kernel!(W_local, Q, A, k, b, ::Val{TILE})
- [ ] Tiled GEMM-like kernel using @localmem.

### @kernel apply_trailing_kernel!(A, Q, W, k, b, ::Val{TILE})
- [ ] Tiled GEMM-like kernel for A -= QW.

### @kernel broadcast_panel_kernel!(Qk_replicated, Qk_local, ::Val{Px})
- [ ] Stub for single-device (no-op or copy).

#### Tests after geqrf_2p5d
- File: test/geqrf_2p5d.jl
- [ ] End-to-end QR residual + orthogonality
- [ ] Non-multiple-of-b panel case
- [ ] 2D fallback (c = 1)

---

## Deferred until QR is solid
- src/hessenberg.jl
- src/bulge_chase.jl
- src/qdwh.jl
- src/schur.jl
