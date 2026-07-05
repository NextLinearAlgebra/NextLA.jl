export compress!

# ─── Kernels ──────────────────────────────────────────────────────────────────

@kernel function _copy_diag_kernel!(D::AbstractArray{T,3},
    A::AbstractMatrix{T}, tile_m::Int, tile_n::Int) where {T}
    row, col, b = @index(Global, NTuple)
    p0 = (b - 1) * tile_m + 1
    q0 = (b - 1) * tile_n + 1
    @inbounds D[row, col, b] = A[p0+row-1, q0+col-1]
end

"""
    _copy_diagonal_from_dense!(A_tlr, A) -> A_tlr

Populate `A_tlr`'s dense diagonal storage (`A_tlr.D` and, if present,
`A_tlr.D_corner`) from the corresponding tiles of dense matrix `A`. Called
once at the start of [`compress!`](@ref), before any off-diagonal sketching.
"""
function _copy_diagonal_from_dense!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}) where {T}
    n_full_diag = _nfull_diag_tiles(A_tlr)
    _copy_diag_kernel!(A_tlr.backend)(
        A_tlr.D, A, A_tlr.tile_m, A_tlr.tile_n;
        ndrange=(A_tlr.tile_m, A_tlr.tile_n, n_full_diag),
    )
    if size(A_tlr.D_corner, 3) != 0
        tile_k = ndiag_tiles(A_tlr)
        p0, q0 = tile_origin_coords(A_tlr, tile_k, tile_k)
        tm, tn = tile_size(A_tlr, tile_k, tile_k)
        copyto!(view(A_tlr.D_corner, 1:tm, 1:tn, 1), view(A, p0:(p0 + tm - 1), q0:(q0 + tn - 1)))
    end
    return A_tlr
end

#= Squared magnitude accumulated in Float64: the error indicator is a difference
   of two nearly equal sums, so per-element squaring must not round in T. =#
@inline _abs2_f64(x::Real) = abs2(Float64(x))
@inline _abs2_f64(x::Complex) = abs2(Float64(real(x))) + abs2(Float64(imag(x)))

#= Per-slab Cholesky-QR shift from the Gram diagonal:
     rescue pass:  √eps · tr(G)/r      — rescues rank-deficient sketches
     refine pass:  eps · r · max(diag) — eps-level, so the second pass undoes the
                   column-norm deflation the rescue shift introduced
   A zero Gram (zero tile) gets shift 1: potrf stays PD and Y stays zero. =#
@kernel function _cholqr_shift_compute_kernel!(regs::AbstractVector{RT},
    G::AbstractArray{Thi,3}, trace_coeff::RT, maxdiag_coeff::RT) where {RT,Thi}
    b = @index(Global, Linear)
    r = size(G, 1)
    tr = zero(RT)
    mx = zero(RT)
    @inbounds for i in 1:r
        d = RT(real(G[i, i, b]))
        tr += d
        mx = max(mx, d)
    end
    reg = trace_coeff * (tr / r) + maxdiag_coeff * mx
    @inbounds regs[b] = ifelse(reg > zero(RT), reg, one(RT))
end

# Adds the per-slab shift regs[b] (from _cholqr_shift_compute_kernel!) to diag(G).
@kernel function _cholqr_shift_apply_kernel!(G::AbstractArray{T,3}, regs::AbstractVector) where {T}
    j, b = @index(Global, NTuple)
    @inbounds G[j, j, b] += T(regs[b])
end

#= Per-tile squared Frobenius norm of the dense source tiles, accumulated in
   Float64 — the reference term of the error indicator ‖A‖² − ‖V‖².
   One workgroup per tile; R threads stride over the tile columns. =#
@kernel function _tile_norm_sq_kernel!(out::AbstractVector{Float64},
    A::AbstractMatrix{T}, p0s, q0s, tm::Int, tn::Int, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    partial = @localmem Float64 (R,)
    p0 = Int(@inbounds p0s[ob]) - 1
    q0 = Int(@inbounds q0s[ob]) - 1
    acc = 0.0
    col = j
    while col <= tn
        @inbounds for row in 1:tm
            acc += _abs2_f64(A[p0+row, q0+col])
        end
        col += R
    end
    @inbounds partial[j] = acc
    @synchronize
    if j == 1
        total = 0.0
        @inbounds for i in 1:R
            total += partial[i]
        end
        @inbounds out[ob] = total
    end
end

#= Fused norm + descending sort + greedy threshold + column gather.
   One workgroup per off-diagonal tile; R = r_eff threads per workgroup.

   The squared error of the truncated factorization decomposes (U orthonormal) as
     ‖A − U_k V_kᴴ‖² = resid + Σ_{dropped} ‖v_j‖²,  resid = ‖A‖² − Σ_j ‖v_j‖²
   where `resid` is the range-capture error the sketch itself left behind
   (the randQB_EI error indicator).  The greedy drop below spends only the
   budget that remains after accounting for it, and the total goes to `err_sq`.

   The columns are ordered by a parallel rank sort (each thread counts the
   columns that outrank its own, ties broken by index) rather than a serial
   bubble sort, and the gather runs threads-along-rows so the workgroup's
   accesses are contiguous in the column-major leading dimension. =#
@kernel function _fused_truncate_kernel!(
    U_out::AbstractArray{T,3}, V_out::AbstractArray{T,3},
    U::AbstractArray{T,3}, V::AbstractArray{T,3},
    rk::AbstractVector, err_sq::AbstractVector{Float64},
    normA_sq::AbstractVector{Float64},
    eps_sq::Float64, rel::Bool, ::Val{R}) where {T,R}
    ob = @index(Group, Linear)
    j = @index(Local, Linear)
    norms        = @localmem Float64 (R,)   # unsorted column norms
    sorted_norms = @localmem Float64 (R,)   # norms in descending order
    sorted_src   = @localmem Int32   (R,)   # source column of each sorted slot
    k_buf = @localmem Int32 (1,)

    #= Phase A — squared norm of each V column (thread ↔ column).
       (locals do not survive @synchronize — KA splits the body there — so sizes
       are recomputed in each phase rather than hoisted.) =#
    acc = 0.0
    @inbounds for row in 1:size(V, 1)
        acc += _abs2_f64(V[row, j, ob])
    end
    @inbounds norms[j] = acc
    @synchronize

    #= Phase B — parallel rank sort: thread j places its column at the descending
       slot given by the count of stronger columns (ties broken by index, so the
       ordering is a total order identical to a stable descending sort). =#
    my_norm = @inbounds norms[j]
    slot = 1
    @inbounds for i in 1:R
        ni = norms[i]
        (ni > my_norm || (ni == my_norm && i < j)) && (slot += 1)
    end
    @inbounds sorted_norms[slot] = my_norm
    @inbounds sorted_src[slot] = Int32(j)
    @synchronize

    # Phase C — EI-corrected budget accounting on the sorted norms (thread 1).
    if j == 1
        total = 0.0
        @inbounds for i in 1:R
            total += sorted_norms[i]
        end
        nA_sq = @inbounds normA_sq[ob]
        resid = max(nA_sq - total, 0.0)
        #= resid = ‖A‖² − ‖V‖² is a difference of two O(‖A‖²) sums, so it cannot
           resolve range-capture error below the rounding floor of V (formed in T
           with GEMM inner dimension size(U,1)). Below that floor the value is pure
           cancellation noise — treat the range as captured, or an oversampled
           sketch would report a spurious residual and never truncate. A genuinely
           under-captured range sits orders of magnitude above the floor and
           survives, still driving the FAIL path. =#
        epsT = Float64(eps(real(T)))
        resid_floor = Float64(size(U, 1)) * epsT * nA_sq
        resid = ifelse(resid < resid_floor, 0.0, resid)
        #= A precision-T sketch resolves relative error no finer than √eps(T), so a
           requested tol below that floor cannot be honoured: the sketch's noise
           columns then straddle the budget and inflate the rank by 1–2 spurious
           columns. Floor the drop budget at eps(T)·‖A‖² so those columns are
           dropped and the true rank is recovered; the reconstruction error then
           sits at the √eps(T) floor rather than at the (unreachable) tol. =#
        target = rel ? eps_sq * nA_sq : eps_sq
        budget = max(target, epsT * nA_sq) - resid
        k_val = R
        dropped = 0.0
        if budget >= 0.0
            @inbounds for idx in R:-1:1
                cost = sorted_norms[idx]
                cost <= budget || break
                budget -= cost
                dropped += cost
                k_val -= 1
            end
        end
        rk[ob] = eltype(rk)(k_val)
        err_sq[ob] = resid + dropped
        k_buf[1] = Int32(k_val)
    end
    @synchronize

    #= Phase D — coalesced gather: loop over output columns, threads stride the
       rows so consecutive threads touch consecutive (contiguous) memory. =#
    k = Int(@inbounds k_buf[1])
    bU = size(U, 1)
    bV = size(V, 1)
    @inbounds for jj in 1:R
        if jj <= k
            src = Int(sorted_src[jj])
            row = j
            while row <= bU
                U_out[row, jj, ob] = U[row, src, ob]
                row += R
            end
            row = j
            while row <= bV
                V_out[row, jj, ob] = V[row, src, ob]
                row += R
            end
        else
            row = j
            while row <= bU
                U_out[row, jj, ob] = zero(T)
                row += R
            end
            row = j
            while row <= bV
                V_out[row, jj, ob] = zero(T)
                row += R
            end
        end
    end
end

# ─── Tile-view helpers ────────────────────────────────────────────────────────

"""
    _tile_views(A, A_tlr, obs) -> Vector{<:AbstractMatrix}

Dense-source views for each off-diagonal tile in `obs`, in category-local
order. Shared with `uncompress.jl`.
"""
function _tile_views(A::AbstractMatrix, A_tlr::TLRMatrix, obs::Vector{Int})
    return map(obs) do ob
        lin = _linear_from_offdiag(A_tlr, ob)
        ti, tj = _inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, ti, tj)
        tm, tn = tile_size(A_tlr, ti, tj)
        view(A, p0:(p0+tm-1), q0:(q0+tn-1))
    end
end

"""
    _batch_views(A, r=size(A, 2)) -> Vector{<:AbstractMatrix}

Per-batch-entry views into a `[m, maxrank, n]` factor array, trimmed to the
first `r` columns. Used to build `gemm_batched!` operand vectors; shared with
`uncompress.jl`.
"""
@inline _batch_views(A::AbstractArray{T,3}, r::Int=size(A, 2)) where {T} =
    [view(A, :, 1:r, k) for k in axes(A, 3)]

# ─── Precision helpers ────────────────────────────────────────────────────────

#= Accumulation precision used for orthogonalisation (Cholesky-QR) and norm
bookkeeping. Use higher precision accumulation for better numerical stability. =#
@inline _compress_accum_type(::Type{Float16}) = Float32
@inline _compress_accum_type(::Type{Float32}) = Float64
@inline _compress_accum_type(::Type{Float64}) = Float64
@inline _compress_accum_type(::Type{ComplexF32}) = ComplexF64
@inline _compress_accum_type(::Type{ComplexF64}) = ComplexF64
@inline _compress_accum_type(::Type{T}) where {T} = T

"""
    _adjoint_blas_char(T) -> Char

BLAS transpose flag for the conjugate transpose of element type `T`: `'C'` for
complex types, `'T'` otherwise.
"""
@inline _adjoint_blas_char(::Type{<:Complex}) = 'C'
@inline _adjoint_blas_char(::Type) = 'T'

# ─── Workspace structs ────────────────────────────────────────────────────────

"""
    CompressCategoryWorkspace

Scratch buffers and cached views for one off-diagonal tile category
(interior, right-boundary, or bottom-boundary) during [`compress!`](@ref).
`U`/`V` alias the corresponding `A_tlr` panel; every other field is scratch
carved from the flat buffers in [`CompressWorkspace`](@ref) by
[`_carve_category_workspace`](@ref).

One buffer is reused across pipeline steps rather than separately allocated:
- `V` doubles as Ω: filled with `randn!` for the range sketch (step 1), then
  overwritten by the co-range `Aᴴ·U` (step 3).
- `U_tmp` and `V_tmp` hold the truncation kernel's sorted output before it is
  copied back into `U`/`V`.

See [`_compress_category!`](@ref) for the step-by-step pipeline.
"""
struct CompressCategoryWorkspace{
    ObsT  <: Vector{Int},
    UT    <: AbstractArray,   # aliases A_tlr panel — U factors + sketch target
    VT    <: AbstractArray,   # aliases A_tlr panel — Omega initially, then V = A^T U
    YHiT  <: AbstractArray,   # high-precision U copy for cholqr (r_eff wide)
    GHiT  <: AbstractArray,   # high-precision Gram matrix for cholqr (r_eff²)
    UTmpT <: AbstractArray,   # sorted-U output (truncation)
    VTmpT <: AbstractArray,   # sorted-V output (truncation)
    TileVT,                   # cached Vector-of-views into U/V (r_eff columns)
    RanksT,
    ErrT,                     # device Float64 — per-tile squared error estimate
    NormT,                    # device Float64 — per-tile ‖A_tile‖²_F
    CoordT,                   # device Int32 — tile origin coordinates
    RegT,                     # device real(Thi) — per-slab cholqr shifts
}
    obs::ObsT
    r_eff::Int                # sketch width: min(maxrank, tile_m, tile_n)
    U::UT
    V::VT
    U_tiles::TileVT           # cached GEMM operand views (rebuilt only on alloc)
    V_tiles::TileVT
    Y_hi::YHiT
    G_hi::GHiT
    U_tmp::UTmpT
    V_tmp::VTmpT
    ranks_local::RanksT
    err_sq_local::ErrT
    normA_sq::NormT
    p0s::CoordT
    q0s::CoordT
    regs::RegT
end

"""
    CompressWorkspace

Top-level scratch space for [`compress!`](@ref): one
[`CompressCategoryWorkspace`](@ref) per tile category (`interior`, `right`,
`bottom`), backed by two flat device allocations shared across all three —
`buf_T` in working precision `T` (`U_tmp` + `V_tmp`) and `buf_hi` in
accumulation precision `Thi` (`Y_hi` + `G_hi`) — plus one stream per category
for GPU backends. Build with [`alloc_workspace`](@ref) and reuse across calls
on matrices of the same layout.
"""
struct CompressWorkspace{IntWS, RightWS, BottomWS, BufT, BufHiT, StreamV}
    interior::IntWS
    right::RightWS
    bottom::BottomWS
    buf_T::BufT      # flat T-precision backing buffer (U_tmp + V_tmp, all categories)
    buf_hi::BufHiT   # flat Thi-precision backing buffer (Y_hi + G_hi, all categories)
    streams::StreamV
end

# ─── Workspace allocation ─────────────────────────────────────────────────────

"""
    _buf_view(buf, off, dims...) -> reshaped view

Reshaped view of `prod(dims)` elements from the flat buffer `buf`, starting
at 1-based offset `off`. Callers are responsible for keeping `off` in sync
with the element counts predicted by [`_scratch_sizes`](@ref) — see
[`_carve_category_workspace`](@ref).
"""
@inline function _buf_view(buf, off::Int, dims::Vararg{Int})
    len = prod(dims)
    reshape(view(buf, off:off + len - 1), dims...)
end

"""
    _category_specs(A_tlr, b, tail_m, tail_n) -> (interior, right, bottom)

Geometry and storage for the three off-diagonal tile categories, as named
tuples `(; obs, U, V, tm, tn)`. `obs` indexes into `A_tlr.ranks`/`A_tlr.resid`;
`U`/`V` are the corresponding panel arrays; `tm`/`tn` are the tile dimensions
for that category (`tail_m`/`tail_n` may be smaller than `b` at a boundary).
"""
@inline _category_specs(A_tlr, b, tail_m, tail_n) = (
    (; obs=A_tlr.obs_int, U=A_tlr.int_U, V=A_tlr.int_V, tm=b, tn=b),
    (; obs=A_tlr.obs_right, U=A_tlr.right_U, V=A_tlr.right_V, tm=b, tn=tail_n),
    (; obs=A_tlr.obs_bottom, U=A_tlr.bottom_U, V=A_tlr.bottom_V, tm=tail_m, tn=b),
)

"""
    _scratch_sizes(tm, tn, r, n) -> (n_T, n_hi)

Element counts of the working-precision (`n_T`: `U_tmp` + `V_tmp`) and
accumulation-precision (`n_hi`: `Y_hi` + `G_hi`, at `r_eff = min(r, tm, tn)`)
scratch for one category of `n` tiles of size `tm × tn` and sketch width `r`.
Must stay in lockstep with the carving order in
[`_carve_category_workspace`](@ref); used by `alloc_workspace` to size the two
flat backing buffers up front.
"""
@inline function _scratch_sizes(tm, tn, r, n)
    r_eff = min(r, tm, tn)
    ((tm*r + tn*r) * n, (tm*r_eff + r_eff*r_eff) * n)
end

"""
    _tile_origin_vectors(A_tlr, obs, proto) -> (p0s, q0s)

Device vectors of the 1-based `(row, col)` origin of each tile in `obs`,
built on the host and uploaded once. Consumed by `_tile_norm_sq_kernel!` to
locate each tile within the dense source matrix without recomputing tile
geometry on every `compress!` call.
"""
function _tile_origin_vectors(A_tlr::TLRMatrix, obs::Vector{Int}, proto)
    n = length(obs)
    p0_host = Vector{Int32}(undef, n)
    q0_host = Vector{Int32}(undef, n)
    for (k, ob) in enumerate(obs)
        lin = _linear_from_offdiag(A_tlr, ob)
        ti, tj = _inverse_tile_index(A_tlr, lin)
        p0, q0 = tile_origin_coords(A_tlr, ti, tj)
        p0_host[k] = Int32(p0)
        q0_host[k] = Int32(q0)
    end
    p0s = similar(proto, Int32, n)
    q0s = similar(proto, Int32, n)
    copyto!(p0s, p0_host)
    copyto!(q0s, q0_host)
    p0s, q0s
end

"""
    _carve_category_workspace(A_tlr, buf_T, pT, buf_hi, pH, spec, r, rank_type, proto, Thi)
        -> (CompressCategoryWorkspace, pT′, pH′)

Build one category's [`CompressCategoryWorkspace`](@ref): `U_tmp`/`V_tmp`
and `Y_hi`/`G_hi` are views carved from `buf_T`/`buf_hi` at offsets `pT`/`pH`
(advanced and returned as `pT′`/`pH′` for the next category); `U`/`V` alias the
live `A_tlr` panel from `spec` rather than being freshly allocated. The carve
order here must match the element counts [`_scratch_sizes`](@ref) predicted.
"""
function _carve_category_workspace(A_tlr::TLRMatrix, buf_T, pT::Int, buf_hi, pH::Int,
    spec, r::Int, rank_type, proto, ::Type{Thi}) where {Thi}
    n = length(spec.obs)
    r_eff = min(r, spec.tm, spec.tn)
    U_tmp = _buf_view(buf_T,  pT, spec.tm, r,     n); pT += spec.tm*r*n
    V_tmp = _buf_view(buf_T,  pT, spec.tn, r,     n); pT += spec.tn*r*n
    Y_hi  = _buf_view(buf_hi, pH, spec.tm, r_eff, n); pH += spec.tm*r_eff*n
    G_hi  = _buf_view(buf_hi, pH, r_eff,   r_eff, n); pH += r_eff*r_eff*n
    p0s, q0s = _tile_origin_vectors(A_tlr, spec.obs, proto)
    gemm_r = max(r_eff, 1)   # width of the cached GEMM operand views
    U_tiles = _batch_views(spec.U, gemm_r)
    V_tiles = _batch_views(spec.V, gemm_r)
    cat = CompressCategoryWorkspace(
        spec.obs, r_eff, spec.U, spec.V, U_tiles, V_tiles, Y_hi, G_hi, U_tmp, V_tmp,
        similar(proto, rank_type, n),
        similar(proto, Float64, n),
        similar(proto, Float64, n),
        p0s, q0s,
        similar(proto, typeof(real(zero(Thi))), n),
    )
    cat, pT, pH
end

"""
    alloc_workspace(A_tlr) → CompressWorkspace

Pre-allocate all scratch buffers for `compress!`.  Two flat device allocations
(one in working precision T, one in accumulation precision Thi) serve all three
tile categories; U and V alias A_tlr storage directly.  Reuse across repeated
calls on the same matrix layout:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end
"""
function alloc_workspace(A_tlr::TLRMatrix{<:Any,T}) where {T}
    Thi       = _compress_accum_type(T)
    proto     = A_tlr.D
    rank_type = eltype(A_tlr.ranks)
    r  = A_tlr.maxrank
    b  = A_tlr.tile_m
    mt, nt = tilegrid_size(A_tlr)
    tail_m = max(A_tlr.m - (mt - 1) * b, 1)
    tail_n = max(A_tlr.n - (nt - 1) * b, 1)
    specs = _category_specs(A_tlr, b, tail_m, tail_n)

    sT = sH = 0
    for spec in specs
        dT, dH = _scratch_sizes(spec.tm, spec.tn, r, length(spec.obs))
        sT += dT
        sH += dH
    end

    buf_T  = similar(proto, T, max(sT, 1))
    buf_hi = similar(proto, Thi, max(sH, 1))
    fill!(buf_T,  zero(T))
    fill!(buf_hi, zero(Thi))

    pT = Ref(1)
    pH = Ref(1)
    cats = map(specs) do spec
        cat, pT[], pH[] = _carve_category_workspace(
            A_tlr, buf_T, pT[], buf_hi, pH[], spec, r, rank_type, proto, Thi)
        cat
    end

    CompressWorkspace(cats..., buf_T, buf_hi, create_streams(A_tlr.backend, 3))
end

# ─── Orthogonalisation helpers ────────────────────────────────────────────────

"""
    _cholqr_pass!(Y_hi, G_hi, regs, backend; rescue::Bool) -> Y_hi

One shifted Cholesky-QR pass, orthogonalising the columns of each batch entry
of `Y_hi` in place (`Y_hi ← Y_hi · R⁻¹` via `G_hi = Y_hiᴴY_hi = RᴴR`).

`rescue=true` uses the `√eps·tr(G)/r` shift that survives rank-deficient
sketches; `rescue=false` uses an `eps`-level shift so the pass restores the
column norms the rescue shift deflated — the truncation step's error
indicator needs orthonormal columns to be trustworthy. See
[`_cholqr_shift_compute_kernel!`](@ref) for the shift formulas.
"""
function _cholqr_pass!(Y_hi::AbstractArray{Thi,3}, G_hi, regs, backend; rescue::Bool) where {Thi}
    count = size(Y_hi, 3)
    count == 0 && return Y_hi
    r = size(Y_hi, 2)
    RT = typeof(real(zero(Thi)))
    gemm_batched!(_adjoint_blas_char(Thi), 'N', one(Thi), Y_hi, Y_hi, zero(Thi), G_hi)
    trace_coeff   = rescue ? RT(sqrt(eps(RT))) : zero(RT)
    maxdiag_coeff = rescue ? zero(RT) : RT(r) * eps(RT)
    _cholqr_shift_compute_kernel!(backend)(regs, G_hi, trace_coeff, maxdiag_coeff; ndrange=(count,))
    _cholqr_shift_apply_kernel!(backend)(G_hi, regs; ndrange=(r, count))
    potrf_batched!('U', G_hi)
    trsm_batched!('R', 'U', 'N', 'N', G_hi, Y_hi, one(Thi))
    Y_hi
end

"""
    _truncate!(U, V, rk, err_sq, normA_sq, U_tmp, V_tmp, eps_sq, rel, backend) -> U

Rank detection and truncation via [`_fused_truncate_kernel!`](@ref): sorts each
tile's columns by descending norm, greedily drops columns while staying within
the error-indicator-corrected budget, and copies the retained (zero-padded)
columns back into `U`/`V` using `U_tmp`/`V_tmp` as gather scratch. Writes the
retained rank to `rk` and the squared error estimate to `err_sq`.
"""
function _truncate!(U::AbstractArray{T,3}, V, rk, err_sq, normA_sq, U_tmp, V_tmp,
    eps_sq::Float64, rel::Bool, backend) where {T}
    noff = size(U, 3)
    noff == 0 && return U
    R = size(U, 2)
    kernel! = _fused_truncate_kernel!(backend, R)
    kernel!(U_tmp, V_tmp, U, V, rk, err_sq, normA_sq, eps_sq, rel, Val{R}();
        ndrange=(R * noff,), workgroupsize=R)
    U .= U_tmp
    V .= V_tmp
    U
end

# ─── Per-category compression pipeline ───────────────────────────────────────

"""
    _compress_category!(backend, A_tlr, A, cat, eps_sq, rel) -> cat

Compress every tile in one [`CompressCategoryWorkspace`](@ref) `cat`, writing
the retained factors into `cat.U`/`cat.V` and the per-tile rank/error estimate
into `cat.ranks_local`/`cat.err_sq_local`. No-op if `cat.obs` is empty.

Pipeline, `r_eff = cat.r_eff` columns wide (degenerates to rank 0 for every
tile when `r_eff == 0`, i.e. `maxrank == 0`):
1. Range sampling: `Y = A·Ω → U`, `Ω` drawn into `cat.V`.
2. Orthogonalise `U` with two shifted Cholesky-QR passes in higher precision.
3. Co-range: `V = Aᴴ·U`, overwriting `Ω`.
4. Rank detection + truncation against the error-indicator-corrected budget.

Called once per category from [`_compress_all_categories!`](@ref).
"""
function _compress_category!(
    backend,
    A_tlr::TLRMatrix,
    A::AbstractMatrix{T},
    cat::CompressCategoryWorkspace,
    eps_sq::Float64,
    rel::Bool,
) where {T}
    isempty(cat.obs) && return cat

    n = length(cat.obs)
    r_eff = cat.r_eff
    r_full = size(cat.U, 2)
    tm = size(cat.U, 1)
    tn = size(cat.V, 1)

    if r_eff == 0   # maxrank == 0: every tile degenerates to rank 0
        _tile_norm_sq_kernel!(backend, 1)(
            cat.normA_sq, A, cat.p0s, cat.q0s, tm, tn, Val{1}();
            ndrange=(n,), workgroupsize=1)
        fill!(cat.ranks_local, zero(eltype(cat.ranks_local)))
        cat.err_sq_local .= cat.normA_sq
        return cat
    end

    # Step 0: per-tile ‖A‖²_F — reference term for the error indicator (step 4).
    _tile_norm_sq_kernel!(backend, r_eff)(
        cat.normA_sq, A, cat.p0s, cat.q0s, tm, tn, Val{r_eff}();
        ndrange=(r_eff * n,), workgroupsize=r_eff)

    A_tiles = _tile_views(A, A_tlr, cat.obs)
    U_tiles = cat.U_tiles   # cached views into cat.U (first r_eff columns)
    V_tiles = cat.V_tiles
    U_eff = view(cat.U, :, 1:r_eff, :)
    V_eff = view(cat.V, :, 1:r_eff, :)

    # Step 1: range sampling  Y = A·Ω → U
    # V holds Ω (filled with randn!) for now; step 3 will overwrite it with A^T·U.
    Random.randn!(cat.V)
    gemm_batched!('N', 'N', one(T), A_tiles, V_tiles, zero(T), U_tiles)

    # Step 2: orthogonalise U (in higher precision)
    cat.Y_hi .= U_eff
    _cholqr_pass!(cat.Y_hi, cat.G_hi, cat.regs, backend; rescue=true)
    _cholqr_pass!(cat.Y_hi, cat.G_hi, cat.regs, backend; rescue=false)
    U_eff .= cat.Y_hi

    # Step 3: co-range  V = Aᴴ·U  (overwrites the Ω we no longer need)
    gemm_batched!(_adjoint_blas_char(T), 'N', one(T), A_tiles, U_tiles, zero(T), V_tiles)

    # Step 4: rank detection + truncation (fused SMEM kernel, EI-corrected budget)
    _truncate!(U_eff, V_eff, cat.ranks_local, cat.err_sq_local, cat.normA_sq,
        view(cat.U_tmp, :, 1:r_eff, :), view(cat.V_tmp, :, 1:r_eff, :),
        eps_sq, rel, backend)

    # randn! polluted the padding columns of V with Ω samples; downstream GEMM
    # relies on zeros beyond the effective rank.
    if r_eff < r_full
        view(cat.V, :, (r_eff+1):r_full, :) .= zero(T)
    end
    cat
end

# ─── Storage helpers ──────────────────────────────────────────────────────────

"""
    _store_category_results!(A_tlr, cat)

Copy one category's local ranks and squared-error estimates (device or host
vectors on `cat`) back into the global `A_tlr.ranks`/`A_tlr.resid`, converting
squared error to the reported Frobenius residual. No-op if `cat.obs` is empty.
"""
function _store_category_results!(A_tlr::TLRMatrix, cat::CompressCategoryWorkspace)
    isempty(cat.obs) && return
    rk_host  = cat.ranks_local isa Vector ? cat.ranks_local : Array(cat.ranks_local)
    err_host = cat.err_sq_local isa Vector ? cat.err_sq_local : Array(cat.err_sq_local)
    @inbounds for (k, ob) in enumerate(cat.obs)
        A_tlr.ranks[ob] = rk_host[k]
        A_tlr.resid[ob] = sqrt(max(err_host[k], 0.0))
    end
end

# ─── Orchestration ────────────────────────────────────────────────────────────

"""
    _compress_all_categories!(A_tlr, A, ws, eps_sq, rel) -> A_tlr

Run [`_compress_category!`](@ref) for all three tile categories (interior,
right-boundary, bottom-boundary) and store their results into `A_tlr`. On the
CPU backend the categories run sequentially; on GPU backends each runs on its
own stream (from `ws.streams`) and all streams are synchronised before results
are stored, so the three categories overlap on-device.
"""
function _compress_all_categories!(
    A_tlr::TLRMatrix{<:Any,T},
    A::AbstractMatrix{T},
    ws::CompressWorkspace,
    eps_sq::Float64,
    rel::Bool,
) where {T}
    cats = (ws.interior, ws.right, ws.bottom)
    if A_tlr.backend isa KernelAbstractions.CPU
        for cat in cats
            _compress_category!(A_tlr.backend, A_tlr, A, cat, eps_sq, rel)
        end
    else
        for (cat, stream) in zip(cats, ws.streams)
            with_stream(A_tlr.backend, stream) do
                _compress_category!(A_tlr.backend, A_tlr, A, cat, eps_sq, rel)
            end
        end
        for stream in ws.streams
            sync_stream(A_tlr.backend, stream)
        end
    end
    for cat in cats
        _store_category_results!(A_tlr, cat)
    end
    A_tlr
end

"""
    compress!(A_tlr, A [, ws]; tol=0.0, rel=false)

Compress dense matrix `A` into the TLR container `A_tlr` in-place.

Per-tile effective ranks are detected via greedy V-column-norm thresholding
against an error-indicator-corrected budget and stored in `ranks(A_tlr)`; the
estimated per-tile Frobenius error lands in `residuals(A_tlr)`.  The indicator
(`‖A_tile‖²_F − ‖V‖²_F`, à la randQB_EI) accounts for the range-capture error
of the sketch, so a tile whose spectrum does not fit within `maxrank` keeps
full rank and reports a residual above `tol` instead of silently claiming
convergence — check `residuals` to route such tiles to dense storage or a
higher-rank second pass.

Factor arrays are updated in-place; call `alloc_workspace` once to amortise
device allocations across repeated calls:

    ws = alloc_workspace(A_tlr)
    for A in matrices
        compress!(A_tlr, A, ws; tol=1f-3)
    end

## Keywords

`tol` — per-tile Frobenius error budget (default `0.0`). The budget is floored
at `√eps(T)·‖A_tile‖_F`: a precision-`T` sketch cannot resolve relative error
below `√eps(T)` (≈ `3.4e-4` for `Float32`), so a `tol` finer than that recovers
the true rank at the `√eps(T)` accuracy floor rather than retaining spurious
noise columns to chase an unreachable target.

`rel` — when `true`, the budget for each tile is `tol * ‖A_tile‖_F` instead of
the absolute `tol`.

The sketch basis is orthogonalised with two shifted Cholesky-QR passes in
higher precision.
"""
compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T}; kwargs...) where {T} =
    compress!(A_tlr, A, alloc_workspace(A_tlr); kwargs...)

function compress!(A_tlr::TLRMatrix{<:Any,T}, A::AbstractMatrix{T},
    ws::CompressWorkspace;
    tol::Real=0.0, rel::Bool=false) where {T}

    size(A, 1) == A_tlr.m && size(A, 2) == A_tlr.n ||
        throw(DimensionMismatch("A dimensions must match A_tlr"))
    A_tlr.m == A_tlr.n && A_tlr.tile_m == A_tlr.tile_n ||
        throw(ArgumentError("compress! currently requires square matrices with square tiles"))
    tol >= 0 || throw(ArgumentError("tol must be >= 0"))

    _copy_diagonal_from_dense!(A_tlr, A)

    eps_sq = Float64(tol)^2
    _compress_all_categories!(A_tlr, A, ws, eps_sq, rel)

    A_tlr
end
