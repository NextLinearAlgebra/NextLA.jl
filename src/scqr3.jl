export scqr3!, scqr3_gram!, scqr3_fukaya_shift

# Static `Val{TILE}` dispatch for `scqr3_gram_kernel!` (`@localmem (TILE, TILE)` × 2).
# Discrete τ-like panel tiles (manuscript §3.4 streaming / fast-memory): small factors and
# powers-of-two common in blocked SYRK/GEMM, up to 256 for large CPU scratch budgets.
const _SCQR3_GRAM_TILE_CANDIDATES =
	(4, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128, 160, 192, 256)

"""
    _scqr3_gram_backend_caps(backend, ::Type{T}) -> (max_threads, max_shmem_bytes)

Upper bounds for one workgroup of size `(TILE, TILE)` and static `@localmem` scratch
`2·TILE²·sizeof(T)`. Specialized in `ext/cudaext.jl` (CUDA) and `ext/amdext.jl` (ROCm); CPU uses
generous defaults from `probe_device`-style reasoning.
"""
function _scqr3_gram_backend_caps(backend, ::Type{T}) where {T}
	if backend isa KernelAbstractions.CPU
		return (262_144, 1 << 30)
	end
	# Unknown GPU backend: conservative so launches stay valid (e.g. 1024 threads/block).
	return (1024, 96 * 1024)
end

function _scqr3_gram_feasible_tiles(backend, ::Type{T}) where {T}
	max_th, max_shmem = _scqr3_gram_backend_caps(backend, T)
	out = Int[]
	for t in _SCQR3_GRAM_TILE_CANDIDATES
		t * t <= max_th || continue
		2 * t * t * sizeof(T) <= max_shmem || continue
		push!(out, t)
	end
	return out
end

"""
    _scqr3_pick_gram_tile(backend, T, raw) -> Int

Choose the Gram launch tile: **smallest** value in `_scqr3_gram_feasible_tiles` that is `≥ raw`,
or the **largest** feasible tile if `raw` exceeds all feasible sizes. There is no clamp-to-`[4,256]`
and no nearest-neighbor snap among candidates.

`raw` is typically `DeviceParams.TILE_DIM` (= `⌊√M⌋` from `compute_params`), which may not itself be a
compiled tile size; this function only maps it to a **valid** launch size.
"""
function _scqr3_pick_gram_tile(backend, ::Type{T}, raw::Integer) where {T}
	feas = _scqr3_gram_feasible_tiles(backend, T)
	isempty(feas) &&
		throw(ArgumentError("no Gram tile in $_SCQR3_GRAM_TILE_CANDIDATES fits backend limits for $(typeof(backend)) and T=$T"))
	r = Int(raw)
	if r <= first(feas)
		return first(feas)
	end
	for t in feas
		if t >= r
			return t
		end
	end
	return last(feas)
end

"""
    scqr3_fukaya_shift(::Type{T}, m, b, tr) -> real(T)

Diagonal shift `s` for the first shifted Cholesky QR step, following Fukaya et al. (2018), eq. (67)
(arXiv:1809.11085 / SIAM SISC), with `n = b` panel columns:

``s = 11(m n + n(n+1))\\,\\mathbf{u}\\,\\|X\\|_2^2``.

Here `\\mathbf{u}` is `eps(real(T))`, matching the real floating-point model in the reference.
Lemma 3.1 of the same paper assumes a shift compatible with their eq. (7); the closed form (67)
used in shiftedCholeskyQR3 is the one implemented here.

`tr` must be ``\\mathrm{tr}(A^\\mathrm{H}A) = \\|A\\|_F^2`` (e.g. after `scqr3_gram!` and any
`panel_allreduce!`). Using ``\\|A\\|_F^2`` in place of ``\\|A\\|_2^2`` is the conservative substitute
noted after (67) in the paper.
"""
function scqr3_fukaya_shift(::Type{T}, m::Integer, b::Integer, tr) where {T}
	RT = real(T)
	m = Int(m)
	b = Int(b)
	n = b
	u = eps(RT)
	# Coefficient 11·(m n + n(n+1)) from Fukaya et al. (2018), eq. (67).
	coef = RT(11) * (RT(m * n) + RT(n * (n + 1))) * u
	s = coef * RT(tr)
	return max(zero(RT), s)
end

"""
    scqr3_gram!(G, A_panel, m, b; params=nothing, gram_tile=nothing)

Fill the leading `b×b` block of `G` with the Gram matrix ``A^\\mathrm{H} A`` of `A_panel[1:m, 1:b]`
(Hermitian inner product: ``G_{ij} = \\sum_k \\overline{A_{ki}} A_{kj}``). Overwrites `G`;
entries outside the `b×b` leading block are left untouched.

Uses `scqr3_gram_kernel!` with tiled `@localmem` loads (SYRK-style over row tiles of `A`).

# Tile size

Compiled candidates are `_SCQR3_GRAM_TILE_CANDIDATES` (§3.4-style streaming tile ``τ`` plus common
blocked factors). The launch tile is the **smallest** candidate that is both feasible under
`_scqr3_gram_backend_caps` (CPU defaults or values from `ext/cudaext.jl` / `ext/amdext.jl`) and
`≥ raw`, where `raw` comes from `gram_tile` if set, else `params.TILE_DIM` when `params !== nothing`,
else `16`. If no candidate is `≥ raw`, the **largest** feasible tile is used. There is no clamp of
`raw` to the candidate list and no nearest-neighbor rounding.
"""
function scqr3_gram!(G::AbstractMatrix{T}, A_panel::AbstractMatrix{T}, m::Integer, b::Integer;
					 params::Union{Nothing, DeviceParams} = nothing,
					 gram_tile::Union{Nothing, Integer} = nothing,
					 ) where {T}
	m = Int(m)
	b = Int(b)
	m >= 1 || throw(ArgumentError("m must be ≥ 1"))
	b >= 1 || throw(ArgumentError("b must be ≥ 1"))
	size(A_panel, 1) >= m && size(A_panel, 2) >= b ||
		throw(ArgumentError("A_panel too small for $(m)×$(b) Gram"))
	size(G, 1) >= b && size(G, 2) >= b ||
		throw(ArgumentError("G must be at least $(b)×$(b)"))
	be = KernelAbstractions.get_backend(G)
	KernelAbstractions.get_backend(A_panel) === be ||
		throw(ArgumentError("`G` and `A_panel` must share the same KernelAbstractions backend"))
	# Fast path: BLAS/cuBLAS GEMM (CPU: OpenBLAS/MKL; GPU: cuBLAS via mul! dispatch).
	# Full G = A^H A so the upper Cholesky sees the correct upper-triangle entries.
	if T <: LinearAlgebra.BlasFloat
		Gv = view(G, 1:b, 1:b)
		Av = view(A_panel, 1:m, 1:b)
		mul!(Gv, Av', Av)
		return G
	end
	raw = if gram_tile !== nothing
		Int(gram_tile)
	elseif params !== nothing
		params.TILE_DIM
	else
		16
	end
	TILE = _scqr3_pick_gram_tile(be, T, raw)
	fill!(view(G, 1:b, 1:b), zero(T))
	_scqr3_gram_launch!(be, G, A_panel, m, b, TILE)
	KernelAbstractions.synchronize(be)
	return G
end

# Tiled accumulation: for each output tile of G, sum contributions along row tiles of A (dimension m).
@kernel unsafe_indices = true function scqr3_gram_kernel!(G_local, A_panel, m::Int, b::Int, ::Val{TILE}) where {TILE}
	gi, gj = @index(Group, NTuple)
	ti, tj = @index(Local, NTuple)
	Sa = @localmem eltype(G_local) (TILE + 1, TILE)
	Sb = @localmem eltype(G_local) (TILE + 1, TILE)
	acc = @private eltype(G_local) 1
	@inbounds acc[1] = zero(eltype(G_local))
	# Row-tile count along m; @uniform so it is not trapped in a CPU workitem-only segment
	# (KernelAbstractions splits around @synchronize).
	@uniform n_k_tiles = cld(m, TILE)
	@uniform ElT = eltype(G_local)
	for kt in 1:n_k_tiles
		i0 = (gi - 1) * TILE
		j0 = (gj - 1) * TILE
		k0 = (kt - 1) * TILE
		@inbounds Sa[ti, tj] =
			(k0 + ti <= m && i0 + tj <= b) ? A_panel[k0 + ti, i0 + tj] : zero(eltype(G_local))
		@inbounds Sb[ti, tj] =
			(k0 + ti <= m && j0 + tj <= b) ? A_panel[k0 + ti, j0 + tj] : zero(eltype(G_local))
		@synchronize
		# Do not use `io`/`jo` aliases here: CPU lowering splits on @synchronize and those
		# bindings would not carry across segments.
		if (gi - 1) * TILE + ti <= b && (gj - 1) * TILE + tj <= b
			s = zero(ElT)
			for kk in 1:TILE
				# Avoid conj() — oneAPI SPIR-V rejects the im global it emits for complex types.
				@inbounds begin
					a = Sa[kk, ti]
					s += (ElT <: Complex ? Complex(real(a), -imag(a)) : a) * Sb[kk, tj]
				end
			end
			acc[1] += s
		end
		@synchronize
	end
	if (gi - 1) * TILE + ti <= b && (gj - 1) * TILE + tj <= b
		@inbounds G_local[(gi - 1) * TILE + ti, (gj - 1) * TILE + tj] = acc[1]
	end
end

let
	msg_suf = string("; allowed ", _SCQR3_GRAM_TILE_CANDIDATES)
	acc = :(throw(ArgumentError(string("invalid Gram TILE=", TILE, $(msg_suf)))))
	for tile in reverse(collect(_SCQR3_GRAM_TILE_CANDIDATES))
		acc = Expr(
			:if,
			:(TILE === $(tile)),
			:(begin
				scqr3_gram_kernel!(backend, ($(tile), $(tile)))(G, A_panel, m, b, Val($(tile)); ndrange = nd)
				return nothing
			end),
			acc,
		)
	end
	Core.eval(
		@__MODULE__,
		quote
			function _scqr3_gram_launch!(backend, G::AbstractMatrix{T}, A_panel::AbstractMatrix{T}, m::Int, b::Int, TILE::Int) where {T}
				nxi = cld(b, TILE)
				nxj = cld(b, TILE)
				nd = (TILE * nxi, TILE * nxj)
				$(acc)
			end
		end,
	)
end

# Pack real diagonal of leading b×b block into length-N vector (N power of two; tail zero).
@kernel unsafe_indices = true function scqr3_trace_pack_kernel!(src, G, b::Int)
	wi = @index(Global, Linear)
	z = zero(eltype(src))
	@inbounds src[wi] = wi <= b ? real(G[wi, wi]) : z
end

# Add real shift `s` to each diagonal G[j,j], j = 1:b (Gram diagonal is real).
@kernel unsafe_indices = true function scqr3_diag_shift_kernel!(G, b::Int, s)
	j = @index(Global, Linear)
	T = eltype(G)
	if j <= b
		@inbounds G[j, j] = G[j, j] + convert(T, s)
	end
end

# Fused on-device shift: reads tr = trace_out[1] from device memory (no D2H sync) and
# applies the Fukaya shift s = coef * tr to each G[j,j] in 1:b. Saves one D2H roundtrip
# and one CPU-side scalar computation per first sCQR3 iteration.
@kernel unsafe_indices = true function scqr3_shift_diag_from_trace_kernel!(G, b::Int, trace_out, coef)
	j = @index(Global, Linear)
	T = eltype(G)
	if j <= b
		@inbounds tr = trace_out[1]
		s = coef * tr
		@inbounds G[j, j] = G[j, j] + convert(T, s)
	end
end

@kernel unsafe_indices = true function scqr3_fill_diag_kernel!(A, val, b::Int)
	j = @index(Global, Linear)
	if j <= b
		@inbounds A[j, j] = val
	end
end

# Set A[j,j] = val for j in 1:b on the same backend as A (no allocation, async).
function _scqr3_fill_diag!(be, A::AbstractMatrix, val, b::Int)
	scqr3_fill_diag_kernel!(be)(A, val, b; ndrange = b)
end

"""
    scqr3!(m, b, A_panel, R, G, info; params, partials=nothing)

Three-pass shifted Cholesky QR on one panel: each pass forms the Gram matrix ``A^\\mathrm{H} A``
(`scqr3_gram!` into `G`), optional `panel_allreduce!` when `params.c > 1` (requires `partials`),
adds a diagonal shift `s` on the **first** pass only (Fukaya et al. (2018) eq. (67) via `scqr3_fukaya_shift`,
using ``\\mathrm{tr}(G)=\\|A\\|_F^2`` after the Gram and optional `panel_allreduce!`; device-side trace
and `workgroup_reduce!` when `nextpow(2, b) ≤ 1024`, otherwise a host `b×b` fallback), then overwrites the leading `b×b` block of `G`
with its upper Cholesky factor, accumulates `R := R_k · R` in the leading `b×b` block of `R`,
and applies `RightUpperTRSM!` so `A_panel ← A_panel · R_k^{-1}`.

Throws `PosDefException` if Cholesky fails (`info[1]` from `scqr3_cholesky!`).

# Arguments
- `partials`: when `params.c > 1`, an `b×b×(Px·Pz)` array (same backend as `A_panel`) used by `panel_allreduce!`.
  For a single replica, this implementation fans out `G/K` into each slab so the sum matches `G`.
"""
function scqr3!(m::Integer, b::Integer, A_panel::AbstractMatrix{T},
               R::AbstractMatrix{T}, G::AbstractMatrix{T}, info::AbstractVector{Int};
               params::DeviceParams{T},
               partials::Union{Nothing, AbstractArray{T, 3}} = nothing,
               ) where {T}
	m = Int(m)
	b = Int(b)
	m >= 1 || throw(ArgumentError("m must be ≥ 1, got m=$m"))
	b >= 1 || throw(ArgumentError("b must be ≥ 1, got b=$b"))
	params.b == b || throw(ArgumentError("params.b ($(params.b)) must match argument b ($b)"))
	size(A_panel, 1) >= m && size(A_panel, 2) >= b ||
		throw(ArgumentError("A_panel must be at least $(m)×$(b), got $(size(A_panel))"))
	size(G, 1) >= b && size(G, 2) >= b ||
		throw(ArgumentError("G must be at least $(b)×$(b), got $(size(G))"))
	size(R, 1) >= b && size(R, 2) >= b ||
		throw(ArgumentError("R must be at least $(b)×$(b), got $(size(R))"))
	length(info) >= 1 || throw(ArgumentError("info must have length ≥ 1"))
	be = KernelAbstractions.get_backend(A_panel)
	KernelAbstractions.get_backend(R) === be &&
		KernelAbstractions.get_backend(G) === be &&
		KernelAbstractions.get_backend(info) === be ||
		throw(ArgumentError("A_panel, R, G, and info must share the same KernelAbstractions backend"))
	if params.c > 1
		partials === nothing &&
			throw(ArgumentError("params.c=$(params.c) > 1 requires keyword `partials` with size (b,b,Px·Pz) for panel_allreduce!"))
		K = params.Px * params.Pz
		size(partials, 1) == b && size(partials, 2) == b && size(partials, 3) >= K ||
			throw(ArgumentError("partials must be $(b)×$(b)×(≥$K), got $(size(partials))"))
		KernelAbstractions.get_backend(partials) === be ||
			throw(ArgumentError("`partials` must use the same backend as A_panel"))
	end

	# Racc accumulates R = G_3 G_2 G_1; Rwrk is the swap partner for the TRMM product.
	# Identity init is required so the iter-1 TRMM sees zero strict-lower (potrf leaves
	# the lower triangle holding the original Gram entries) and keeps Racc upper-triangular
	# across iterations. The swap below avoids the per-iter copyto!(Racc, Rwrk).
	Racc = similar(G, b, b)
	fill!(Racc, zero(T))
	_scqr3_fill_diag!(be, Racc, one(T), b)
	Rwrk = similar(G, b, b)
	RT = real(T)
	Ntr = nextpow(2, b)
	use_device_trace = Ntr <= _WORKGROUP_REDUCE_MAX
	trace_src = use_device_trace ? similar(G, RT, (Ntr,)) : nothing
	trace_out = use_device_trace ? similar(G, RT, (1,)) : nothing

	for it in 1:3
		scqr3_gram!(G, A_panel, m, b; params = params)
		if params.c > 1
			invK = one(T) / (params.Px * params.Pz)
			@inbounds for k in 1:(params.Px * params.Pz)
				@views partials[:, :, k] .= G .* invK
			end
			panel_allreduce!(G, partials, params)
		end
		if it == 1
			if use_device_trace
				scqr3_trace_pack_kernel!(be)(trace_src, G, b; ndrange = Ntr)
				workgroup_reduce!(trace_out, trace_src; op = +, N = Ntr)
				# On-device shift: avoid D2H roundtrip. The shift kernel reads trace_out[1]
				# directly and applies s = coef * tr. coef matches `scqr3_fukaya_shift`:
				# coef = 11*(m*b + b*(b+1))*u (max with 0 enforced inside the kernel via
				# `coef ≥ 0` since u, m, b are all positive).
				coef = RT(11) * (RT(m * b) + RT(b * (b + 1))) * eps(RT)
				scqr3_shift_diag_from_trace_kernel!(be)(G, b, trace_out, coef; ndrange = b)
			else
				# Fallback when nextpow(2, b) exceeds workgroup_reduce! scratch cap.
				Gb = Matrix{T}(undef, b, b)
				copyto!(Gb, view(G, 1:b, 1:b))
				tr = sum(j -> (@inbounds real(Gb[j, j])), 1:b)
				s = scqr3_fukaya_shift(T, m, b, tr)
				@inbounds for j in 1:b
					Gb[j, j] += s
				end
				copyto!(view(G, 1:b, 1:b), Gb)
			end
		end
		_scqr3_potrf!(be, G, b)
		Gv = view(G, 1:b, 1:b)
		# R_acc := R_iter * R_acc; swap so Racc holds the latest product without copyto!.
		mul!(Rwrk, UpperTriangular(Gv), Racc)
		Racc, Rwrk = Rwrk, Racc
		# A_panel[:, 1:b] = A_panel[:, 1:b] * R_iter^{-1}  (right upper TRSM).
		rdiv!(view(A_panel, 1:m, 1:b), UpperTriangular(Gv))
	end
	copyto!(view(R, 1:b, 1:b), Racc)
	return nothing
end

"""
    scqr3!(A_panel, R; params=nothing)

Convenience wrapper: builds `compute_params(...; b = ncols, c = 1)` when `params` is omitted,
allocates `G`, `info`, and optional `partials`, then calls the low-level `scqr3!`.
"""
function scqr3!(A_panel::AbstractMatrix{T}, R::AbstractMatrix{T};
               params::Union{DeviceParams{T}, Nothing} = nothing) where {T}
	be = KernelAbstractions.get_backend(A_panel)
	KernelAbstractions.get_backend(R) === be ||
		throw(ArgumentError("`A_panel` and `R` must use the same KernelAbstractions backend"))
	m, n = size(A_panel)
	n >= 1 || throw(ArgumentError("A_panel must have at least one column"))
	N = max(m, n)
	# Force c = 1 so b_min = 1 and requested b = n is not clamped upward on large-memory hosts.
	p = params === nothing ? compute_params(be, T, N; b = n, c = 1) : params
	b = p.b
	b >= 1 || throw(ArgumentError("params.b must be ≥ 1 for scqr3!, got b=$b"))
	size(A_panel, 2) >= b ||
		throw(ArgumentError("A_panel must have at least params.b ($(b)) columns, got $(size(A_panel, 2))"))
	G = similar(A_panel, (b, b))
	info = fill!(similar(A_panel, Int, (1,)), 0)
	partials = p.c > 1 ? similar(A_panel, (b, b, p.Px * p.Pz)) : nothing
	scqr3!(m, b, A_panel, R, G, info; params = p, partials = partials)
	return nothing
end

# ── Vendor Cholesky dispatch ──────────────────────────────────────────────────
# `_scqr3_potrf!(be, G, b)` computes the in-place upper Cholesky of G[1:b, 1:b].
# On success the upper triangle holds R with R^H R = G_input[1:b, 1:b].
# Throws PosDefException on failure.
#
# CPU: LAPACK POTRF (direct in-place, no alloc).
# GPU (CUDA): overridden in ext/cudaext.jl via CUSOLVER POTRF.
# Generic fallback: serial KA kernel (for non-BlasFloat or unrecognised backends).

function _scqr3_potrf!(::KernelAbstractions.CPU, G::AbstractMatrix, b::Int)
	_, info = LAPACK.potrf!('U', view(G, 1:b, 1:b))
	info != 0 && throw(PosDefException(info))
end

function _scqr3_potrf!(be, G::AbstractMatrix, b::Int)
	info_scratch = fill!(similar(G, Int, 1), 0)
	scqr3_cholesky!(G, info_scratch, b)
	inum = Array(info_scratch)[1]
	inum != 0 && throw(PosDefException(inum))
end

"""
    scqr3_cholesky!(G, info, b)

In-place **upper** Cholesky factorization of the leading `b×b` block of `G`, treating that
block as Hermitian with the **upper** triangle storing the matrix entries (strict lower
triangle is ignored). On success, the upper triangle holds `R` with `R'R` equal to the
input block (same convention as `LinearAlgebra.cholesky(Hermitian(G[1:b,1:b], :U))`).

`info[1]` is set to `0` on success. On failure (non-positive pivot), `info[1]` is the
1-based index of the diagonal that was non-positive (either a divisor `R[i,i]` while forming
column `j>i`, or the Schur complement pivot at step `j`).
The caller may apply a shift on the host (e.g. add `s ≥ 0` to the diagonal of the Gram block)
before calling when the block is only semidefinite.

Runs as a single serial work-item (`ndrange = 1`); `G` and `info` must share the same
KernelAbstractions backend.
"""
function scqr3_cholesky!(G::AbstractMatrix, info::AbstractVector{Int}, b::Integer)
    b = Int(b)
    b >= 0 || throw(ArgumentError("b must be non-negative, got b=$b"))
    size(G, 1) >= b && size(G, 2) >= b ||
        throw(ArgumentError("G must be at least $(b)×$(b), got $(size(G))"))
    length(info) >= 1 || throw(ArgumentError("info must have length ≥ 1"))
    backend = KernelAbstractions.get_backend(G)
    KernelAbstractions.get_backend(info) === backend ||
        throw(ArgumentError("`G` and `info` must use the same KernelAbstractions backend"))
    scqr3_cholesky_kernel!(backend, ())(G, info, b; ndrange = 1)
    KernelAbstractions.synchronize(backend)
    return nothing
end

# Upper Cholesky (Potrf U): one thread performs the full `b×b` factorization (Phase S2 / Q1).
@kernel unsafe_indices = true function scqr3_cholesky_kernel!(G, info, b::Int)
    wi = @index(Global, Linear)
    if wi == 1
        if b <= 0
            @inbounds info[1] = 0
        else
            T = eltype(G)
            zr = zero(real(T))
            fail = 0
            @inbounds for j in 1:b
                fail != 0 && break
                for i in 1:(j - 1)
                    s = G[i, j]
                    for k in 1:(i - 1)
                        gki = G[k, i]
                        s -= (T <: Complex ? Complex(real(gki), -imag(gki)) : gki) * G[k, j]
                    end
                    d = G[i, i]
                    if real(d) <= zr
                        fail = i
                        break
                    end
                    G[i, j] = s / d
                end
                fail != 0 && break
                s = G[j, j]
                for k in 1:(j - 1)
                    gkj = G[k, j]
                    s -= (T <: Complex ? Complex(real(gkj), -imag(gkj)) : gkj) * gkj
                end
                rs = real(s)
                if rs <= zr
                    fail = j
                    break
                end
                G[j, j] = convert(T, sqrt(rs))
            end
            @inbounds info[1] = fail
        end
    end
end
