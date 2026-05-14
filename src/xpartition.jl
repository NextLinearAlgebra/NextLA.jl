export DeviceParams, probe_device, compute_params, panel_cu_set, block_owner, workgroup_reduce!, panel_allreduce!, verify_budget

struct DeviceParams{T}
	P::Int
	M::Int
	b::Int
	c::Int
	P1::Int
	Px::Int
	Py::Int
	Pz::Int
	TILE_DIM::Int
	b_min::Int
	b_max::Int
	AI_target::T
end

function DeviceParams(P::Integer, M::Integer, b::Integer, c::Integer,
					  P1::Integer, Px::Integer, Py::Integer, Pz::Integer,
					  TILE_DIM::Integer, b_min::Integer, b_max::Integer,
					  AI_target::T) where {T}
	return DeviceParams{T}(Int(P), Int(M), Int(b), Int(c), Int(P1), Int(Px),
						   Int(Py), Int(Pz), Int(TILE_DIM), Int(b_min),
						   Int(b_max), AI_target)
end

function probe_device(backend, ::Type{T}) where {T}
	# CPU: treat P as thread count and M as per-thread share of total memory.
	P = max(1, Threads.nthreads())
	bytes = max(1, Int(Sys.total_memory()))
	M = max(1, (bytes ÷ max(1, sizeof(T))) ÷ P)
	return P, M
end

function compute_params(backend, ::Type{T}, N; b=nothing, c=nothing) where {T}
	P, M = probe_device(backend, T)
	N_int = Int(N)
	N_int <= 0 && return DeviceParams(P, M, 0, 1, P, max(1, P), 1, 1, 0, 0, 0, zero(T))

	if c === nothing
		c_calc = (P * M) ÷ (N_int * N_int)
		c_val = max(1, c_calc)
		if c_calc < 1
			@warn "2D fallback (c < 1): PM=$(P*M) < N^2=$(N_int^2)"
		end
	else
		c_val = max(1, Int(c))
	end

	P1 = max(1, P ÷ c_val)
	Px = max(1, isqrt(P1))
	Py = Px
	Pz = c_val
	# Abstract fast-memory / streaming tile; `scqr3_gram!` maps `TILE_DIM` to the nearest entry in
	# `_SCQR3_GRAM_TILE_CANDIDATES` that fits `_scqr3_gram_backend_caps` (CPU defaults or CUDA/AMD ext).
	TILE_DIM = max(1, Int(floor(sqrt(float(M)))))
	b_min = c_val
	b_max = max(1, N_int ÷ Px)

	if b === nothing
		b_val = max(b_min, min(b_max, TILE_DIM))
	else
		b_val = Int(b)
		if b_val < b_min
			@warn "b below b_min; clamping" b=b_val b_min=b_min
			b_val = b_min
		elseif b_val > b_max
			@warn "b above b_max; clamping" b=b_val b_max=b_max
			b_val = b_max
		end
	end

	ai_target = T((2 / 3) * sqrt(float(M)))
	return DeviceParams(P, M, b_val, c_val, P1, Px, Py, Pz, TILE_DIM, b_min, b_max, ai_target)
end

function panel_cu_set(k, params::DeviceParams)
	j_k = mod(Int(k), params.Py)
	total = params.Px * params.Pz
	coords = Vector{NTuple{3, Int}}(undef, total)
	idx = 1
	for r in 0:(params.Px - 1)
		for z in 0:(params.Pz - 1)
			coords[idx] = (r, j_k, z)
			idx += 1
		end
	end
	return coords
end

"""
    block_owner(IJ, params) -> Vector{NTuple{3,Int}}

Global block `(I, J)` maps to `(mod(I,Px), mod(J,Py), z)` for each `z` in `0:Pz-1` (`Pz == c`).
"""
function block_owner(IJ, params::DeviceParams)
	I = Int(IJ[1])
	J = Int(IJ[2])
	r = mod(I, params.Px)
	j = mod(J, params.Py)
	return NTuple{3, Int}[(r, j, z) for z in 0:(params.Pz - 1)]
end

# Upper bound on workgroup size for `workgroup_reduce!` / `workgroup_reduce_kernel!`
# (shared scratch is statically sized for the largest supported `N`).
const _WORKGROUP_REDUCE_MAX = 1024

"""
    workgroup_reduce!(out, src; op=+, N=256)

Tree-reduce the first `length(src)` elements of `src` across one workgroup of size `N`,
writing the result to `out[1]` via `workgroup_reduce_kernel!`.

Requires `N` a power of two, `length(src) ≤ N`, `length(out) ≥ 1`, matching `eltype`s,
and `op === (+)` (other operators are not supported in the kernel yet).
"""
function workgroup_reduce!(out, src; op=+, N::Int=256)
	op === (+) || throw(ArgumentError("workgroup_reduce! only supports `op=+`"))
	N >= 1 || throw(ArgumentError("N must be positive, got N=$N"))
	Base.ispow2(N) || throw(ArgumentError("N must be a power of two, got N=$N"))
	N > _WORKGROUP_REDUCE_MAX &&
		throw(ArgumentError("N (got $N) exceeds compile-time scratch limit $(_WORKGROUP_REDUCE_MAX)"))
	len = length(src)
	len <= N || throw(ArgumentError("length(src) ($len) must be ≤ N ($N)"))
	length(out) >= 1 || throw(ArgumentError("length(out) must be ≥ 1"))
	eltype(out) === eltype(src) ||
		throw(ArgumentError("eltype(out) ($(eltype(out))) must match eltype(src) ($(eltype(src)))"))
	KernelAbstractions.get_backend(out) === KernelAbstractions.get_backend(src) ||
		throw(ArgumentError("`out` and `src` must use the same KernelAbstractions backend"))
	nlevels = N == 1 ? 0 : Int(log2(N))
	backend = KernelAbstractions.get_backend(src)
	workgroup_reduce_kernel!(backend, (N,))(out, src, len, N, nlevels; ndrange=N)
	KernelAbstractions.synchronize(backend)
	return out
end

"""
    panel_allreduce!(G, partials, params)

In-place sum of panel Gram contributions: for each entry `(i,j)`, sets
`G[i,j] = sum(partials[i,j,k] for k in 1:K)` with `K = params.Px * params.Pz`.

`partials` must be an `AbstractArray` with `size(partials, 1) == size(partials, 2) == params.b`
and `size(partials, 3) >= K`. Padding beyond `K` is ignored (treated as absent contributions).

Uses a power-of-two tree size `nextpow(2, K)` for validation (must fit the same scratch cap as
`workgroup_reduce!`). The kernel uses **one workgroup** of up to `min(b^2, $(_WORKGROUP_REDUCE_MAX))`
local threads: each thread sums all `K` slabs for a disjoint subset of linear indices in `G`.
The tree bound is reserved for a future tree reduction over `k` within each thread.

**No-op when `params.c == 1`:** returns immediately without reading `partials` or writing `G`
(caller is responsible for having the correct `G` when there is no replication).

Requires `params.Px * params.Pz <= $(_WORKGROUP_REDUCE_MAX)` so each per-entry reduction fits
one workgroup.
"""
function panel_allreduce!(G, partials, params::DeviceParams)
	params.c == 1 && return G
	b = params.b
	b <= 0 && return G
	K = params.Px * params.Pz
	tree = nextpow(2, K)
	tree > _WORKGROUP_REDUCE_MAX &&
		throw(ArgumentError("panel tree size $tree (from K=$K) exceeds limit $(_WORKGROUP_REDUCE_MAX); reduce Px·Pz or raise scratch cap"))
	size(G, 1) == b && size(G, 2) == b ||
		throw(ArgumentError("G must be $(b)×$(b), got $(size(G,1))×$(size(G,2))"))
	size(partials, 1) == b && size(partials, 2) == b ||
		throw(ArgumentError("partials first two dims must be $(b)×$(b), got $(size(partials,1))×$(size(partials,2))"))
	size(partials, 3) >= K ||
		throw(ArgumentError("size(partials,3) ($(size(partials,3)))) must be ≥ K ($K)"))
	eltype(G) === eltype(partials) ||
		throw(ArgumentError("eltype(G) ($(eltype(G))) must match eltype(partials) ($(eltype(partials)))"))
	KernelAbstractions.get_backend(G) === KernelAbstractions.get_backend(partials) ||
		throw(ArgumentError("`G` and `partials` must use the same KernelAbstractions backend"))
	backend = KernelAbstractions.get_backend(G)
	fill!(G, zero(eltype(G)))
	n_idx = b * b
	wg = min(n_idx, _WORKGROUP_REDUCE_MAX)
	panel_allreduce_kernel!(backend, (wg,))(G, partials, b, K; ndrange=wg)
	KernelAbstractions.synchronize(backend)
	return G
end

"""
    verify_budget(params::DeviceParams; N=nothing)

Throw `ArgumentError` if `params` violates invariants implied by `compute_params` / the
manuscript layout (processor grid, replication axis, block-size window, panel reduction cap).

When `N` is passed, also checks the replication memory identity
`c = max(1, ⌊PM/N²⌋)` and `b_max = ⌊N / P_x⌋`.
"""
function verify_budget(params::DeviceParams; N::Union{Nothing, Integer} = nothing)
	p = params
	p.P >= 1 || throw(ArgumentError("verify_budget: P ($(p.P)) must be ≥ 1"))
	p.M >= 1 || throw(ArgumentError("verify_budget: M ($(p.M)) must be ≥ 1"))
	p.c >= 1 || throw(ArgumentError("verify_budget: c ($(p.c)) must be ≥ 1"))
	p.Pz == p.c || throw(ArgumentError("verify_budget: Pz ($(p.Pz)) must equal c ($(p.c))"))
	p.Py == p.Px || throw(ArgumentError("verify_budget: Py ($(p.Py)) must equal Px ($(p.Px))"))
	p.P1 == max(1, p.P ÷ p.c) ||
		throw(ArgumentError("verify_budget: P1 ($(p.P1)) must equal max(1, P÷c) ($(max(1, p.P ÷ p.c)))"))
	p.Px == max(1, isqrt(p.P1)) ||
		throw(ArgumentError("verify_budget: Px ($(p.Px)) must equal max(1, isqrt(P1)) ($(max(1, isqrt(p.P1))))"))
	p.b_min == p.c || throw(ArgumentError("verify_budget: b_min ($(p.b_min)) must equal c ($(p.c))"))
	p.b_max >= 1 || throw(ArgumentError("verify_budget: b_max ($(p.b_max)) must be ≥ 1"))
	p.b < 0 && throw(ArgumentError("verify_budget: b ($(p.b)) must be ≥ 0"))
	p.b > 0 && !(p.b_min <= p.b <= p.b_max) &&
		throw(ArgumentError("verify_budget: b ($(p.b)) must lie in [b_min, b_max] = [$(p.b_min), $(p.b_max)]"))
	p.Px * p.Pz <= _WORKGROUP_REDUCE_MAX ||
		throw(ArgumentError("verify_budget: Px·Pz ($(p.Px * p.Pz)) exceeds panel reduction cap $(_WORKGROUP_REDUCE_MAX)"))

	if N !== nothing
		N_int = Int(N)
		N_int >= 1 || throw(ArgumentError("verify_budget: N ($N_int) must be ≥ 1"))
		p.b_max == N_int ÷ p.Px ||
			throw(ArgumentError("verify_budget: b_max ($(p.b_max)) must equal N÷Px ($(N_int ÷ p.Px))"))
		PM = p.P * p.M
		N2 = N_int * N_int
		c_calc = PM ÷ N2
		c_expect = max(1, c_calc)
		p.c == c_expect ||
			throw(ArgumentError("verify_budget: c ($(p.c)) must equal max(1, PM÷N²) ($(c_expect))"))
		if c_calc >= 1
			PM >= p.c * N2 || throw(ArgumentError("verify_budget: need P·M ≥ c·N² (got PM=$PM, c·N²=$(p.c * N2))"))
		else
			PM < N2 || throw(ArgumentError("verify_budget: c=1 fallback requires P·M < N² (got PM=$PM, N²=$N2)"))
		end
	end
	return nothing
end

@kernel function workgroup_reduce_kernel!(out, src, len::Int, wg::Int, nlevels::Int)
	i = @index(Local, Linear)
	scratch = @localmem eltype(src) _WORKGROUP_REDUCE_MAX
	@inbounds scratch[i] = i <= len ? src[i] : zero(eltype(src))
	for lvl in 1:nlevels
		@synchronize
		@uniform halfw = wg >> lvl
		@inbounds if i <= halfw
			scratch[i] = scratch[i] + scratch[i + halfw]
		end
	end
	@synchronize
	if i == 1
		@inbounds out[1] = scratch[1]
	end
end

# One workgroup (`ndrange == workgroupsize`): local threads partition linear indices `1:(b*b)`
# and each sums `partials[:, :, k]` over all `k`. Host enforces the same power-of-two tree
# bound on `K` as a future tree reduction over `k`.
@kernel function panel_allreduce_kernel!(G, partials, b::Int, K::Int)
	wgsize = @groupsize()[1]
	i = @index(Local, Linear)

	innerblock_size = fld((size(partials, 1) * size(partials, 2)), wgsize)
	overflow = i == wgsize ? (size(partials, 1) * size(partials, 2)) % wgsize : 0
	inner_range = ((i-1) * innerblock_size + 1):(i * innerblock_size + overflow)

	for k in 1:K
		for idx in inner_range
			row = mod(idx - 1, b) + 1
			col = div(idx - 1, b) + 1
			@inbounds G[row, col] += partials[row, col, k]
		end
	end
end
