# QR-focused subset of `runtests.jl` (no LU, Schur, etc.).
using NextLA
using Test
using LinearAlgebra
using LinearAlgebra.LAPACK
using Random

include(joinpath(@__DIR__, "lapack_helpers.jl"))
include(joinpath(@__DIR__, "gpu_backends.jl"))

# Same helpers as `runtests.jl` (needed by `geqrf_2p5d.jl`).
function parameter_creation(path::String, imat::Int, M::Int, N::Int)
	kl = 0
	ku = 0
	anorm = 1.0
	mode = :decay
	cndnum = 10.0
	dist = :uniform
	type = 'N'
	if path == "GE"
		type = 'N'
		if imat == 1
			mode = :decay
			cndnum = 2.0
		elseif imat == 2
			mode = :decay
			cndnum = 1e2
		elseif imat == 3
			mode = :one_large
			cndnum = 1e6
		elseif imat == 4
			kl = 2
			ku = 2
			mode = :decay
			cndnum = 1e2
		elseif imat == 5
			mode = :decay
			cndnum = 1e8
		elseif imat == 6
			mode = :decay
			cndnum = 1e12
		end
	end
	return (; type, kl, ku, anorm, mode, cndnum, dist)
end

function matrix_generation(type, M, N; dist = :uniform, mode = :decay,
	cndnum = 10.0, anorm = 1.0, kl = 0, ku = 0, seed = 1234)
	rng = MersenneTwister(seed)
	(M == 0 || N == 0) && return zeros(type, M, N)
	k = min(M, N)
	σ = zeros(real(type), k)
	if mode == :one_large
		σ .= 1.0
		k > 0 && (σ[1] = cndnum)
	elseif mode == :decay
		if k == 1
			σ[1] = anorm
		else
			for i in 1:k
				σ[i] = anorm * (1 / cndnum)^((i - 1) / (k - 1))
			end
		end
	elseif mode == :random
		σ .= rand(rng, real(type), k)
		maximum(σ) > 0 && (σ .*= anorm / maximum(σ))
	else
		error("Unsupported mode")
	end
	U, _ = qr(randn(rng, type, M, M))
	V, _ = qr(randn(rng, type, N, N))
	Σ = Diagonal(σ)
	A = U[:, 1:k] * Σ * V[:, 1:k]'
	if kl > 0 || ku > 0
		for i in 1:M, j in 1:N
			(j > i + ku || i > j + kl) && (A[i, j] = zero(type))
		end
	end
	return A
end

include(joinpath(@__DIR__, "geqr2.jl"))
include(joinpath(@__DIR__, "geqrt.jl"))
include(joinpath(@__DIR__, "scqr3.jl"))
include(joinpath(@__DIR__, "geqrf_2p5d.jl"))
