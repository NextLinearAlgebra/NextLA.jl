module NextLA

using LinearAlgebra
import LinearAlgebra: Adjoint, Diagonal, Bidiagonal, Tridiagonal
import LinearAlgebra: LowerTriangular, PosDefException, Transpose, UpperTriangular
import LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular, diagind, ishermitian, issymmetric
import LinearAlgebra: PivotingStrategy, BlasFloat, BlasInt
import LinearAlgebra: BLAS, LAPACK
import LinearAlgebra.BLAS: @blasfunc
using Random: Random
using KernelAbstractions
# @unroll: chol_kernel_register! unrolls over its register strip. TileLowRank
# imports it separately, being its own module.
using KernelAbstractions.Extras: @unroll

# BFloat16s.jl's BFloat16: Core.BFloat16 with its conversions and arithmetic from Julia
# 1.11, and a software BFloat16 before that, so one working type on every version.
using BFloat16s: BFloat16

"""
	lamch(::Type{T}, cmach) where{T<: Number}

Determines single / double precision machine parameters

# Arguments
- T : type, currently only tested Float32 and Float64
- 'cmach' : specifies the value to be returned by lamch
	- = 'E': returns eps
	- = 'S': returns sfmin
	- = 'P': returns eps*base
	
	- where
		- eps = relative machine precision
		- sfmin = safe min, such that 1/sfmin does not overflow
		- base = base of the machine

# Examples
```jldoctest
julia> NextLA.lamch(Float64, 'E') == eps(Float64) / 2
true

julia> NextLA.lamch(Float32, 'S') == floatmin(Float32)
true
```
"""
function lamch(::Type{T}, cmach) where {T <: Number}
	ep = eps(T)
	one = oneunit(T)
	rnd = one

	if one == rnd
		ep *= 0.5
	end

	if cmach == 'E'
		return ep
	elseif cmach == 'S'
		sfmin = floatmin(T)
		small = one / floatmax(T)

		if small >= sfmin
			sfmin = small*(one + ep)
		end
		return sfmin
	else # assume cmach = 'P'
		# assume base of machine is 2
		return ep*2
	end
end

@inline SUBGROUP_SIZE(::Type{<:KernelAbstractions.CPU}) = Val(1)
@inline unwrap(::Val{x}) where {x} = x
@inline supports_pointer_batched(backend) = supports_pointer_batched(typeof(backend))
@inline supports_pointer_batched(::Type) = false

@inline _rec_split(n::Integer) = ispow2(n) ? n ÷ 2 : prevpow(2, n)

include("streams.jl")
include("dense/core/NextLAMatrix.jl")
include("dense/blas/gemm/gemm.jl")
include("dense/blas/trsm/trsm.jl")
include("dense/blas/syrk/syrk.jl")
include("dense/blas/axpy.jl")
include("dense/blas/gerc.jl")
include("dense/lapack/lu/lu.jl")
include("dense/lapack/cholesky/cholesky.jl")
include("dense/lapack/qr/qr.jl")
include("dense/lapack/rotations/rotations.jl")
include("mxp/core/mxp.jl")
include("tlr/core/tlr.jl")

end
