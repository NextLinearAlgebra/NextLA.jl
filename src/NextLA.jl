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

include("NextLAMatrix.jl")
include("lu.jl")
include("trmm.jl")
include("trsm.jl")
include("rectrxm.jl")
include("matmul.jl")
include("lauu2.jl")
include("lauum.jl")

include("geqrt.jl")
include("geqr2.jl")
include("larf.jl")
include("larfg.jl")
include("larft.jl")
include("larfb.jl")
include("unmqr.jl")
include("gerc.jl")
include("tsqrt.jl")
include("tsmqr.jl")
include("parfb.jl")
include("pamm.jl")
include("axpy.jl")
include("pemv.jl")
include("ttmqr.jl")
include("ttqrt.jl")
include("slas2.jl")
include("srot.jl")
include("slamrg.jl")
end
