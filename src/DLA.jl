module DLA

using LinearAlgebra
import LinearAlgebra
import LinearAlgebra: Adjoint, BLAS, Diagonal, Bidiagonal, Tridiagonal, LAPACK, LowerTriangular, PosDefException, Transpose, UpperTriangular, UnitLowerTriangular, UnitUpperTriangular, diagind, ishermitian, issymmetric, PivotingStrategy, BlasFloat, BlasInt
import Random



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
function lamch(::Type{T}, cmach) where{T<: Number}
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

# Write your package code here.
include("DLAMatrix.jl")
include("lu.jl")
include("zlauum.jl")
end
