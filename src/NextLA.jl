module NextLA

using LinearAlgebra
import LinearAlgebra
import LinearAlgebra: Adjoint, BLAS, Diagonal, Bidiagonal, Tridiagonal, LAPACK
import LinearAlgebra: LowerTriangular, PosDefException, Transpose, UpperTriangular
import LinearAlgebra: UnitLowerTriangular, UnitUpperTriangular, diagind, ishermitian, issymmetric
import LinearAlgebra: PivotingStrategy, BlasFloat, BlasInt
import Random
using KernelAbstractions
using StaticArrays

DEV = :NVIDIA

if DEV == :NVIDIA
    using CUDA
    ArrayKA = CUDA.CuArray
    Backend = CUDA.CUDABackend()
elseif DEV == :AMD
    using AMDGPU
    ArrayKA = AMDGPU.ROCArray
    Backend = AMDGPU.ROCBackend()
elseif DEV == :oneAPI
    using oneAPI 
    ArrayKA = oneAPI.oneArray
    Backend = oneAPI.oneAPIBackend()
elseif DEV == :Metal
    using Metal 
    ArrayKA = Metal.MtlArray
    Backend = Metal.MetalBackend()
else DEV == :CPU
    ArrayKA = Array
    Backend = CPU()
end

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
include("NextLAMatrix.jl")
include("lu.jl")
include("trmm.jl")
include("trsm.jl")
include("rectrxm.jl")
include("matmul.jl")
include("zlauu2.jl") 
include("zlauum.jl")

end
