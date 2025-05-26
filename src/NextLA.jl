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

include("NextLAMatrix.jl")
include("axpy.jl")   
include("lauum.jl")         
include("rectrxm.jl") 
include("zlarfb_v0.jl")   
include("zlarfg.jl")  
include("zpemv.jl")   
include("zunmqr_v0.jl")
include("gerc.jl")    
include("lu.jl")            
include("trmm.jl")     
include("zlarfb_v1.jl")   
include("zlarf.jl")   
include("ztsmqr.jl") 
include("zunmqrwrap.jl")
include("getc2.jl")   
include("matmul.jl")        
include("trsm.jl")     
include("zlarfb_v2.jl")   
include("zlarft.jl")  
include("ztsqrt.jl")
include("getrf2.jl")  
include("zgeqr2.jl")   
include("zlarfb_v3.jl")   
include("zpamm.jl")   
include("zttmqr.jl")
include("lauu2.jl")   
include("zgeqrt.jl")   
include("zlarfbwrap.jl")  
include("zparfb.jl")  
include("zttqrt.jl")

end
