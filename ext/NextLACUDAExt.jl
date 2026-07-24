module NextLACUDAExt

using NextLA
using CUDA

include("cuda/common.jl")
include("cuda/gemm.jl")
include("cuda/syrk.jl")
include("cuda/trsm.jl")
include("cuda/potrf.jl")
include("cuda/streams.jl")

function NextLA.TLRmodule._row_basis_eigh!(A::CUDA.StridedCuMatrix{T}) where {T<:Union{Float32,Float64}}
    return CUDA.CUSOLVER.syevd!('V', 'U', A)
end

end
