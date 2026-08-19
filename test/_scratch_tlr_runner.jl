using NextLA
using Test
using LinearAlgebra
using LinearAlgebra.LAPACK
using Random

include("lapack_helpers.jl")
include("gpu_backends.jl")
include("backend_test_helpers.jl")
backends = available_backends()
@info "Test backends" backends=[b[1] for b in backends]

include("TLR/helpers.jl")
include("TLR/containers.jl")
include("TLR/compressed_ftlr.jl")
include("TLR/numerics.jl")
include("TLR/ara.jl")
include("TLR/compress.jl")
include("TLR/dense_gemm.jl")
include("TLR/compressed_ftlr_dense_gemm.jl")
include("TLR/tlr_gemm.jl")
