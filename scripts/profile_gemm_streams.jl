# Focused nsys profiling of the four region streams in `gemm!` on a boundary case.
#
#   nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi \
#        --capture-range-end=stop --output=gemm_streams \
#        julia --project=../gpuenv scripts/profile_gemm_streams.jl
#
# Then open gemm_streams.nsys-rep in Nsight Systems and look at the CUDA HW rows:
# during the traced call you should see the interior / right / bottom / corner work
# on four distinct streams (the boundary regions overlapping the interior).
#
# `CUDA.@profile external=true` fences exactly one warmed-up `gemm!` for nsys, so the
# trace excludes precompilation and warmup.

using CUDA, LinearAlgebra, Random
using NextLA
const M = NextLA.TLRmodule
const T = Float64

function make_tlr(backend, m, b, r, order)
    X = M.TLRMatrix(backend, T, m, m, b, r; tile_order=order)
    randn!(X.int_U); randn!(X.int_V); randn!(X.D)
    size(X.D_corner, 3) != 0 && randn!(X.D_corner)
    size(X.right_U, 3)  != 0 && (randn!(X.right_U);  randn!(X.right_V))
    size(X.bottom_U, 3) != 0 && (randn!(X.bottom_U); randn!(X.bottom_V))
    X.ranks .= r
    return X
end

CUDA.functional() || error("no CUDA device")
backend = CUDA.CUDABackend()

# Moderate Q with a boundary tail: interior doesn't saturate the GPU, so the
# right/bottom/corner streams have room to overlap it.  n = nt·b + tail.
b, nt, tail, r = 128, 8, 64, 32
m = nt * b + tail
A = make_tlr(backend, m, b, r, M.TileRowMajor)   # (k,j): row family, fused Stage 1
B = make_tlr(backend, m, b, r, M.TileRowMajor)
C = CUDA.CuArray(randn(T, m, m))
budget = M.gemm_maximum_workspace_bytes(A, B)

# warmup (compile + caches), untraced
for _ in 1:3
    M.gemm!(C, A, B; alpha=1.0, beta=0.5, max_workspace=budget)
end
CUDA.synchronize()

# one traced call
CUDA.@profile external=true begin
    M.gemm!(C, A, B; alpha=1.0, beta=0.5, max_workspace=budget)
    CUDA.synchronize()
end
println("profiled gemm! on n=$m (b=$b, nt=$nt, tail=$tail)")
