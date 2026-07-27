"""
    TLRmodule

Internal module implementing tile low-rank containers, geometry, and
compression algorithms used by `NextLA`.
"""
module TLRmodule

using LinearAlgebra
using Random
using KernelAbstractions
using KernelAbstractions: zeros, allocate
using KernelAbstractions.Extras: @unroll

using ..NextLA: gemm_batched!, trsm_batched!, potrf_batched!
using ..NextLA: precision_gemm!, precision_gemm_batched!, gemm_compute_mode, gemm_compute_type
using ..NextLA: GEMMCompute, validate_gemm_signature
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default
using ..NextLA: SUBGROUP_SIZE, unwrap
using ..NextLA: BatchPtrDescriptor, swap_batch_ptrs!, precision_gemm_batched_ptrs!
using ..NextLA: supports_pointer_batched

export TileColMajor, TileRowMajor
export AbstractTLRMatrix, TLRDenseDiagMatrix, TLRMatrix
export compress!, alloc_workspace
export uncompress!
export get_factors
export maxrank, ranks, residuals, dense_diag, dense_diag_corner, grid_size
export nominal_tile_size, tail_tile_size
export ndiag_tiles, tile_origin_coords, tile_size
export gemm_minimum_workspace_bytes, gemm_maximum_workspace_bytes
export DenseGemmWorkspace, InteriorFirstWorkspace
export TLRGemmWorkspace, tlr_gemm_workspace_bytes

include("container/order.jl")
include("container/abstract.jl")
include("container/dense_diag_tlr_matrix.jl")
include("container/full_tlr_matrix.jl")

include("numerics/precision.jl")
include("numerics/norms.jl")
include("numerics/ara.jl")

include("compression/compress.jl")
include("compression/uncompress.jl")
include("gemm/dense_result/axis_strategy.jl")
include("gemm/operands.jl")
include("gemm/dense_result.jl")
include("gemm/tlr_result.jl")

end
