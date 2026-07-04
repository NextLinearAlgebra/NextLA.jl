"""
    TLRmodule

Internal module implementing tile low-rank containers, geometry, and
compression algorithms used by `NextLA`.
"""
module TLRmodule

using LinearAlgebra
using Random
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using ..NextLA: gemm_batched!, gemmEx_batched!, syrk_batched!, trsm_batched!, potrf_batched!
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default

export TileColMajor, TileRowMajor
export TLRMatrix
export compress!, alloc_workspace, workspace_info
export uncompress!
export tile_u, tile_v
export blocksize, maxrank, ranks, residuals, dense_diag, dense_diag_corner, tilegrid_size
export left_factors, right_factors
export ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_size
export alloc_workspace, workspace_info

include("container/order.jl")

include("container/tlrmatrix.jl")

include("algorithms/compress.jl")
include("algorithms/uncompress.jl")
include("algorithms/gemm/gemm.jl")

end
