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
export compress!
export tile_u, tile_v
export blocksize, maxrank, compress_diag, ranks, dense_diag
export left_factors, right_factors, nstored_tiles
export ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_sizes
export offdiag_linear_index, tile_linear_index, tile_storage_index, inverse_tile_index
export tile_stride, tile_coords, inverse_tile_coords

include("geometry/order.jl")
include("geometry/tilemap.jl")

include("container/tlrmatrix.jl")
include("container/access.jl")

include("experimental/rademacher_sampling.jl")
include("algorithms/compress.jl")
include("algorithms/gemm.jl")

end
