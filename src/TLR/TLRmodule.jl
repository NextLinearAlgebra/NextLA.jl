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

using ..NextLA: gemm_batched!, gemmEx_batched!, syrk_batched!, trsm_batched!, potrf_batched!
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default
using ..NextLA: SUBGROUP_SIZE, unwrap

export TileColMajor, TileRowMajor
export AbstractTLRMatrix, TLRMatrix, FullTLRMatrix
export compress!, alloc_workspace
export uncompress!
export get_factors
export maxrank, ranks, residuals, dense_diag, dense_diag_corner, tilegrid_size
export nominal_tile_size, tail_tile_size
export ndiag_tiles, noffdiag_tiles, tile_origin_coords, tile_size

include("container/order.jl")
include("container/abstract.jl")
include("container/dense_diag.jl")
include("container/full.jl")

include("algorithms/compress.jl")
include("algorithms/uncompress.jl")
include("algorithms/gemm/gemm.jl")

end
