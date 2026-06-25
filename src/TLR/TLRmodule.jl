"""
    TLRmodule

Internal module implementing tile low-rank containers, geometry, storage, and
compression algorithms used by `NextLA`.
"""
module TLRmodule

using LinearAlgebra
using Random
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using ..NextLA: gemm_batched!, gemmEx_batched!, syrk_batched!, trsm_batched!, potrf_batched!

export TileOrderStyle, TileOrder, ColMajor, RowMajor, TileColMajor, TileRowMajor
export TileMap
export AbstractTLRStorage, UVTileStorage
export TLRMatrix
export compress!

include("geometry/order.jl")
include("geometry/tilemap.jl")

include("storage/abstract_storage.jl")
include("storage/uv_storage.jl")

include("container/tlrmatrix.jl")
include("container/access.jl")

include("experimental/rademacher_sampling.jl")
include("algorithms/compress/compress.jl")

end
