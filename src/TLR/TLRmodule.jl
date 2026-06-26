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
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default

export TileOrderStyle, TileOrder, ColMajor, RowMajor, TileColMajor, TileRowMajor
export TileMap
export AbstractTLRStorage, UniformTileStorage, CompactTileStorage
export TLRMatrix
export compress!, compact!, alloc_workspace
export CPUCompressWorkspace, GPUCompressWorkspace
export tile_u, tile_v

include("geometry/order.jl")
include("geometry/tilemap.jl")

include("storage/abstract_storage.jl")
include("storage/uniform_storage.jl")
include("storage/compact_storage.jl")

include("container/tlrmatrix.jl")
include("container/access.jl")

include("experimental/rademacher_sampling.jl")
include("algorithms//compress.jl")
include("algorithms/compact.jl")

end
