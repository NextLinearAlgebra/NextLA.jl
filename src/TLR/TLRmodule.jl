"""
    TLRmodule

Internal module implementing tile low-rank containers, geometry, storage, and
compression algorithms used by `NextLA`.
"""
module TLRmodule

using LinearAlgebra
using KernelAbstractions
using KernelAbstractions.Extras: @unroll

using ..NextLA: gemm_batched!, gemm_batched_ptrs!, syrk_batched!, trsm_batched!, potrf_batched!, supports_pointer_batched

export TileOrderStyle, TileOrder, ColMajor, RowMajor, TileColMajor, TileRowMajor
export TileMap
export AbstractTLRStorage, UVTileStorage
export AbstractTLROperator, TLRLinearOperator
export TLRMatrix
export compress!

include("geometry/order.jl")
include("geometry/tilemap.jl")

include("storage/abstract_storage.jl")
include("storage/uv_storage.jl")

include("container/tlrmatrix.jl")
include("container/access.jl")
include("operator.jl")

include("algorithms/compress/to_tiles.jl")
include("algorithms/compress/prepare_dense.jl")
include("algorithms/compress/prepare_operator.jl")
include("algorithms/compress/ara_core.jl")
include("algorithms/compress/ara_dense_sampling.jl")
include("algorithms/compress/ara_operator_sampling.jl")
include("experimental/rademacher_sampling.jl")
include("algorithms/compress/interface.jl")
include("algorithms/compress/compress.jl")

end
