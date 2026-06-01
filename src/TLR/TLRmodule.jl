module TLRmodule

using LinearAlgebra
using KernelAbstractions

export AbstractTileOrder, TileColMajor, TileRowMajor
export TileFactorBuffer
export TLRMatrix, GeneralTLRMatrix, similar_tlr, tile_linear_index

include("tile_order.jl")
include("tile_views.jl")
include("uniform_tlr.jl")
include("general_tlr.jl")

end
