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

import ..NextLA: gemm, gemm!
using ..NextLA: gemm_batched!, trsm_batched!, potrf_batched!
using ..NextLA: precision_gemm!, precision_gemm_batched!, precision_gemm_grouped!, GroupedGemmTask
using ..NextLA: gemm_compute_mode, gemm_compute_type, supports_grouped_gemm, supports_bfloat16_grouped_gemm
using ..NextLA: GEMMCompute, validate_gemm_signature
using ..NextLA: gemm_alignment_quantum
using ..NextLA: AbstractPreparedGroupedGemm, prepare_precision_gemm_grouped
using ..NextLA: precision_gemm_grouped_prepared!, destroy_prepared_grouped_gemm!
using ..NextLA: _with_grouped_host_pointer_mode, _with_grouped_device_pointer_mode
using ..NextLA: PreparedGroupedGemmBundle
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default
using ..NextLA: create_event, record_event!, wait_event!, sync_event
using ..NextLA: SUBGROUP_SIZE, unwrap
using ..NextLA: BatchPtrDescriptor, swap_batch_ptrs!, set_batch_ptrs!
using ..NextLA: precision_gemm_batched_ptrs!
using ..NextLA: supports_pointer_batched

export TileColMajor, TileRowMajor
export AbstractTLRMatrix, TLRMatrix, CompressedFTLRMatrix
export offdiagonal
export uncompress!
export get_factors
export FTLRCompressionWorkspace
export maxrank, ranks, rank_multiple, maximum_storage_rank
export residuals, dense_diag, dense_diag_corner, grid_size
export nominal_tile_size, tail_tile_size
export ndiag_tiles, tile_origin_coords, tile_size
export gemm_minimum_workspace_bytes, gemm_maximum_workspace_bytes, gemm_workspace_bytes
export DenseGemmWorkspace
export CompressedGemmAnalysis, CompressedMixedGemmAnalysis, analyze_compressed_gemm

include("container/order.jl")
include("container/abstract.jl")
include("container/compressed_ftlr_matrix.jl")
include("container/tlr_matrix.jl")

include("numerics/precision.jl")
include("numerics/norms.jl")
include("numerics/ara.jl")

include("compression/compress.jl")
include("compression/uncompress.jl")
include("gemm/common/precision.jl")
include("gemm/common/arena.jl")
include("gemm/dense_result.jl")
include("gemm/compressed_result.jl")

end
