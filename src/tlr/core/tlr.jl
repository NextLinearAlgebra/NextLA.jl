"""
    TileLowRank

Tile low-rank (TLR) matrices. A `TLRMatrix` splits a matrix into tiles, keeps the
diagonal tiles dense and stores each off-diagonal tile as low-rank factors `U*V'`;
a `CompressedFTLRMatrix` stores every tile that way. Both are built by compressing a
dense matrix, expanded back with `uncompress!`, and multiplied with TLR methods of
`gemm!`. Opt in with `using NextLA.TileLowRank`; this module is not re-exported by `NextLA`.
"""
module TileLowRank

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
using ..NextLA: gemm_alignment_quantum, aligned_leading_dimension, is_gemm_aligned
using ..NextLA: AbstractPreparedGroupedGemm, prepare_precision_gemm_grouped
using ..NextLA: precision_gemm_grouped_prepared!, destroy_prepared_grouped_gemm!
using ..NextLA: with_grouped_host_pointer_mode, with_grouped_device_pointer_mode
using ..NextLA: PreparedGroupedGemmBundle
using ..NextLA: create_streams, with_stream, sync_stream, sync_streams_with_default
using ..NextLA: create_event, record_event!, wait_event!, sync_event
using ..NextLA: SUBGROUP_SIZE, unwrap, BFloat16
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

include("interface.jl")
include("compressed_ftlr_matrix.jl")
include("tlr_matrix.jl")

include("compression/blas_utils.jl")
include("compression/norms.jl")
include("compression/ara.jl")
include("compression/workspace.jl")
include("compression/compress.jl")
include("compression/uncompress.jl")

include("../blas/gemm/compute_policy.jl")
include("../blas/gemm/arena.jl")

include("../blas/gemm/dense_accumulation/workspace.jl")
include("../blas/gemm/dense_accumulation/runs.jl")
include("../blas/gemm/dense_accumulation/schedule.jl")
include("../blas/gemm/dense_accumulation/three_stage.jl")
include("../blas/gemm/dense_accumulation/two_stage.jl")
include("../blas/gemm/dense_accumulation/dense_diagonal.jl")
include("../blas/gemm/dense_accumulation/driver.jl")

include("../blas/gemm/compressed_accumulation/workspace.jl")
include("../blas/gemm/compressed_accumulation/run_coupling.jl")
include("../blas/gemm/compressed_accumulation/rolling_schedule.jl")
include("../blas/gemm/compressed_accumulation/driver.jl")

end
