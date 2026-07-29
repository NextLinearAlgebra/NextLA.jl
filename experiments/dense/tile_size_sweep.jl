module DenseTileSizeSweep
using Main.DenseGemmCommon
using Main.TileSizeSweepExperiment

const MATRIX_SIZE = 16384
const TILE_SIZES = [128, 256, 512, 1024, 2048]
const RANK_TILE_RATIO = 1 / 8
const CASES = [
    MatrixCase(:padded_constant, :padded, :constant, nothing, nothing),
    MatrixCase(:compressed_constant, :compressed, :constant, nothing, nothing),
]

run(run_config) = tile_size_sweep(TileSizeSweepConfig(
    MATRIX_SIZE, TILE_SIZES, RANK_TILE_RATIO, CASES, run_config))
end
