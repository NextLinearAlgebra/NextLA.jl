module DenseMatrixShapeSweep
using Main.DenseGemmCommon
using Main.MatrixShapeSweepExperiment

const BASE_SIZE = 16384
const TILE_SIZE = 512
const RANK = 64
const RATIOS = [(1, 1, 1), (4, 1, 1), (1, 4, 1), (1, 1, 4), (1, 0.25, 1)]
const CASES = [
    MatrixCase(:padded_constant, :padded, :constant, 64, 64),
    MatrixCase(:compressed_constant, :compressed, :constant, 64, 64),
    MatrixCase(:compressed_uniform, :compressed, :uniform, 16, 64),
    MatrixCase(:compressed_skewed, :compressed, :skewed, 16, 64),
]

run(run_config) = matrix_shape_sweep(MatrixShapeSweepConfig(
    BASE_SIZE, TILE_SIZE, RANK, RATIOS, CASES, run_config))
end
