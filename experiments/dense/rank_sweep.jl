module DenseRankSweep
using Main.DenseGemmCommon
using Main.RankSweepExperiment

const MATRIX_SIZE = 16384
const TILE_SIZE = 512
const RANKS = [8, 16, 32, 64, 128, 256]
const CASES = [
    MatrixCase(:padded_constant, :padded, :constant, 0, 0),
    MatrixCase(:compressed_constant, :compressed, :constant, 0, 0)
]

run(run_config) = rank_sweep(RankSweepConfig(
    MATRIX_SIZE, TILE_SIZE, RANKS, CASES, run_config))
end
