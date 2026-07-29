module TLROutputOverlapSweep

using Main.TLROutputExperiment

const MATRIX_SIZE = 16384
const TILE_SIZE = 512
const RANKS = (64, 64)
const OUTPUT_RANK = 128
const SHARED_RANKS = [0, 8, 16, 32, 48, 64]

run(run_config) = tlr_output_overlap_sweep(TLROutputOverlapConfig(
    MATRIX_SIZE, TILE_SIZE, RANKS, OUTPUT_RANK, SHARED_RANKS, run_config))

end
