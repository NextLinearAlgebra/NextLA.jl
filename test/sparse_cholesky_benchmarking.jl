# Catalog expansion: 20 downloaded real matrices, checkpointed results and process timeouts.
# See test/catalog/README.md. Synthetic dense benchmarks are not repeated here.
include(joinpath(@__DIR__, "catalog", "run.jl"))
abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
