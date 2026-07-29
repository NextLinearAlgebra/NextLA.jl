"""Compatibility entry point for the dense-output campaign."""
include(joinpath(@__DIR__, "dense", "run_experiments.jl"))

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
