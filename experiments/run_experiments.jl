"""Run every Julia benchmark campaign in a fresh process."""

const RUNNERS = (
    joinpath(@__DIR__, "dense_output", "run_experiments.jl"),
    joinpath(@__DIR__, "padded_ftlr_output", "run_experiments.jl"),
)

function main()
    project = abspath(@__DIR__)
    for script in RUNNERS
        println("\n==> ", relpath(script, @__DIR__))
        flush(stdout)
        run(`$(Base.julia_cmd()) --project=$project $script`)
    end
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
