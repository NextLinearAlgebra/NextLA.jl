using LinearAlgebra, Random, TOML
include("../catalog/common.jl")
include("../catalog/matrix_io.jl")
include("methods.jl")
include("symbolic_tiles.jl")

setting(name, default) = CatalogSweep.positive_env(name, default; integer=true)

function run_case(name, n, path, order, out)
    BLAS.set_num_threads(1)
    order == "amd" && (@eval import AMD)
    order == "scotch" && (@eval import Scotch)
    order == "stiles" && (@eval import sTiles)
    order in ("natural", "amd", "scotch", "stiles", "rcm") || error("Unsupported ordering: $order")
    widths = unique(parse.(Int, split(get(ENV, "SYMBOLIC_TILE_WIDTHS", "128,256,512,1024"), ',')))
    all(>(0), widths) || error("Tile widths must be positive")
    depth = setting("SYMBOLIC_DEPTH", 7)
    limit = setting("SYMBOLIC_NNZ_LIMIT", 30_000_000)
    A = load_input(path, parse(Int, n))
    digest = CatalogSweep.file_hash(path)
    p = nothing
    ordering_seconds = @elapsed p = Base.invokelatest(ORDERERS[order], A, 20261105)
    length(p) == size(A,1) && isperm(p) || error("Invalid permutation")
    mkpath(out)
    open(joinpath(out, "plan.toml"), "w") do io
        TOML.print(io, Dict("matrix"=>name, "ordering"=>order, "matrix_sha256"=>digest,
                            "permutation"=>p, "ordering_seconds"=>ordering_seconds))
    end
    println("Permutation ready for $name / $order; starting symbolic reach")
    flush(stdout)
    upper = order == "natural" ? A : A[p, p]
    A = nothing
    metadata = (; matrix=name, ordering=order, matrix_sha256=digest, ordering_seconds)
    rows = try
        t0 = time_ns()
        factor_nnz, panels = observe_panels(upper, regions(parse(Int,n), depth), widths; limit)
        symbolic_seconds = (time_ns() - t0) / 1e9
        [merge(metadata, (; status="OK", factor_nnz, symbolic_seconds),
            panel_metrics(panel, w, wi)) for panel in panels for (wi,w) in enumerate(widths)]
    catch e
        e isa SymbolicLimit || rethrow()
        [merge(metadata, (; status="FILL_LIMIT", detail=sprint(showerror, e)))]
    end
    isempty(rows) && error("No off-diagonal recursion regions to measure")
    columns = unique(String.(vcat(collect(keys(first(rows))), [:detail])))
    CatalogSweep.write_csv(joinpath(out, "panels.csv"),
        [Dict(String(k)=>v for (k,v) in pairs(row)) for row in rows]; columns)
    println("$name / $order: $(first(rows).status)")
end

run_case(ARGS...)
