using TOML, Dates
include("../catalog/common.jl")

const CATALOG = normpath(joinpath(@__DIR__, "..", "catalog"))
const PRIORITY = ["inla_graph_spacetime", "inla_graph_83o4NNNo", "bodyy5",
                  "inla_graph_sem_n5000", "inla_graph_animal1"]
const PREFIX = ["matrix", "ordering", "status", "detail"]
const SUM_FIELDS = split("tiles empty_tiles input_tiles fill_only_tiles partial_row_tiles partial_col_tiles contiguous_partial_row_tiles contiguous_partial_col_tiles dense_cells kept_cells row_packed_cells col_packed_cells both_packed_cells dense_schur_products source_schur_products shared_col_schur_products")
const SUMMARY = vcat(PREFIX, split("tile_width factor_nnz symbolic_seconds panels empty_panels partial_row_panels partial_col_panels active_panel_rows active_panel_cols panel_rows panel_cols empty_tile_fraction skipped_cell_fraction"), SUM_FIELDS)

number(row, key) = parse(Int, row[key])

function summarize_panels(rows)
    first(rows)["status"] == "OK" || return [first(rows)]
    output = Dict{String,Any}[]
    for width in unique(getindex.(rows, "tile_width"))
        selected = filter(row -> row["tile_width"] == width, rows)
        row = Dict{String,Any}(first(selected))
        total(key) = sum(r -> number(r, key), selected)
        for field in vcat(SUM_FIELDS, ["active_panel_rows", "active_panel_cols", "panel_rows", "panel_cols"])
            row[field] = total(field)
        end
        row["panels"] = length(selected)
        row["empty_panels"] = count(r -> number(r,"factor_entries") == 0, selected)
        row["partial_row_panels"] = count(r -> 0 < number(r,"active_panel_rows") < number(r,"panel_rows"), selected)
        row["partial_col_panels"] = count(r -> 0 < number(r,"active_panel_cols") < number(r,"panel_cols"), selected)
        row["empty_tile_fraction"] = row["empty_tiles"] / row["tiles"]
        row["skipped_cell_fraction"] = 1 - row["kept_cells"] / row["dense_cells"]
        push!(output, row)
    end
    output
end

function record_gpu_pairs(out, rows)
    fields = ["matrix", "ordering", "method", "natural_status", "reordered_status",
              "natural_ms", "reordered_ms", "speedup", "natural_forward_error",
              "reordered_forward_error", "natural_factor_residual", "reordered_factor_residual"]
    base = Dict((r["matrix"], r["method"])=>r for r in rows if r["ordering"] == "natural")
    pairs = Dict{String,Any}[]
    for row in rows
        row["ordering"] == "natural" && continue
        reference = get(base, (row["matrix"], row["method"]), Dict{String,String}())
        pair = Dict{String,Any}(field=>row[field] for field in fields[1:3])
        pair["natural_status"] = get(reference, "status", "MISSING")
        pair["reordered_status"] = row["status"]
        for (key, source) in (("natural", reference), ("reordered", row))
            pair[key * "_ms"] = get(source, "median_ms", "")
            for field in ("forward_error", "factor_residual")
                pair[key * "_" * field] = get(source, field, "")
            end
        end
        if pair["natural_status"] == pair["reordered_status"] == "OK"
            pair["speedup"] = parse(Float64, pair["natural_ms"]) / parse(Float64, pair["reordered_ms"])
        end
        push!(pairs, pair)
    end
    CatalogSweep.write_csv(joinpath(out, "gpu-pairs.csv"), pairs; columns=fields)
end

function main()
    mode = only(ARGS)
    mode in ("symbolic", "gpu") || error("Expected symbolic or gpu")
    config = CatalogSweep.settings()
    entries = vcat(CatalogSweep.read_csv(joinpath(CATALOG, "matrices.csv")),
                   CatalogSweep.read_csv(joinpath(CATALOG, "legacy-matrices.csv")))
    by_name = Dict(row["name"]=>row for row in entries)
    selection = strip.(split(get(ENV, "REAL_MATRICES", join(PRIORITY, ',')), ','))
    names = selection == ["all"] ? vcat(PRIORITY, setdiff(getindex.(entries,"name"), PRIORITY)) : unique(selection)
    all(haskey(by_name, name) for name in names) || error("Unknown matrix in REAL_MATRICES")
    orders = unique(strip.(split(get(ENV, "SPARSITY_ORDERS", "natural,amd,scotch"), ',')))
    all(o -> o in ("natural", "amd", "scotch", "stiles", "rcm"), orders) || error("Invalid SPARSITY_ORDERS")
    mode == "gpu" && (orders = union(["natural"], orders))
    out = abspath(get(ENV, "SPARSITY_OUTPUT", joinpath(CatalogSweep.ROOT, "benchmark-results", "sparsity")))
    mkpath(out)
    manifest = joinpath(out, "matrices.csv")
    CatalogSweep.write_csv(manifest, entries; columns=["name", "group", "application", "n", "expected_nnz"])
    config["manifest"] = manifest
    config["input_hashes"] = Dict{String,String}()
    paths = Dict{String,String}()
    legacy = expanduser(get(ENV, "LEGACY_MATRIX_DIR", config["matrix_dir"]))
    for name in names
        candidates = [joinpath(dir, name * ".mtx") for dir in (config["matrix_dir"], legacy)]
        found = findfirst(isfile, candidates)
        found === nothing || (paths[name] = candidates[found])
    end
    for (name, path) in paths
        config["input_hashes"][name] = CatalogSweep.file_hash(path)
    end
    stop_at = min(time() + config["sweep_timeout"],
                  parse(Float64, get(ENV,"SLURM_JOB_END_TIME","Inf")) - 60)
    timeout = CatalogSweep.positive_env("SYMBOLIC_CASE_SECONDS", 240; integer=true)
    rows = Dict{String,Any}[]
    gpu_fields = unique(vcat(PREFIX, CatalogSweep.COLUMNS))
    n_ok = 0
    for name in names, order in orders
        remaining = floor(Int, max(0, stop_at - time()))
        remaining > 0 || break
        case = joinpath(out, "cases", name, order)
        mkpath(case)
        row = Dict{String,Any}("matrix"=>name, "ordering"=>order)
        # An unbounded run is inappropriate for the roughly half-billion-entry input.
        if mode == "symbolic" && name == "inla_graph_animal2" && selection == ["all"]
            row["status"], row["detail"] = "OPT_IN", "Select this matrix explicitly to analyze it"
        elseif !haskey(paths, name)
            row["status"], row["detail"] = "MISSING", "No input in MATRIX_DIR or LEGACY_MATRIX_DIR"
        elseif mode == "symbolic"
            result = joinpath(case, "panels.csv")
            isfile(result) && rm(result)
            plan = joinpath(case, "plan.toml")
            isfile(plan) && rm(plan)
            path = paths[name]
            n = by_name[name]["n"]
            command = `$(Base.julia_cmd()) --startup-file=no --project=$(@__DIR__) $(joinpath(@__DIR__, "symbolic_case.jl")) $name $n $path $order $case`
            status, _, elapsed = CatalogSweep.run_bounded(command, min(remaining, timeout), joinpath(case, "symbolic.log"))
            row["status"], row["detail"] = status == "DONE" ? "MISSING_RESULT" : status, "Process wall time $(round(elapsed; digits=2)) s. See symbolic.log."
            if isfile(result)
                measured = summarize_panels(CatalogSweep.read_csv(result))
                append!(rows, measured)
                n_ok += first(measured)["status"] == "OK"
            end
        else
            plan = joinpath(case, "plan.toml")
            output = joinpath(case, "gpu.csv")
            isfile(output) && rm(output)
            if order == "natural" || isfile(plan)
                config["matrix_dir"] = dirname(paths[name])
                config_path = joinpath(case, "gpu-settings.toml")
                open(io -> TOML.print(io, config), config_path, "w")
                worker = joinpath(CATALOG, "worker.jl")
                methods = join(config["gpu_methods"], ',')
                command = `$(Base.julia_cmd()) --startup-file=no --project=$CATALOG $worker $config_path $name gpu $output $methods`
                order == "natural" || (command = `$command $plan`)
                status, _, elapsed = CatalogSweep.run_bounded(command, min(remaining, floor(Int, config["gpu_timeout"])), joinpath(case,"gpu.log"))
                row["status"], row["detail"] = status == "DONE" ? "MISSING_RESULT" : status, "Process wall time $(round(elapsed; digits=2)) s. See gpu.log."
            else
                row["status"], row["detail"] = "MISSING_PLAN", "Run symbolic mode first for this ordering"
            end
            completed = isfile(output) ? CatalogSweep.read_csv(output) : []
            for method in config["gpu_methods"]
                i = findfirst(r -> r["method"] == method, completed)
                status_row = copy(row)
                status_row["method"] = method
                i === nothing || merge!(status_row, completed[i])
                n_ok += status_row["status"] == "OK"
                push!(rows, status_row)
            end
        end
        if mode == "symbolic" && !any(r -> r["matrix"] == name && r["ordering"] == order, rows)
            push!(rows, row)
        end
        CatalogSweep.write_csv(joinpath(out, mode * "-summary.csv"), rows; columns=mode == "gpu" ? gpu_fields : SUMMARY)
        mode == "gpu" && record_gpu_pairs(out, rows)
        println("$mode $name / $order: ", last(rows)["status"])
        flush(stdout)
    end
    n_ok > 0 || error("No successful $mode cases. See the case logs and summary in $out.")
end

main()
