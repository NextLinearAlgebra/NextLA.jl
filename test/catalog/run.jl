# Parent process: no CUDA, sTiles or matrix-loading dependencies.
include("common.jl")
using .CatalogSweep
using Dates, TOML

function main(args=ARGS; worker=joinpath(@__DIR__, "worker.jl"))
    config = CatalogSweep.settings()
    entries = CatalogSweep.selection(config)
    if "--list" in args || "--check" in args
        for entry in entries
            path = joinpath(config["matrix_dir"], entry["name"] * ".mtx")
            println(isfile(path) ? "FOUND   " : "MISSING ", path, "  n=", entry["n"])
        end
        return
    end
    outdir = abspath(expanduser(get(ENV, "RESULTS_DIR", joinpath(CatalogSweep.ROOT,
        "benchmark-results", "catalog-" * Dates.format(now(), "yyyymmdd-HHMMSS")))))
    mkpath(joinpath(outdir, "cases"))
    mkpath(joinpath(outdir, "logs"))
    config["source_hash"] = CatalogSweep.source_hash()
    config["git_commit"] = CatalogSweep.git_commit()
    config["selection"] = getindex.(entries, "name")
    config["manifest_hash"] = CatalogSweep.file_hash(config["manifest"])
    config["julia_version"] = string(VERSION)
    project = Base.active_project()
    (project === nothing || !isfile(project)) && error("Run setup.jl first, then Julia with --project=test/catalog")
    config["project_hash"] = CatalogSweep.file_hash(project)
    manifest = joinpath(dirname(project), "Manifest.toml")
    config["environment_hash"] = isfile(manifest) ? CatalogSweep.file_hash(manifest) : "unresolved"
    config["blas_env"] = get(ENV, "OPENBLAS_NUM_THREADS", "unset")
    config["omp_env"] = get(ENV, "OMP_NUM_THREADS", "unset")
    # Input hashes prevent mixing replacement files into an existing run on resume.
    config["input_hashes"] = Dict(entry["name"] => (isfile(joinpath(config["matrix_dir"], entry["name"] * ".mtx")) ?
        CatalogSweep.file_hash(joinpath(config["matrix_dir"], entry["name"] * ".mtx")) : "missing") for entry in entries)
    configpath = joinpath(outdir, "settings.toml")
    if isfile(configpath)
        TOML.parsefile(configpath) == config || error("Settings, sources, environment or inputs changed; use a new RESULTS_DIR")
    else
        open(io -> TOML.print(io, config), configpath, "w")
    end
    println("Results: ", outdir)
    started = time()
    backends = config["methods"] == "all" ? ["gpu", "stiles"] : [config["methods"]]
    for entry in entries, backend in backends
        remaining = config["sweep_timeout"] - (time() - started)
        if remaining <= 0
            println("Sweep wall-time budget reached; resume with the same RESULTS_DIR.")
            break
        end
        name = entry["name"]
        csv = joinpath(outdir, "cases", name * "--" * backend * ".csv")
        existing = isfile(csv) ? CatalogSweep.read_csv(csv) : []
        finished = getindex.(existing, "method")
        methods = setdiff(CatalogSweep.case_methods(config, backend), finished)
        isempty(methods) && continue
        input = joinpath(config["matrix_dir"], name * ".mtx")
        if !isfile(input)
            println("MISSING: ", input)
            for method in methods
                row = CatalogSweep.base_row(entry, config, method)
                row["status"], row["detail"] = "MISSING", input
                CatalogSweep.append_row(csv, row)
            end
            CatalogSweep.merge_results(outdir)
            continue
        end
        limit = min(config[backend * "_timeout"], remaining)
        println("Running $name / $backend (wall-time limit $(round(limit; digits=1)) s)")
        command = `$(Base.julia_cmd()) --startup-file=no --project=$(dirname(project)) $worker $configpath $name $backend $csv $(join(methods, ','))`
        status, exitcode, elapsed = CatalogSweep.run_bounded(command, limit,
            joinpath(outdir, "logs", name * "--" * backend * ".log"))
        budget_stop = status == "TIMEOUT" && limit < config[backend * "_timeout"]
        budget_stop && (status = "SWEEP_LIMIT")
        completed = isfile(csv) ? getindex.(CatalogSweep.read_csv(csv), "method") : String[]
        for method in (budget_stop ? String[] : setdiff(methods, completed))
            row = CatalogSweep.base_row(entry, config, method)
            row["status"] = status == "DONE" ? "ERROR" : status
            row["detail"] = "Worker exit=$exitcode; backend wall time=$elapsed s; limit=$limit s. See log."
            row["elapsed_seconds"] = elapsed
            CatalogSweep.append_row(csv, row)
        end
        CatalogSweep.append_row(joinpath(outdir, "cases-summary.csv"),
            Dict("matrix" => name, "backend" => backend, "status" => status,
                 "elapsed_seconds" => elapsed, "exit_code" => exitcode);
            columns=["matrix", "backend", "status", "elapsed_seconds", "exit_code"])
        CatalogSweep.merge_results(outdir)
        if budget_stop
            println("Sweep budget reached; unfinished methods remain available for resume.")
            break
        end
    end
    rows = CatalogSweep.merge_results(outdir)
    println("Saved $(length(rows)) method rows to ", joinpath(outdir, "results.csv"))
end

abspath(PROGRAM_FILE) == abspath(@__FILE__) && main()
