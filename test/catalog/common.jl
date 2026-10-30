module CatalogSweep

using Dates, DelimitedFiles, SHA, TOML

const ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const GPU_METHODS = ["Pure_F32", "Pure_F64", "M1", "M2", "M4", "M6", "M7", "cuSOLVER_F32", "cuSOLVER_F64"]
const COLUMNS = split("matrix,group,application,n,nnz,density,method,status,detail,min_ms,median_ms,analysis_ms,factor_ms,repetition_ms,analysis_repetition_ms,timing_scope,factor_residual,quantization_error,forward_error,solve_residual,backward_error,scaled_blocks,max_scale,max_abs,min_abs_nonzero,dominant_fraction,worst_margin,probes,nrhs,repeats,seed,shift,stiles_cores,blas_threads,cpu_model,gpu,julia_version,git_commit,source_hash,matrix_sha256,elapsed_seconds", ',')

function read_csv(path)
    data, header = readdlm(path, ',', String; header=true)
    names = replace.(vec(header), '\ufeff' => "")
    [Dict(zip(names, data[i, :])) for i in axes(data, 1)]
end

csv_cell(x) = '"' * replace(replace(string(x), '\n' => ' ', '\r' => ' '), '"' => "\"\"") * '"'
function append_row(path, row; columns=COLUMNS)
    mkpath(dirname(path))
    newfile = !isfile(path) || filesize(path) == 0
    open(path, "a") do io
        newfile && println(io, join(csv_cell.(columns), ','))
        println(io, join((csv_cell(get(row, key, "")) for key in columns), ','))
        flush(io)
    end
end

function write_csv(path, rows; columns=COLUMNS)
    temporary = path * ".tmp"
    open(temporary, "w") do io
        println(io, join(csv_cell.(columns), ','))
        for row in rows
            println(io, join((csv_cell(get(row, key, "")) for key in columns), ','))
        end
    end
    mv(temporary, path; force=true)
end

function positive_env(name, default; integer=false)
    value = parse(integer ? Int : Float64, get(ENV, name, string(default)))
    isfinite(value) && value > 0 || error("$name must be finite and positive")
    value
end

function settings()
    methods = get(ENV, "METHODS", "all")
    methods in ("all", "gpu", "stiles") || error("METHODS must be all, gpu, or stiles")
    chosen = filter(!isempty, strip.(split(get(ENV, "GPU_METHODS", join(GPU_METHODS, ',')), ',')))
    all(in(GPU_METHODS), chosen) && !isempty(chosen) || error("Unknown or empty GPU_METHODS")
    Dict{String,Any}(
        "matrix_dir" => abspath(expanduser(get(ENV, "MATRIX_DIR", joinpath(ROOT, "data", "matrices")))),
        "manifest" => abspath(get(ENV, "MATRIX_MANIFEST", joinpath(@__DIR__, "matrices.csv"))),
        "methods" => methods, "gpu_methods" => chosen,
        "repeats" => positive_env("BENCH_REPEATS", 5; integer=true),
        "probes" => positive_env("ACCURACY_PROBES", 8; integer=true),
        "nrhs" => positive_env("ACCURACY_RHS", 4; integer=true),
        "stiles_cores" => positive_env("STILES_CORES", get(ENV, "SLURM_CPUS_PER_TASK", "1"); integer=true),
        "gpu_timeout" => positive_env("GPU_TIMEOUT_SECONDS", 600),
        "stiles_timeout" => positive_env("STILES_TIMEOUT_SECONDS", 300),
        "sweep_timeout" => positive_env("SWEEP_TIMEOUT_SECONDS", 10800),
        "gpu_memory_copies" => positive_env("GPU_MEMORY_COPIES", 5),
        "seed" => 20261029, "shift" => 0,
    )
end

function selection(config)
    rows = read_csv(config["manifest"])
    names = get(ENV, "REAL_MATRICES", "")
    if !isempty(names)
        requested = strip.(split(names, ','))
        missing = setdiff(requested, getindex.(rows, "name"))
        isempty(missing) || error("Names absent from manifest: $(join(missing, ','))")
        rows = filter(row -> row["name"] in requested, rows)
    end
    isempty(rows) && error("No matrices selected")
    all(row -> occursin(r"^[A-Za-z0-9_-]+$", row["name"]), rows) || error("Unsafe matrix name")
    length(unique(getindex.(rows, "name"))) == length(rows) || error("Duplicate matrix names")
    rows
end

file_hash(path) = open(path) do io
    bytes2hex(sha256(io))
end

function source_hash()
    paths = sort(vcat(filter(p -> endswith(p, ".jl"), readdir(@__DIR__; join=true)),
        filter(p -> endswith(p, ".jl"), readdir(joinpath(ROOT, "src"); join=true))))
    bytes2hex(sha256(join((relpath(p, ROOT) * ":" * file_hash(p) for p in paths), '\n')))
end

function git_commit()
    try
        strip(read(`git -C $ROOT rev-parse HEAD`, String))
    catch
        "unknown"
    end
end

function base_row(entry, config, method)
    Dict{String,Any}("matrix" => entry["name"], "group" => entry["group"],
        "application" => entry["application"], "n" => entry["n"], "method" => method,
        "status" => "ERROR", "probes" => config["probes"], "nrhs" => config["nrhs"],
        "repeats" => config["repeats"], "seed" => config["seed"], "shift" => 0,
        "stiles_cores" => config["stiles_cores"], "julia_version" => string(VERSION),
        "git_commit" => git_commit(), "source_hash" => source_hash())
end

case_methods(config, backend) = backend == "gpu" ? config["gpu_methods"] : ["sTiles_F64"]

"""A process deadline, including loading, compilation, I/O, warm-up and all repeats.
GNU timeout owns the Linux process group, so a blocked native call is killed too.
The fallback is used by the Windows supervisor tests.
"""
function run_bounded(command::Cmd, seconds, logfile)
    timeout = Sys.islinux() ? Sys.which("timeout") : nothing
    bounded = timeout === nothing ? command : `$timeout --signal=TERM --kill-after=5s $(string(seconds) * "s") $command`
    started = time()
    open(logfile, "a") do io
        println(io, "Worker started $(now()); wall-time limit=$seconds s")
        flush(io)
        process = run(pipeline(ignorestatus(bounded); stdout=io, stderr=io); wait=false)
        try
            if timeout !== nothing
                wait(process)
                return (process.exitcode in (124, 137) ? "TIMEOUT" : success(process) ? "DONE" : "ERROR",
                        process.exitcode, time() - started)
            end
            deadline = timedwait(() -> process_exited(process), seconds; pollint=0.05)
            if deadline == :timed_out
                kill(process)
                timedwait(() -> process_exited(process), 2; pollint=0.05) == :timed_out && kill(process, Base.SIGKILL)
                wait(process)
                return ("TIMEOUT", process.exitcode, time() - started)
            end
            wait(process)
            return (success(process) ? "DONE" : "ERROR", process.exitcode, time() - started)
        finally
            if process_running(process)
                kill(process)
                timedwait(() -> process_exited(process), 6; pollint=0.05) == :timed_out && kill(process, Base.SIGKILL)
                wait(process)
            end
        end
    end
end

function merge_results(outdir)
    rows = Dict{String,String}[]
    for path in sort(readdir(joinpath(outdir, "cases"); join=true))
        endswith(path, ".csv") && append!(rows, read_csv(path))
    end
    write_csv(joinpath(outdir, "results.csv"), rows)
    rows
end

end
