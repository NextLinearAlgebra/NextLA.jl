# Download the selected *catalog versions* directly on ORCD; no CUDA needed.
# julia test/catalog/download.jl [--check-only]
include("common.jl")
using .CatalogSweep
using Downloads, Dates

const DOWNLOAD_BASE = "https://esmail.r-inla-download.org/mtx"

function check_matrix_file(path, entry; count_entries=true)
    open(path) do io
        banner = split(lowercase(strip(readline(io))))
        length(banner) == 5 && banner[1:3] == ["%%matrixmarket", "matrix", "coordinate"] ||
            error("Not a MatrixMarket coordinate matrix (possibly an HTML download error)")
        banner[4] in ("real", "integer") || error("Expected numeric real matrix values")
        banner[5] in ("symmetric", "general") || error("Unsupported symmetry tag")
        header = ""
        while !eof(io)
            line = strip(readline(io))
            (isempty(line) || startswith(line, '%')) && continue
            header = line
            break
        end
        dimensions = parse.(Int, split(header))
        length(dimensions) == 3 || error("Missing matrix dimensions")
        n = parse(Int, entry["n"])
        dimensions[1:2] == [n,n] || error("Dimensions do not match catalog: $dimensions")
        dimensions[3] == parse(Int, entry["nnz_lower"]) ||
            error("Stored entry count does not match catalog: $(dimensions[3])")
        if count_entries
            count = 0
            for line in eachline(io)
                trimmed = strip(line)
                (isempty(trimmed) || startswith(trimmed, '%')) && continue
                count += 1
            end
            count == dimensions[3] || error("Incomplete file: $count entries, expected $(dimensions[3])")
        end
    end
    true
end

function download_main(args=ARGS)
    config = CatalogSweep.settings()
    entries = CatalogSweep.selection(config)
    default_dir = joinpath(homedir(), "orcd", "scratch", "nextla-matrices")
    destination = abspath(expanduser(get(ENV, "MATRIX_DIR", default_dir)))
    mkpath(destination)
    check_only = "--check-only" in args
    timeout = CatalogSweep.positive_env("DOWNLOAD_TIMEOUT_SECONDS", 600)
    retries = CatalogSweep.positive_env("DOWNLOAD_RETRIES", 3; integer=true)
    failures = String[]
    println("Matrix directory: ", destination)
    for entry in entries
        name = entry["name"]
        path = joinpath(destination, name * ".mtx")
        url = "$DOWNLOAD_BASE/$(entry["group"])/$name.mtx"
        row = Dict{String,Any}("matrix" => name, "url" => url, "timestamp" => string(now()))
        valid = false
        if isfile(path)
            try
                valid = check_matrix_file(path, entry)
            catch exception
                println("INVALID $name: ", sprint(showerror, exception))
            end
        end
        if valid
            row["status"] = "EXISTING"
            println("EXISTING $name")
        elseif check_only
            row["status"], row["detail"] = "MISSING_OR_INVALID", path
            push!(failures, name)
            println("MISSING_OR_INVALID $name")
        else
            temporary = path * ".part"
            row["status"] = "FAILED"
            for attempt in 1:retries
                println("DOWNLOAD $name ($attempt/$retries)"); flush(stdout)
                try
                    Downloads.download(url, temporary; timeout=timeout)
                    check_matrix_file(temporary, entry)
                    mv(temporary, path; force=true)
                    row["status"] = "DOWNLOADED"
                    valid = true
                    break
                catch exception
                    row["detail"] = sprint(showerror, exception)
                    # A missing catalog file will not be fixed by retrying it.
                    occursin("404", row["detail"]) && break
                    attempt < retries && sleep(min(attempt, 3))
                end
            end
            if !valid
                push!(failures, name)
                println("FAILED $name: ", row["detail"])
            end
        end
        if valid
            row["bytes"], row["sha256"] = filesize(path), CatalogSweep.file_hash(path)
            pop!(row, "detail", nothing)
        end
        CatalogSweep.append_row(joinpath(destination, "download-status.csv"), row;
            columns=["timestamp", "matrix", "status", "url", "bytes", "sha256", "detail"])
    end
    if !isempty(failures)
        println("Still unavailable/invalid: ", join(failures, ", "))
        println("Successful downloads are retained. Rerun to retry only missing/invalid files.")
        return false
    end
    println("All $(length(entries)) selected files validated.")
    true
end

if abspath(PROGRAM_FILE) == abspath(@__FILE__)
    download_main() || exit(1)
end
