include("common.jl")
using .CatalogSweep
using LinearAlgebra, SparseArrays, Statistics, TOML, Random
include("matrix_io.jl")
include("accuracy.jl")

const CONFIG = TOML.parsefile(ARGS[1])
const MATRIX_NAME, BACKEND, CSV_PATH = ARGS[2:4]
const METHODS = split(ARGS[5], ',')

if BACKEND == "gpu"
    using CUDA
    include("solver.jl")
else
    using sTiles
end

function checkpoint(row)
    rows = isfile(CSV_PATH) ? CatalogSweep.read_csv(CSV_PATH) : Dict{String,Any}[]
    filter!(old -> old["method"] != row["method"], rows)
    CatalogSweep.write_csv(CSV_PATH, vcat(rows, [row]))
end

function timed_gpu(operation, reset; repeats=CONFIG["repeats"])
    reset(); operation(); CUDA.synchronize() # warm-up, excluded
    times = Float64[]
    for _ in 1:repeats
        reset()
        CUDA.synchronize()
        start = time_ns()
        operation()
        CUDA.synchronize()
        push!(times, (time_ns() - start) / 1e6)
    end
    times
end

function run_gpu(A, dense, method, inputs, row=Dict{String,Any}())
    S = CatalogSolver
    mixed = startswith(method, "M")
    work = nothing
    scales = (0, 1.0)
    quant_error = NaN
    if mixed
        depth = parse(Int, method[2:end])
        precisions = vcat(fill(Float16, depth), [Float32])
        build() = S.SymmMixedPrec(dense, 'L'; precisions=precisions)
        # Quantization diagnostic is separate and outside the timed operation.
        initial = build()
        scales = scale_stats(initial)
        decoded = decode_blocks(initial)
        quant_error = norm(decoded * CUDA.CuArray(inputs.omega) - CUDA.CuArray(inputs.aomega)) / norm(inputs.aomega)
        row["quantization_error"] = quant_error
        row["scaled_blocks"],row["max_scale"] = scales
        initial = decoded = nothing
        GC.gc(true); CUDA.reclaim()
        function operation()
            work = build()
            S.potrf_recursive!(work)
            nothing
        end
        # Allocation/GC policy is explicit: no retained factor from the preceding repeat.
        reset() = (work = nothing; GC.gc(true); CUDA.reclaim(); nothing)
        times = timed_gpu(operation, reset)
        factor = tril!(decode_blocks(work))
        scope = "mixed construction + factorization; excludes input conversion/transfer/reset/accuracy"
    else
        T = endswith(method, "F64") ? Float64 : Float32
        clean = T.(dense)
        quant_error = norm(Float64.(clean) * CUDA.CuArray(inputs.omega) - CUDA.CuArray(inputs.aomega)) / norm(inputs.aomega)
        row["quantization_error"] = quant_error
        work = copy(clean)
        operation = startswith(method, "cuSOLVER") ? (() -> CUDA.CUSOLVER.potrf!('L', work)) :
            (() -> S.potrf_recursive!(work, 4096))
        times = timed_gpu(operation, () -> copyto!(work, clean))
        # Release the reset copy and reuse F64 work for decoding. This reduces
        # peak memory outside timing on the 50k–64k legacy inputs.
        clean = nothing
        factor = T == Float64 ? tril!(work) : tril!(Float64.(work))
        work = nothing
        GC.gc(true); CUDA.reclaim()
        scope = "factorization; excludes input conversion/transfer/reset/accuracy"
    end
    println("$method minimum_ms=$(minimum(times)) median_ms=$(median(times)); checking accuracy")
    flush(stdout)
    diagonal = Array(diag(factor))
    all(x -> isfinite(x) && x > 0, diagonal) || error("Invalid factor diagonal")
    all(isfinite, factor) || error("Nonfinite decoded factor")
    omega = CUDA.CuArray(inputs.omega)
    action = Array(factor * (transpose(factor) * omega))
    solution = CUDA.CuArray(inputs.rhs)
    CUDA.CUBLAS.trsm!('L', 'L', 'N', 'N', 1.0, factor, solution)
    CUDA.CUBLAS.trsm!('L', 'L', 'T', 'N', 1.0, factor, solution)
    metrics = solution_metrics(A, Array(solution), inputs)
    metrics["factor_residual"] = probe_error(action, inputs)
    metrics["quantization_error"] = quant_error
    metrics["min_ms"], metrics["median_ms"] = minimum(times), median(times)
    metrics["repetition_ms"] = join(times, ';')
    metrics["scaled_blocks"], metrics["max_scale"] = scales
    metrics["timing_scope"] = scope
    metrics
end

function run_stiles(A, inputs)
    tiny = spdiagm(-1 => fill(-1.0, 5), 0 => fill(4.0, 6), 1 => fill(-1.0, 5))
    F = sTiles.analyze(tiny; cores=CONFIG["stiles_cores"], inverse=false)
    try
        sTiles.factorize!(F)
    finally
        close(F)
    end
    analysis, factor = Float64[], Float64[]
    metrics = Dict{String,Any}()
    for repeat in 1:CONFIG["repeats"]
        F = nothing
        try
            start = time_ns()
            F = sTiles.analyze(A; cores=CONFIG["stiles_cores"], inverse=false)
            push!(analysis, (time_ns() - start) / 1e6)
            start = time_ns()
            sTiles.factorize!(F)
            push!(factor, (time_ns() - start) / 1e6)
            if repeat == CONFIG["repeats"]
                # Native solve; no factor reconstruction is claimed for sTiles.
                solution = hcat((F \ inputs.rhs[:, j] for j in axes(inputs.rhs, 2))...)
                merge!(metrics, solution_metrics(A, solution, inputs))
            end
        finally
            F === nothing || close(F)
        end
    end
    best = argmin(analysis .+ factor)
    merge!(metrics, Dict("min_ms" => minimum(factor), "median_ms" => median(factor),
        "analysis_ms" => analysis[best], "factor_ms" => factor[best],
        "repetition_ms" => join(factor, ';'), "analysis_repetition_ms" => join(analysis, ';'),
        "timing_scope" => "numeric factorization; analysis/factor pair from minimum combined repetition"))
    metrics
end

function main()
    entries = CatalogSweep.read_csv(CONFIG["manifest"])
    entry = only(filter(row -> row["name"] == MATRIX_NAME, entries))
    path = joinpath(CONFIG["matrix_dir"], MATRIX_NAME * ".mtx")
    A = load_input(path, parse(Int, entry["n"]))
    if !isempty(get(entry,"expected_nnz",""))
        nnz(A)==parse(Int,entry["expected_nnz"]) || error("Input nonzero count differs from the earlier matrix: $(nnz(A))")
    end
    diagnostics = input_diagnostics(A)
    diagnostics["matrix_sha256"] = CatalogSweep.file_hash(path)
    diagnostics["matrix_sha256"] == CONFIG["input_hashes"][MATRIX_NAME] || error("Input changed after sweep started")
    diagnostics["blas_threads"] = BLAS.get_num_threads()
    diagnostics["cpu_model"] = first(Sys.cpu_info()).model
    inputs = accuracy_inputs(A, CONFIG)
    dense = nothing
    if BACKEND == "gpu"
        CUDA.functional() || error("CUDA is not functional on this node")
        diagnostics["gpu"] = CUDA.name(CUDA.device())
        # One FP64 dense input plus factors, decoding and solver workspace.
        needed = CONFIG["gpu_memory_copies"] * 8.0 * size(A, 1)^2
        if needed > CUDA.available_memory()
            for method in METHODS
                row = merge(CatalogSweep.base_row(entry, CONFIG, method), diagnostics)
                row["status"], row["detail"] = "SKIP_MEMORY", "Configured dense-copy memory estimate exceeds currently available GPU memory"
                checkpoint(row)
            end
            return
        end
        dense = CUDA.CuArray(Matrix(A))
    end
    for method in METHODS
        start = time()
        row = merge(CatalogSweep.base_row(entry, CONFIG, method), diagnostics)
        try
            metrics = BACKEND == "gpu" ? run_gpu(A, dense, method, inputs, row) : run_stiles(A, inputs)
            merge!(row, metrics)
            errors = [value for (key, value) in metrics if endswith(key, "error") || endswith(key, "residual")]
            row["status"] = all(isfinite, errors) ? "OK" : "NONFINITE"
        catch exception
            row["status"], row["detail"] = "ERROR", sprint(showerror, exception)
            showerror(stderr, exception, catch_backtrace()); println(stderr)
        end
        row["elapsed_seconds"] = time() - start
        checkpoint(row)
        println("$MATRIX_NAME $method $(row["status"])"); flush(stdout)
        if BACKEND == "gpu"
            GC.gc(true); CUDA.reclaim()
        end
    end
end

main()
