using KernelAbstractions

function run_manual_benchmark(func_to_benchmark, backend; min_time_s::Float64 = 2.0, min_iters::Int = 50)
    # warm up (for compiler)
    func_to_benchmark()
    KernelAbstractions.synchronize(backend)

    best_time_ns = 1e12 
    elapsed_time_ns = 0.0
    i = 0

    while elapsed_time_ns < min_time_s * 1e9 || i < min_iters
        KernelAbstractions.synchronize(backend)
        start_time = time_ns()

        func_to_benchmark()

        KernelAbstractions.synchronize(backend)
        end_time = time_ns()

        this_duration = end_time - start_time
        elapsed_time_ns += this_duration
        best_time_ns = min(best_time_ns, this_duration)
        i += 1
    end

    return best_time_ns
end