# NextLA tests, run by ParallelTestRunner.jl: each test runs in its own Julia
# process, so a test that fails or crashes does not stop the others, and any
# set of tests can be run by name (names are matched with startswith; a name
# starting with ! is left out).
#
#   Pkg.test("NextLA")                                   # everything
#   Pkg.test("NextLA"; test_args=`--list`)               # the test names
#   Pkg.test("NextLA"; test_args=`mxp`)                  # mixed precision only
#   Pkg.test("NextLA"; test_args=`tlr`)                  # tile low-rank only
#   Pkg.test("NextLA"; test_args=`mxp/blas/trsm`)        # one routine
#   Pkg.test("NextLA"; test_args=`dense/lapack/qr`)      # one family of dense tests
#   Pkg.test("NextLA"; test_args=`--jobs=1 mxp`)         # one file at a time
#   Pkg.test("NextLA"; test_args=`--backends=CPU mxp`)   # mixed precision on the CPU only
#
# --backends=CPU (or CPU,CUDA, ...) limits every test to those backends; see
# test/gpu_backends.jl. Workers start with one thread unless JULIA_NUM_THREADS
# is set. test/README.md has the details.

using NextLA
using ParallelTestRunner

const TESTDIR = @__DIR__

# --backends=<names>: the backends every test runs on.
args = parse_args(ARGS; custom=["backends"])
const BACKENDS = let flag = args.custom["backends"]
    if flag === nothing                 # not given: every available backend
        ""
    elseif something(flag) === nothing  # given without a value
        error("--backends needs a value, e.g. --backends=CPU or --backends=CPU,CUDA")
    else
        String(something(flag))
    end
end
const WANTED = isempty(BACKENDS) ? String[] : lowercase.(strip.(split(BACKENDS, ',')))
# The names test/gpu_backends.jl knows (its backend_defs, plus CPU). Checked here,
# so a misspelt name stops the run once instead of failing every test.
const KNOWN_BACKENDS = ["cpu", "cuda", "amdgpu", "oneapi", "metal"]
let unknown = setdiff(WANTED, KNOWN_BACKENDS)
    isempty(unknown) || error("--backends: unknown ", join(unknown, ", "),
                              "; known: CPU, CUDA, AMDGPU, oneAPI, Metal")
end

# Evaluated at the top level of each test's own module before the test (the
# test itself runs inside a @testset, a local scope, so constants go here).
const init_code = quote
    using NextLA, Test, LinearAlgebra, Random
    const TEST_BACKENDS_WANTED = $BACKENDS
end

# One test per file under test/dense, test/mxp and test/tlr, named by its path
# without ".jl", e.g. "dense/lapack/qr/geqrt" or "mxp/blas/trsm/rectrxm_mixed".
# Each starts with the helpers of its part of the suite (test/setup/dense.jl,
# test/setup/mxp.jl or test/setup/tlr.jl); the helper files themselves are not
# tests.
const SETUP = Dict(
    "dense" => [joinpath(TESTDIR, "setup", "dense.jl")],
    "mxp"   => [joinpath(TESTDIR, "gpu_backends.jl"), joinpath(TESTDIR, "setup", "mxp.jl")],
    "tlr"   => [joinpath(TESTDIR, "gpu_backends.jl"), joinpath(TESTDIR, "setup", "tlr.jl")],
)
# A folder under test/ with test files that is not a part above would never run;
# stop instead, so a new folder cannot be skipped without anyone noticing.
for d in readdir(TESTDIR)
    p = joinpath(TESTDIR, d)
    (isdir(p) && d != "setup" && !haskey(SETUP, d)) || continue
    any(f -> endswith(f, ".jl"), (f for (_, _, fs) in walkdir(p) for f in fs)) &&
        error("test/$d has test files but is not a part in SETUP (test/runtests.jl), so they would not run")
end

testsuite = Dict{String, Expr}()
for (part, setup) in SETUP, (name, body) in find_tests(joinpath(TESTDIR, part))
    testsuite["$part/$name"] = quote
        $((:(include($f)) for f in setup)...)
        $body
    end
end

# ParallelTestRunner picks one test process per 2 GiB of free host memory, up to
# one per CPU thread, but each process that uses CUDA also holds its own context
# and memory pool on the device. As CUDA.jl's own test/runtests.jl does, cap that
# number by the device's free memory (read through NVML when it can, so this
# process creates no CUDA context); an integrated GPU shares host memory, so its
# budget adds the host part. Only when CUDA is installed and asked for; --jobs=N
# overrides it, and any failure here leaves ParallelTestRunner's default.
const GPU_MEMORY_PER_TEST = 2 * 2^30    # context + pool of one test process (estimate)
const HOST_MEMORY_PER_TEST = 2 * 2^30   # ParallelTestRunner's default per process
function cuda_job_cap()
    CUDA = getfield(Main, :CUDA)
    CUDA.functional() || return typemax(Int)
    dev = first(CUDA.devices())
    free = try   # NVML first: it needs no CUDA context in this process
        mig = CUDA.uuid(dev) != CUDA.parent_uuid(dev)
        Int(CUDA.NVML.memory_info(CUDA.NVML.Device(CUDA.uuid(dev); mig)).free)
    catch   # no NVML, or it does not report memory (some integrated GPUs)
        CUDA.device!(() -> Int(CUDA.free_memory()), dev)
    end
    integrated = CUDA.attribute(dev, CUDA.DEVICE_ATTRIBUTE_INTEGRATED) == 1
    budget, per_test = integrated ? (min(free, Int(Sys.free_memory())), GPU_MEMORY_PER_TEST + HOST_MEMORY_PER_TEST) :
                                    (free, GPU_MEMORY_PER_TEST)
    jobs = max(1, budget ÷ per_test)
    @info "Test processes capped by GPU memory" device = CUDA.name(dev) integrated free = Base.format_bytes(free) jobs
    return jobs
end
const MAX_DEFAULT_JOBS = if (isempty(WANTED) || "cuda" in WANTED) && Base.find_package("CUDA") !== nothing
    try
        @eval Main using CUDA
        Base.invokelatest(cuda_job_cap)
    catch err
        @warn "Could not read free GPU memory; using ParallelTestRunner's default number of jobs" err
        typemax(Int)
    end
else
    typemax(Int)
end

# Keep a separate record of test times and last failures for each --backends
# choice (ParallelTestRunner's history_key), so a CPU-only run does not mix with a
# GPU run in the ordering or in the marks that --list shows.
const HISTORY_KEY = isempty(WANTED) ? nothing : join(sort(WANTED), ",")

runtests(NextLA, args; testsuite, init_code, max_default_jobs = MAX_DEFAULT_JOBS, history_key = HISTORY_KEY)
